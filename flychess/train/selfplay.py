"""Stage 2: self-play game generation, replay buffer and the self-play training loop (SPEC §6).

Games are generated with :class:`~flychess.train.mcts.BatchedMCTS` — all games of an iteration are
searched in lockstep so every simulation step is one GPU forward — with Dirichlet noise at the root,
temperature 1 for the first ``cfg.temperature_plies`` plies and argmax afterwards, no resignation, and
``cfg.max_game_plies`` as a hard cap (counted as a draw). The subtree below the chosen move is kept
between plies. With ``cfg.selfplay_workers > 1`` the games are split across that many threads, each
running its own lockstep batch on the shared model: PyTorch releases the GIL inside the CUDA kernels,
so one thread's tree work overlaps the other threads' GPU time.

Training targets per position: quantised planes (exactly the shard format of SPEC §5), ``pi`` = the
root visit distribution and ``z`` = the game result from the mover's perspective. The replay buffer
stores ``pi`` sparsely (top-``k`` visited moves) together with the legal-move indices so that the
policy loss can mask illegal moves like the imitation stage does.

The fly brain is the only thing choosing moves in these games.
"""
from __future__ import annotations

import importlib
import inspect
import io
import logging
import math
import time
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F

from flychess.chessenv.encoding import (
    FLAT_INPUT,
    NUM_MOVES,
    NUM_PLANES,
    board_features,
    index_to_move,
    planes_from_features,
)
from flychess.connectome.graph import BrainGraph
from flychess.data.shards import planes_to_float
from flychess.model.flybrain import FlyBrain, masked_policy_log_softmax
from flychess.train.mcts import (
    BatchedMCTS,
    Node,
    advance_root,
    result_string,
    select_move_index,
    terminal_value,
)
from flychess.train.metrics import MetricsLogger

log = logging.getLogger(__name__)

_RESULT_Z = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def _get(cfg: Any, name: str, default: Any) -> Any:
    """``cfg.name`` with a default so that a minimal config object (or a dict) works too."""
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


# ------------------------------------------------------------------------------------------------
# Game record
# ------------------------------------------------------------------------------------------------
@dataclass
class Game:
    """One finished self-play game.

    ``planes`` uint8 (T, 20, 8, 8) shard-quantised · ``pi`` float16 (T, 4168) root visit distribution ·
    ``z`` int8 (T,) result from each mover's perspective · ``legal`` list of int16 arrays (legal move
    indices per position) · ``moves`` the UCI moves · ``pgn`` · ``result`` · ``plies``.
    """

    planes: np.ndarray
    pi: np.ndarray
    z: np.ndarray
    legal: list[np.ndarray]
    moves: list[str]
    pgn: str
    result: str
    plies: int
    capped: bool = False
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.plies


def _pgn(board: chess.Board, result: str, headers: dict[str, str] | None = None) -> str:
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "fly-chess self-play"
    game.headers["White"] = "fly"
    game.headers["Black"] = "fly"
    game.headers["Result"] = result
    for k, v in (headers or {}).items():
        game.headers[k] = v
    return str(game)


class _Live:
    """Bookkeeping of one in-progress self-play game."""

    __slots__ = ("board", "feats", "index", "legal", "moves", "pi", "root", "turns")

    def __init__(self, index: int) -> None:
        self.index = index
        self.board = chess.Board()
        self.root: Node | None = None
        self.feats: list[np.ndarray] = []
        self.pi: list[tuple[np.ndarray, np.ndarray]] = []   # (legal_idx, probs over legal)
        self.legal: list[np.ndarray] = []
        self.turns: list[bool] = []
        self.moves: list[str] = []

    def finish(self, result: str, capped: bool) -> Game:
        T = len(self.feats)
        planes = planes_from_features(np.stack(self.feats), quantized=True) if T else np.zeros((0, NUM_PLANES, 8, 8), np.uint8)
        pi = np.zeros((T, NUM_MOVES), dtype=np.float16)
        for t, (li, p) in enumerate(self.pi):
            pi[t, li] = p
        z_white = _RESULT_Z[result]
        z = np.array([z_white if turn else -z_white for turn in self.turns], dtype=np.int8)
        return Game(planes=planes, pi=pi, z=z, legal=[li.astype(np.int16) for li in self.legal], moves=list(self.moves),
                    pgn=_pgn(self.board, result), result=result, plies=T, capped=capped)


def _play_batch(mcts: BatchedMCTS, indices: Sequence[int], cfg: Any, rng: np.random.Generator,
                on_game: Callable[[Game], None] | None) -> list[Game]:
    """Play ``len(indices)`` games to the end in lockstep; returns them in ``indices`` order."""
    temperature_plies = int(_get(cfg, "temperature_plies", 30))
    max_plies = int(_get(cfg, "max_game_plies", 300))
    live = [_Live(i) for i in indices]
    done: dict[int, Game] = {}
    active = list(live)
    while active:
        boards = [g.board for g in active]
        results = mcts.search(boards, roots=[g.root for g in active], add_noise=True)
        still: list[_Live] = []
        for g, res in zip(active, results, strict=True):
            ply = len(g.board.move_stack)
            finished: Game | None = None
            if res.terminal:  # cannot normally happen (we test after each push) but keeps the loop total
                finished = g.finish(result_string(g.board), capped=False)
            else:
                legal = res.root.moves.astype(np.int64)
                pol = res.visits[legal]
                pol = pol / pol.sum()
                g.feats.append(board_features(g.board))
                g.pi.append((legal, pol.astype(np.float32)))
                g.legal.append(legal)
                g.turns.append(g.board.turn)
                temperature = 1.0 if ply < temperature_plies else 0.0
                idx = res.best_index() if temperature <= 0 else select_move_index(res.visits, temperature, rng)
                move = index_to_move(idx, g.board)
                g.root = advance_root(res.root, move, g.board)
                g.board.push(move)
                g.moves.append(move.uci())
                if terminal_value(g.board) is not None:
                    finished = g.finish(result_string(g.board), capped=False)
                elif len(g.board.move_stack) >= max_plies:
                    finished = g.finish("1/2-1/2", capped=True)
            if finished is None:
                still.append(g)
            else:
                done[g.index] = finished
                if on_game is not None:
                    on_game(finished)
        active = still
    return [done[i] for i in indices]


def generate_games(
    model: FlyBrain,
    graph: BrainGraph | None,
    cfg: Any,
    n_games: int,
    device: str | torch.device | None = None,
    rng: np.random.Generator | None = None,
    on_game: Callable[[Game], None] | None = None,
    sims: int | None = None,
    workers: int | None = None,
) -> list[Game]:
    """Self-play ``n_games`` games with the fly brain (batched MCTS) and return them.

    ``graph`` is accepted for the shared contract (the model already carries the connectome) and may be
    ``None``. ``sims`` / ``workers`` override ``cfg.mcts_sims`` / ``cfg.selfplay_workers``.
    """
    rng = rng if rng is not None else np.random.default_rng(int(_get(cfg, "seed", 0)))
    device = torch.device(device) if device is not None else next(model.parameters()).device
    sims = int(_get(cfg, "mcts_sims", 64) if sims is None else sims)
    workers = int(_get(cfg, "selfplay_workers", 1) if workers is None else workers)
    workers = max(1, min(workers, n_games)) if n_games > 0 else 1
    was_training = model.training
    model.eval()
    model.structure()  # build the device-side CSR views once, before any thread touches them
    try:
        make = lambda seed: BatchedMCTS(
            model, device, sims=sims, c_puct=float(_get(cfg, "c_puct", 1.5)),
            dirichlet_alpha=float(_get(cfg, "dirichlet_alpha", 0.3)), dirichlet_eps=float(_get(cfg, "dirichlet_eps", 0.25)),
            rng=np.random.default_rng(seed),
        )
        if workers == 1:
            mcts = make(int(rng.integers(2**31)))
            return _play_batch(mcts, list(range(n_games)), cfg, rng, on_game)
        chunks = [list(range(n_games))[w::workers] for w in range(workers)]
        seeds = rng.integers(2**31, size=2 * workers)
        games: list[Game | None] = [None] * n_games
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="selfplay") as pool:
            futures = [pool.submit(_play_batch, make(int(seeds[w])), chunks[w], cfg,
                                   np.random.default_rng(int(seeds[workers + w])), on_game)
                       for w in range(workers) if chunks[w]]
            for w, fut in enumerate(futures):
                for i, game in zip(chunks[w], fut.result(), strict=True):
                    games[i] = game
        return [g for g in games if g is not None]
    finally:
        model.train(was_training)


# ------------------------------------------------------------------------------------------------
# Replay buffer
# ------------------------------------------------------------------------------------------------
class ReplayBuffer:
    """Ring buffer of self-play positions.

    Stores ``planes`` (uint8, shard format), ``pi`` sparsely as the top-``k`` visited moves (index +
    float16 probability, renormalised), the legal move indices (``int16`` padded with -1, at most
    ``lmax`` — moves carrying ``pi`` mass are kept first) and ``z`` (int8).
    """

    def __init__(self, size: int, k: int = 64, lmax: int = 128, seed: int = 0) -> None:
        self.size = int(size)
        self.k = int(k)
        self.lmax = int(lmax)
        self.planes = np.zeros((self.size, NUM_PLANES, 8, 8), dtype=np.uint8)
        self.pi_idx = np.zeros((self.size, self.k), dtype=np.int16)
        self.pi_val = np.zeros((self.size, self.k), dtype=np.float16)
        self.legal = np.full((self.size, self.lmax), -1, dtype=np.int16)
        self.z = np.zeros(self.size, dtype=np.int8)
        self.pos = 0
        self.count = 0
        self.games = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.count

    def add(self, game: Game) -> int:
        """Append every position of ``game``; returns the number of positions added."""
        T = game.plies
        if T == 0:
            return 0
        pi = game.pi.astype(np.float32)
        top = np.argsort(-pi, axis=1)[:, : self.k]                        # (T, k)
        val = np.take_along_axis(pi, top, axis=1)
        val = val / np.maximum(val.sum(axis=1, keepdims=True), 1e-8)
        legal = np.full((T, self.lmax), -1, dtype=np.int16)
        for t in range(T):
            li = np.asarray(game.legal[t], dtype=np.int64)
            if li.shape[0] > self.lmax:  # keep the moves that carry probability mass first
                order = np.argsort(-pi[t, li], kind="stable")
                li = li[order][: self.lmax]
            legal[t, : li.shape[0]] = li
        self.games += 1
        for t in range(T):  # ring write (positions are contiguous unless we wrap)
            p = self.pos
            self.planes[p] = game.planes[t]
            self.pi_idx[p] = top[t]
            self.pi_val[p] = val[t]
            self.legal[p] = legal[t]
            self.z[p] = game.z[t]
            self.pos = (p + 1) % self.size
        self.count = min(self.size, self.count + T)
        return T

    def sample(self, batch: int, rng: np.random.Generator | None = None) -> dict[str, torch.Tensor]:
        """Uniform sample: ``planes float32 (B, 1280)``, ``pi float32 (B, 4168)``, ``z float32 (B,)``,
        ``legal_mask bool (B, 4168)``."""
        if self.count == 0:
            raise ValueError("replay buffer is empty")
        rng = rng if rng is not None else self.rng
        idx = rng.integers(0, self.count, size=int(batch))
        planes = planes_to_float(self.planes[idx]).reshape(len(idx), FLAT_INPUT)
        pi = np.zeros((len(idx), NUM_MOVES), dtype=np.float32)
        rows = np.arange(len(idx))[:, None]
        pi_idx = self.pi_idx[idx].astype(np.int64)
        np.add.at(pi, (np.broadcast_to(rows, pi_idx.shape), pi_idx), self.pi_val[idx].astype(np.float32))
        legal = self.legal[idx].astype(np.int64)
        mask = np.zeros((len(idx), NUM_MOVES), dtype=bool)
        ok = legal >= 0
        mask[np.broadcast_to(rows, legal.shape)[ok], legal[ok]] = True
        mask |= pi > 0
        return {
            "planes": torch.from_numpy(planes),
            "pi": torch.from_numpy(pi),
            "z": torch.from_numpy(self.z[idx].astype(np.float32)),
            "legal_mask": torch.from_numpy(mask),
        }


# ------------------------------------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------------------------------------
def selfplay_loss(logits: torch.Tensor, value: torch.Tensor, pi: torch.Tensor, z: torch.Tensor,
                  legal_mask: torch.Tensor | None = None, value_weight: float = 1.0
                  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``-(pi · log_softmax(logits)) + value_weight · mse(value, z)`` with illegal moves masked out.

    Returns ``(loss, metrics)``; metrics are detached device tensors (``loss, policy_loss, value_loss,
    top1, top3`` where top-k compares the network's argmax with the search's most visited move).
    """
    logp = masked_policy_log_softmax(logits, legal_mask)
    if legal_mask is not None:
        logp = logp.masked_fill(~legal_mask.to(torch.bool), 0.0)   # pi is 0 there: avoid 0 * -inf
    policy_loss = -(pi.float() * logp).sum(dim=-1).mean()
    v = value.float().view(-1)
    value_loss = F.mse_loss(v, z.float().view(-1))
    loss = policy_loss + value_weight * value_loss
    with torch.no_grad():
        target = pi.argmax(dim=-1)
        top3 = logp.masked_fill(~legal_mask.to(torch.bool), float("-inf")).topk(3, dim=-1).indices if legal_mask is not None else logp.topk(3, dim=-1).indices
        hit = top3 == target.unsqueeze(1)
        metrics = {"loss": loss.detach(), "policy_loss": policy_loss.detach(), "value_loss": value_loss.detach(),
                   "top1": hit[:, 0].float().mean(), "top3": hit.any(dim=1).float().mean()}
    return loss, metrics


# ------------------------------------------------------------------------------------------------
# Evaluation against the previous iteration (the fly plays the fly) and against a random opponent
# ------------------------------------------------------------------------------------------------
def elo_from_score(score: float) -> float:
    """Elo difference implied by a score fraction (clamped to ±800)."""
    s = min(max(score, 0.01), 0.99)
    return float(-400.0 * math.log10(1.0 / s - 1.0))


def play_match(player_a: Any, player_b: Any, games: int, max_plies: int = 300, seed: int = 0,
               on_game: Callable[[int, str, str], None] | None = None, keep_pgns: int = 3,
               start_fens: Sequence[str] | None = None) -> dict[str, Any]:
    """``games`` games between two ``Player``-protocol objects (colours alternate) → W/D/L for A + Elo diff.

    Same signature as ``flychess.eval.elo.play_match`` (used when that module is unavailable): game ``i``
    starts from ``start_fens[i % len(start_fens)]`` when given, ``seed`` reseeds players exposing ``rng``,
    ``on_game(index, result, pgn)`` fires per game and the first ``keep_pgns`` PGNs are returned.
    """
    for k, player in enumerate((player_a, player_b)):
        if getattr(player, "rng", None) is not None:
            player.rng = np.random.default_rng(seed * 2 + k)
    wins = draws = losses = 0
    pgns: list[str] = []
    for g in range(games):
        board = chess.Board(start_fens[g % len(start_fens)]) if start_fens else chess.Board()
        a_white = g % 2 == 0
        while terminal_value(board) is None and len(board.move_stack) < max_plies:
            player = player_a if (board.turn == chess.WHITE) == a_white else player_b
            board.push(player.choose(board))
        result = result_string(board)
        result = result if result != "*" else "1/2-1/2"
        z = _RESULT_Z[result]
        if not a_white:
            z = -z
        wins += z > 0
        draws += z == 0
        losses += z < 0
        pgn = ""
        if g < keep_pgns or on_game is not None:
            pgn = _pgn(board, result, {"White": player_a.name if a_white else player_b.name,
                                       "Black": player_b.name if a_white else player_a.name, "Round": str(g + 1)})
            if g < keep_pgns:
                pgns.append(pgn)
        if on_game is not None:
            on_game(g, result, pgn)
    score = (wins + 0.5 * draws) / max(1, games)
    return {"player_a": player_a.name, "player_b": player_b.name, "games": games, "wins": wins, "draws": draws,
            "losses": losses, "score": score, "elo_estimate": elo_from_score(score), "pgns": pgns,
            "pgn": pgns[0] if pgns else ""}


def opening_fens(games: Sequence[Game], n: int, rng: np.random.Generator, min_ply: int = 8, max_ply: int = 12
                 ) -> list[str]:
    """Up to ``n`` distinct non-terminal positions taken from ``games`` (the FEN after ``min_ply``…``max_ply``
    plies of each game, capped at the game's penultimate ply) — the openings of an evaluation match.

    Deterministic players (``fly``) always play the same game from the start position, so a match
    between two of them needs distinct openings for its games to carry any information.
    """
    out: list[str] = []
    seen: set[str] = set()
    for gi in rng.permutation(len(games)):
        g = games[int(gi)]
        if g.plies <= 0:
            continue
        k = int(min(rng.integers(min_ply, max_ply + 1), g.plies - 1))
        board = chess.Board()
        for uci in g.moves[:k]:
            board.push_uci(uci)
        if terminal_value(board) is not None:  # cannot happen for k < plies, but never hand out a finished game
            continue
        fen = board.fen()
        if fen not in seen:
            seen.add(fen)
            out.append(fen)
        if len(out) >= n:
            break
    return out


def paired_start_fens(openings: Sequence[str], games: int) -> list[str] | None:
    """Every opening twice in a row (game ``2j`` / ``2j+1`` swap colours), cycling if there are too few."""
    if not openings or games <= 0:
        return None
    return [openings[(i // 2) % len(openings)] for i in range(games)]


def _match_fn() -> Callable[..., Any]:
    """``flychess.eval.elo.play_match`` when importable, else the built-in matcher above."""
    try:
        elo_mod = importlib.import_module("flychess.eval.elo")
        fn = getattr(elo_mod, "play_match", None)
        if callable(fn):
            return fn
    except Exception as e:  # noqa: BLE001 - the eval package belongs to another module; anything odd → fallback
        log.debug("flychess.eval.elo.play_match unavailable (%r): using the built-in match", e)
    return play_match


def _run_match(player_a: Any, player_b: Any, games: int, max_plies: int, start_fens: Sequence[str] | None,
               seed: int) -> dict[str, Any] | None:
    """Play the match with the shared evaluator (or the fallback) and normalise the result to a dict."""
    match_fn = _match_fn()
    try:
        out = match_fn(player_a, player_b, games, max_plies=max_plies, seed=seed, start_fens=start_fens,
                       keep_pgns=games)
    except TypeError:  # an evaluator with a smaller signature
        out = match_fn(player_a, player_b, games)
    if hasattr(out, "to_dict"):
        pgns = list(getattr(out, "pgns", []) or [])
        out = out.to_dict()
        out["pgns"] = pgns
    if not isinstance(out, dict) or "wins" not in out:
        warnings.warn("play_match returned an unexpected result; skipping the elo record", stacklevel=2)
        return None
    out["openings"] = len(set(start_fens)) if start_fens else 1
    return out


def _evaluate_vs_previous(model: FlyBrain, prev_state: dict[str, torch.Tensor], graph: BrainGraph, cfg: Any,
                          device: torch.device, games: int, start_fens: Sequence[str] | None = None,
                          seed: int = 0) -> dict[str, Any] | None:
    """Current model vs the previous iteration's weights, both playing as ``fly`` (policy + 1-ply value).

    ``fly`` is deterministic, so ``start_fens`` (see :func:`opening_fens` / :func:`paired_start_fens`)
    must vary the openings for the match to be more than one game per colour.
    """
    try:
        from flychess.play.engine import FlyEngine
    except ImportError as e:  # pragma: no cover
        warnings.warn(f"evaluation skipped: {e}", stacklevel=2)
        return None
    prev = FlyBrain(graph, model.config).to(device)
    prev.load_state_dict(prev_state)
    prev.eval()
    current = FlyEngine(model, graph, device).as_player("fly")
    previous = FlyEngine(prev, graph, device).as_player("fly")
    current.name, previous.name = "current", "previous"
    try:
        return _run_match(current, previous, games, int(_get(cfg, "max_game_plies", 300)), start_fens, seed)
    finally:
        del prev


def _evaluate_vs_random(model: FlyBrain, graph: BrainGraph, cfg: Any, device: torch.device, games: int,
                        start_fens: Sequence[str] | None = None, seed: int = 0) -> dict[str, Any] | None:
    """The current model (``fly``) against ``flychess.eval.opponents.RandomPlayer`` — an *opponent*, SPEC §6."""
    try:
        from flychess.eval.opponents import RandomPlayer
        from flychess.play.engine import FlyEngine
    except ImportError as e:
        warnings.warn(f"evaluation vs random skipped: {e}", stacklevel=2)
        return None
    current = FlyEngine(model, graph, device).as_player("fly")
    current.name = "current"
    return _run_match(current, RandomPlayer(seed=seed, name="random"), games,
                      int(_get(cfg, "max_game_plies", 300)), start_fens, seed)


# ------------------------------------------------------------------------------------------------
# Stage 2 loop
# ------------------------------------------------------------------------------------------------
def _gpu_mem_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device)) / 1e9
    return 0.0


def _sample_activity(model: FlyBrain, board: chess.Board, device: torch.device, n: int = 2048
                     ) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(planes_from_features(board_features(board)[None]).reshape(1, -1)).to(device)
    with torch.no_grad():
        _, _, h = model(x, return_activity=True)
    h = h[0].float().cpu().numpy()
    idx = np.linspace(0, h.shape[0] - 1, num=min(n, h.shape[0]), dtype=np.int64)
    return idx, h[idx]


def completed_iterations(run_dir: str | Path, step: int) -> int:
    """Self-play iterations already finished at ``step`` according to the run's metrics.

    Every finished iteration writes a ``status`` record ``{stage: 'selfplay', iteration: n}``; the
    largest ``n`` at or before ``step`` is returned (0 when the run has no such record).
    """
    try:
        from flychess.train.metrics import read_metrics
    except ImportError:  # pragma: no cover
        return 0
    done = 0
    for rec in read_metrics(run_dir, kinds=["status"]):
        if rec.get("stage") == "selfplay" and rec.get("iteration") is not None and int(rec.get("step", -1)) <= step:
            done = max(done, int(rec["iteration"]))
    return done


def _resume_optim_state(run_dir: Path, step: int) -> dict | None:
    """The optimizer state of ``<run_dir>/latest.pt`` when it is a self-play checkpoint saved at ``step``."""
    latest = run_dir / "latest.pt"
    if not latest.exists():
        return None
    try:
        ck = torch.load(latest, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 - a corrupt/foreign checkpoint must not stop the stage
        log.warning("could not read %s for the optimizer state: %r", latest, e)
        return None
    if ck.get("stage") == "selfplay" and int(ck.get("step", -1)) == int(step) and isinstance(ck.get("optim"), dict):
        return ck["optim"]
    return None


def _save(save_checkpoint: Callable[..., Path], step: int, stage: str, optim: torch.optim.Optimizer) -> Path:
    """``save_checkpoint(step, stage, optim=...)`` when the callback accepts an optimizer, else without it."""
    try:
        params = inspect.signature(save_checkpoint).parameters
        takes_optim = "optim" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        takes_optim = False
    if takes_optim:
        return save_checkpoint(step, stage, optim=optim)
    return save_checkpoint(step, stage)


def run_selfplay_stage(
    model: FlyBrain,
    graph: BrainGraph,
    cfg: Any,
    logger: MetricsLogger,
    start_step: int,
    save_checkpoint: Callable[[int, str], Path],
    device: str | torch.device | None = None,
    *,
    optim_state: dict | None = None,
    completed_iters: int | None = None,
) -> int:
    """Alternate self-play generation and training for ``cfg.selfplay_iters`` iterations; returns the final step.

    Each iteration: generate ``cfg.selfplay_games_per_iter`` games → add to the replay buffer →
    ``cfg.selfplay_train_steps_per_iter`` AdamW steps (lr ``cfg.selfplay_lr``) on buffer samples →
    evaluation (below) → ``save_checkpoint(step, 'selfplay')`` (with ``optim=`` when the callback accepts it). Every
    ``cfg.selfplay_eval_every_iters`` iterations (default 5) and after the last one the model plays
    ``cfg.elo_games`` games against the previous iteration's weights and against a random opponent, on
    paired openings taken from the iteration's own games (``log_elo`` with opponent
    ``'previous-iteration'`` / ``'random'`` and the number of distinct ``openings``).

    Resuming: ``completed_iters`` defaults to what the run's metrics say (see :func:`completed_iterations`)
    and ``optim_state`` to the optimizer state of ``<run_dir>/latest.pt`` when that checkpoint was saved by
    this stage at ``start_step``; iterations already done are skipped. A non-finite loss raises
    ``RuntimeError`` (after saving a checkpoint); ``KeyboardInterrupt`` saves a checkpoint and is re-raised.
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    iters = int(_get(cfg, "selfplay_iters", 50))
    games_per_iter = int(_get(cfg, "selfplay_games_per_iter", 64))
    steps_per_iter = int(_get(cfg, "selfplay_train_steps_per_iter", 200))
    batch_size = int(_get(cfg, "selfplay_batch_size", 256))
    value_weight = float(_get(cfg, "value_loss_weight", 1.0))
    grad_clip = float(_get(cfg, "grad_clip", 1.0))
    amp = bool(_get(cfg, "amp", True)) and device.type == "cuda"
    log_every = max(1, int(_get(cfg, "log_every", 20)))
    elo_games = int(_get(cfg, "elo_games", 20))
    eval_every_iters = max(1, int(_get(cfg, "selfplay_eval_every_iters", 5) or 5))
    seed = int(_get(cfg, "seed", 0))
    rng = np.random.default_rng(seed + 12345)
    run_dir = Path(getattr(logger, "run_dir", "."))

    if completed_iters is None:
        completed_iters = completed_iterations(run_dir, start_step) if start_step > 0 else 0
    completed_iters = max(0, min(int(completed_iters), iters))
    if optim_state is None and start_step > 0:
        optim_state = _resume_optim_state(run_dir, start_step)

    buffer = ReplayBuffer(int(_get(cfg, "replay_buffer_size", 200_000)),
                          k=max(64, int(_get(cfg, "mcts_sims", 64))), seed=seed + completed_iters)
    optim = torch.optim.AdamW(model.parameters(), lr=float(_get(cfg, "selfplay_lr", 2e-4)),
                              weight_decay=float(_get(cfg, "weight_decay", 1e-4)))
    if optim_state is not None:
        try:
            optim.load_state_dict(optim_state)
        except (ValueError, KeyError, RuntimeError) as e:
            log.warning("could not restore the self-play optimizer state (%r): starting it fresh", e)
    step = int(start_step)
    total_steps = start_step + (iters - completed_iters) * steps_per_iter
    t_stage = time.time()
    prev_state: dict[str, torch.Tensor] | None = None
    if completed_iters >= iters:
        logger.log_status(step, message=f"self-play stage already finished ({completed_iters}/{iters} iterations)",
                          stage="selfplay", total_steps=total_steps, eta_s=0.0)
        return step
    logger.log_status(step, message=("self-play stage starting" if completed_iters == 0 else
                                     f"self-play stage resuming after iteration {completed_iters}/{iters}"),
                      stage="selfplay", total_steps=total_steps, eta_s=None)

    try:
        for it in range(completed_iters, iters):
            # ---- 1. generate games ----------------------------------------------------------
            t0 = time.time()
            games = generate_games(model, graph, cfg, games_per_iter, device, rng)
            gen_s = time.time() - t0
            positions = sum(g.plies for g in games)
            for g in games:
                buffer.add(g)
            lengths = np.array([g.plies for g in games], dtype=np.float64)
            results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
            for g in games:
                results[g.result] += 1
            capped = sum(g.capped for g in games)
            log.info("selfplay iter %d: %d games, %d positions in %.1fs (%.1f pos/s), plies mean %.1f, %s, capped %d",
                     it, len(games), positions, gen_s, positions / max(gen_s, 1e-9), lengths.mean() if len(lengths) else 0,
                     results, capped)
            logger.log("selfplay", step, iteration=it, games=len(games), positions=positions, gen_seconds=gen_s,
                       pos_per_sec=positions / max(gen_s, 1e-9), plies_mean=float(lengths.mean()) if len(lengths) else 0.0,
                       plies_min=int(lengths.min()) if len(lengths) else 0, plies_max=int(lengths.max()) if len(lengths) else 0,
                       white_wins=results["1-0"], black_wins=results["0-1"], draws=results["1/2-1/2"], capped=capped,
                       buffer=len(buffer))
            sample = max(games, key=lambda g: (not g.capped, g.plies), default=None)
            if sample is not None:
                logger.log_game(step, pgn=sample.pgn, result=sample.result, moves=(sample.plies + 1) // 2, source="selfplay")
            if games:
                board = chess.Board()
                for uci in sample.moves[: min(len(sample.moves), 20)]:
                    board.push_uci(uci)
                idx, vals = _sample_activity(model, board, device)
                logger.log_activity(step, neuron_idx=idx, values=vals)

            # ---- 2. train -----------------------------------------------------------------------
            prev_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            model.train()
            t_train = time.time()
            seen = 0
            for _ in range(steps_per_iter):
                if len(buffer) == 0:
                    break
                batch = buffer.sample(batch_size, rng)
                x = batch["planes"].to(device, non_blocking=True)
                pi = batch["pi"].to(device, non_blocking=True)
                z = batch["z"].to(device, non_blocking=True)
                mask = batch["legal_mask"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    logits, value = model(x)
                loss, metrics = selfplay_loss(logits, value, pi, z, mask, value_weight)
                if not bool(torch.isfinite(loss)):
                    model.eval()
                    logger.log_status(step, message=f"self-play loss is non-finite at step {step}: stopping",
                                      stage="selfplay", total_steps=total_steps, eta_s=None)
                    raise RuntimeError(f"self-play loss became non-finite at step {step} (iteration {it}); "
                                       "the model was not updated with it — lower selfplay_lr or resume from the "
                                       "last checkpoint")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()
                step += 1
                seen += x.shape[0]
                if step % log_every == 0 or step == start_step + 1:
                    m = {k: float(v) for k, v in metrics.items()}
                    elapsed = time.time() - t_train
                    logger.log_train(step, loss=m["loss"], policy_loss=m["policy_loss"], value_loss=m["value_loss"],
                                     top1=m["top1"], top3=m["top3"], lr=optim.param_groups[0]["lr"],
                                     pos_per_sec=seen / max(elapsed, 1e-9), gpu_mem_gb=_gpu_mem_gb(device),
                                     epoch=float(it), stage="selfplay", buffer=len(buffer))
            model.eval()

            done_iters = it + 1
            # ---- 3. evaluate: vs the previous iteration and vs a random opponent -----------------
            if elo_games > 0 and (done_iters % eval_every_iters == 0 or done_iters == iters):
                openings = opening_fens(games, (elo_games + 1) // 2, rng)
                start_fens = paired_start_fens(openings, elo_games)
                opponents = ["previous-iteration", "random"] if prev_state is not None else ["random"]
                for opponent in opponents:
                    try:
                        if opponent == "previous-iteration":
                            out = _evaluate_vs_previous(model, prev_state, graph, cfg, device, elo_games,
                                                        start_fens, seed + done_iters)
                        else:
                            out = _evaluate_vs_random(model, graph, cfg, device, elo_games, start_fens,
                                                      seed + done_iters)
                    except Exception as e:  # noqa: BLE001 - evaluation must never kill training
                        warnings.warn(f"evaluation vs {opponent} failed: {e!r}", stacklevel=2)
                        out = None
                    if out is None:
                        continue
                    logger.log_elo(step, opponent=opponent, games=int(out["games"]), wins=int(out["wins"]),
                                   draws=int(out["draws"]), losses=int(out["losses"]),
                                   elo_estimate=float(out["elo_estimate"]), openings=int(out.get("openings", 1)),
                                   iteration=done_iters)
                    log.info("selfplay iter %d vs %s: +%d =%d -%d (%d openings, Elo %+.0f)", it, opponent,
                             out["wins"], out["draws"], out["losses"], out.get("openings", 1), out["elo_estimate"])
                    pgns = out.get("pgns") or ([out["pgn"]] if out.get("pgn") else [])
                    if pgns:  # a sample evaluation game for the dashboard (rate limited by the logger)
                        logger.log_game(step, pgn=pgns[0], result=_pgn_result(pgns[0]), moves=_pgn_moves(pgns[0]),
                                        source="eval", opponent=opponent)

            # ---- 4. checkpoint + status (after the evaluation, so the checkpoint carries the new Elo) ----
            _save(save_checkpoint, step, "selfplay", optim)
            eta = (time.time() - t_stage) / (done_iters - completed_iters) * (iters - done_iters)
            logger.log_status(step, message=f"self-play iteration {done_iters}/{iters} done", stage="selfplay",
                              total_steps=total_steps, eta_s=eta, iteration=done_iters)

    except KeyboardInterrupt:
        log.warning("interrupted: saving self-play checkpoint at step %d", step)
        _save(save_checkpoint, step, "selfplay", optim)
        logger.log_status(step, message="self-play interrupted (checkpoint saved)", stage="selfplay",
                          total_steps=total_steps, eta_s=None)
        raise
    except RuntimeError:
        _save(save_checkpoint, step, "selfplay", optim)
        raise
    logger.log_status(step, message="self-play stage finished", stage="selfplay", total_steps=total_steps, eta_s=0.0)
    return step


def _pgn_result(pgn: str) -> str:
    for token in ("1-0", "0-1", "1/2-1/2"):
        if f'[Result "{token}"]' in pgn:
            return token
    return "*"


def _pgn_moves(pgn: str) -> int:
    """Full moves in a PGN string (0 if it does not parse)."""
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return 0
    plies = sum(1 for _ in game.mainline_moves())
    return (plies + 1) // 2


__all__ = [
    "Game",
    "ReplayBuffer",
    "completed_iterations",
    "elo_from_score",
    "generate_games",
    "opening_fens",
    "paired_start_fens",
    "play_match",
    "run_selfplay_stage",
    "selfplay_loss",
]
