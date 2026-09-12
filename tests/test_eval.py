"""Tests for flychess.eval (opponents, match play, Elo). Fast: toy / tiny graphs, few short games."""
from __future__ import annotations

import math

import chess
import pytest
import torch

from flychess import paths
from flychess.chessenv.encoding import legal_move_mask, move_to_index
from flychess.connectome.graph import toy_graph
from flychess.eval import (
    BrainPolicyPlayer,
    GreedyMaterialPlayer,
    MatchResult,
    Player,
    RandomPlayer,
    StockfishPlayer,
    elo_estimate,
    make_opponent,
    play_match,
    round_robin,
)
from flychess.eval.opponents import material_balance
from flychess.model.config import BrainConfig
from flychess.model.flybrain import FlyBrain

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def toy_model():
    graph = toy_graph(n=150, nnz=1200, n_in=16, n_out=16, seed=3)
    model = FlyBrain(graph, BrainConfig(graph_path="toy", steps=3, value_hidden=8)).to(DEVICE)
    return model


# ---- Elo -----------------------------------------------------------------------------------------
def test_elo_estimate_sanity():
    assert elo_estimate(5, 0, 5) == 0.0
    assert elo_estimate(0, 10, 0) == 0.0
    assert elo_estimate(0, 0, 0) == 0.0
    assert elo_estimate(7, 0, 3) == pytest.approx(400 * math.log10(0.7 / 0.3))
    assert elo_estimate(3, 0, 7) == pytest.approx(-elo_estimate(7, 0, 3))
    perfect = elo_estimate(20, 0, 0)
    assert math.isfinite(perfect) and 500 < perfect < 800
    assert elo_estimate(0, 0, 20) == pytest.approx(-perfect)
    assert elo_estimate(2, 0, 0) < perfect  # fewer games: less confident
    assert elo_estimate(7, 0, 3, opponent_elo=1500) == pytest.approx(1500 + elo_estimate(7, 0, 3))
    r = MatchResult(wins=1, draws=1, losses=0)
    assert r.games == 2 and r.score == 0.75 and r.elo == elo_estimate(1, 1, 0)


# ---- opponents -----------------------------------------------------------------------------------
def test_random_and_material_players():
    board = chess.Board()
    rp, mp = RandomPlayer(seed=1), GreedyMaterialPlayer(seed=1)
    assert isinstance(rp, Player) and isinstance(mp, Player)
    assert rp.choose(board) in board.legal_moves and mp.choose(board) in board.legal_moves
    # material player grabs the hanging queen
    b = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    assert material_balance(b, chess.WHITE) == -4 and mp.choose(b) == chess.Move.from_uci("d1d5")
    b2 = chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")
    assert mp.choose(b2) in b2.legal_moves
    b3 = chess.Board("k7/8/8/8/8/8/8/3qK3 w - - 0 1")  # king must take the queen
    assert mp.choose(b3) == chess.Move.from_uci("e1d1")
    assert material_balance(chess.Board(), chess.WHITE) == 0
    mated = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert mated.is_checkmate()
    with pytest.raises(ValueError):
        rp.choose(mated)
    with pytest.raises(ValueError):
        mp.choose(mated)
    seq_a = [RandomPlayer(seed=5).choose(chess.Board()) for _ in range(3)]
    seq_b = [RandomPlayer(seed=5).choose(chess.Board()) for _ in range(3)]
    assert seq_a == seq_b
    assert make_opponent("random").name == "random" and make_opponent("material").name == "material"
    with pytest.raises(ValueError):
        make_opponent("nope")
    if not StockfishPlayer.available():
        with pytest.raises(FileNotFoundError):
            StockfishPlayer()


def test_brain_policy_player_is_legal_and_batched(toy_model):
    fly = BrainPolicyPlayer(toy_model, device=DEVICE, temperature=0.0, name="fly")
    boards = [chess.Board(), chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")]
    boards[1].push_san("Bb5")
    probs, values = fly.policy(boards)
    assert probs.shape == (2, 4168) and values.shape == (2,)
    for p, b in zip(probs, boards, strict=True):
        mask = legal_move_mask(b)
        assert p[~mask].sum() == 0 and p.sum() == pytest.approx(1.0, abs=1e-4)
    moves = fly.choose_many(boards)
    assert all(m in b.legal_moves for m, b in zip(moves, boards, strict=True))
    assert fly.choose(boards[0]) == moves[0]
    # argmax matches the max-probability legal index
    assert move_to_index(moves[0], boards[0]) == int(probs[0].argmax())
    assert fly.choose_many([]) == []
    hot = BrainPolicyPlayer(toy_model, device=DEVICE, temperature=1.5, seed=1)
    assert all(hot.choose(chess.Board()) in chess.Board().legal_moves for _ in range(5))
    with pytest.raises(ValueError):
        fly.choose(chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"))  # checkmated: no legal moves
    toy_model.train()  # the player restores the training flag
    fly.choose(chess.Board())
    assert toy_model.training


# ---- matches -------------------------------------------------------------------------------------
def test_play_match_random_vs_material_two_games():
    seen = []
    m = play_match(RandomPlayer(), GreedyMaterialPlayer(), games=2, max_plies=60, seed=3,
                   on_game=lambda i, res, pgn: seen.append((i, res, pgn)))
    assert m.games == 2 == m.wins + m.draws + m.losses
    assert len(m.results) == 2 and all(r in ("1-0", "0-1", "1/2-1/2") for r in m.results)
    assert len(m.pgns) == 2 and '[White "random"]' in m.pgns[0] and '[White "material"]' in m.pgns[1]
    assert all(0 < p <= 60 for p in m.plies)
    assert sorted(i for i, _, _ in seen) == [0, 1] and all(pgn for _, _, pgn in seen)
    assert "random vs material" in str(m) and m.to_dict()["games"] == 2
    # reproducible
    m2 = play_match(RandomPlayer(), GreedyMaterialPlayer(), games=2, max_plies=60, seed=3)
    assert m2.results == m.results and m2.plies == m.plies


def test_play_match_with_brain_uses_choose_many(toy_model):
    fly = BrainPolicyPlayer(toy_model, device=DEVICE)
    calls = []
    orig = fly.choose_many

    def spy(boards):
        calls.append(len(boards))
        return orig(boards)

    fly.choose_many = spy  # type: ignore[method-assign]
    m = play_match(fly, RandomPlayer(seed=2), games=4, max_plies=30, seed=1)
    assert m.games == 4 and max(calls) > 1  # several games evaluated in one forward pass
    assert m.truncated <= 4


def test_round_robin_table():
    table = round_robin([RandomPlayer(name="r1", seed=1), RandomPlayer(name="r2", seed=2),
                         GreedyMaterialPlayer(name="mat")], games=2, max_plies=40, seed=0)
    assert [row["name"] for row in table] and len(table) == 3
    assert all(row["games"] == 4 for row in table)
    assert all(set(row["elo_vs"]) == {n for n in ("r1", "r2", "mat") if n != row["name"]} for row in table)
    assert table[0]["score"] >= table[-1]["score"]
    total_points = sum(row["wins"] + 0.5 * row["draws"] for row in table)
    assert total_points == pytest.approx(6.0)  # 3 matches x 2 games


def test_evaluate_run_with_checkpoint(tmp_path, monkeypatch, toy_model):
    from flychess.eval.elo import evaluate_run
    from flychess.train.config import TrainConfig
    from flychess.train.metrics import read_metrics
    from flychess.train.trainer import save_checkpoint

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    graph = toy_graph(n=150, nnz=1200, n_in=16, n_out=16, seed=3)
    gp = graph.save(tmp_path / "toy.npz")
    cfg = TrainConfig(run="ev", graph=str(gp), brain={"steps": 3, "value_hidden": 8})
    save_checkpoint(tmp_path / "runs" / "ev", toy_model, cfg, gp, step=3, stage="imitation")
    save_checkpoint(tmp_path / "runs" / "ev", toy_model, cfg, gp, step=7, stage="imitation")
    out = evaluate_run("ev", games=2, opponents=["random", "ckpt-3"], device=DEVICE, max_plies=20, verbose=False)
    assert set(out) == {"random", "ckpt-3"} and out["random"]["games"] == 2 and out["ckpt-3"]["games"] == 2
    assert out["random"]["player_a"] == "ev@7" and out["ckpt-3"]["player_b"] == "ckpt-3@3"
    recs = read_metrics(tmp_path / "runs" / "ev", kinds=["elo"])
    assert [r["opponent"] for r in recs] == ["random", "ckpt-3"] and all(r["step"] == 7 for r in recs)
