"""Four ways to ask Jev to pick a chess move.

Question IDs are for our code only — TypeSafe does not send them to the model —
so the two “both questions in one call” variants differ only in JSON key order
of the ``questions`` map. That is the thing we are measuring.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import chess
from typesafe_sdk import Choice

from experiments.board_state import criteria_from_moves, legal_moves, position_state

STRATEGIES = (
    "best_this_turn",
    "best_win_rate",
    "both_turn_then_win",
    "both_win_then_turn",
)

TURN_ID = "best_this_turn"
WIN_ID = "best_win_rate"

TURN_INSTRUCTIONS = (
    "Which of the listed legal chess moves is the best move to play this turn? "
    "Each option key is a UCI move from `legal_moves`. Read `board` (White at the "
    "bottom) and `side_to_move`. Choose exactly one legal move."
)

WIN_INSTRUCTIONS = (
    "Which of the listed legal chess moves most increases the side to move's chance "
    "of winning the full game (game win rate), even if it is not the most forcing or "
    "flashy move this turn? Each option key is a UCI move from `legal_moves`. "
    "Read `board` (White at the bottom) and `side_to_move`. Choose exactly one legal move."
)

SystemOneFn = Callable[[Any, Mapping[str, Any]], Any]


@dataclass
class ChoiceView:
    choice: str
    probabilities: dict[str, float]
    confidence: float

    def top(self, k: int = 5) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:k]


@dataclass
class Pick:
    """One strategy's answer on one position."""

    strategy: str
    uci: str
    san: str
    played_question: str
    answers: dict[str, ChoiceView] = field(default_factory=dict)
    question_order: list[str] = field(default_factory=list)
    legal_count: int = 0
    skipped_api: bool = False
    latency_s: float = 0.0
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    model: str | None = None
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "uci": self.uci,
            "san": self.san,
            "played_question": self.played_question,
            "question_order": self.question_order,
            "answers": {
                k: {
                    "choice": v.choice,
                    "confidence": v.confidence,
                    "top": v.top(),
                    "probabilities": v.probabilities,
                }
                for k, v in self.answers.items()
            },
            "legal_count": self.legal_count,
            "skipped_api": self.skipped_api,
            "latency_s": round(self.latency_s, 4),
            "usage": self.usage,
            "model": self.model,
            "note": self.note,
        }


def _choice(instructions: str, criteria: dict[str, str]) -> Choice:
    return Choice(instructions=instructions, criteria=criteria)


def questions_for(strategy: str, criteria: dict[str, str]) -> dict[str, Choice]:
    """Build the questions map. Insertion order *is* the experimental variable."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    turn = _choice(TURN_INSTRUCTIONS, criteria)
    win = _choice(WIN_INSTRUCTIONS, criteria)
    if strategy == "best_this_turn":
        return {TURN_ID: turn}
    if strategy == "best_win_rate":
        return {WIN_ID: win}
    if strategy == "both_turn_then_win":
        return {TURN_ID: turn, WIN_ID: win}
    return {WIN_ID: win, TURN_ID: turn}


def _as_choice(ans: Any) -> ChoiceView:
    if isinstance(ans, dict):
        return ChoiceView(
            choice=str(ans["choice"]),
            probabilities={str(k): float(v) for k, v in dict(ans["probabilities"]).items()},
            confidence=float(ans["confidence"]),
        )
    return ChoiceView(
        choice=str(ans.choice),
        probabilities={str(k): float(v) for k, v in dict(ans.probabilities).items()},
        confidence=float(ans.confidence),
    )


def _answers_from(result: Any) -> tuple[dict[str, ChoiceView], dict[str, int], str | None]:
    raw = result.answers if hasattr(result, "answers") else result["answers"]
    views = {str(k): _as_choice(v) for k, v in dict(raw).items()}
    usage = {"input_tokens": 0, "output_tokens": 0}
    u = getattr(result, "usage", None)
    if u is not None:
        usage = {
            "input_tokens": int(getattr(u, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
        }
    elif isinstance(result, dict) and isinstance(result.get("usage"), dict):
        usage = {
            "input_tokens": int(result["usage"].get("input_tokens") or 0),
            "output_tokens": int(result["usage"].get("output_tokens") or 0),
        }
    model = getattr(result, "model", None)
    if model is None and isinstance(result, dict):
        model = result.get("model")
    return views, usage, str(model) if model else None


def default_system_one(state: Any, questions: Mapping[str, Any]) -> Any:
    from typesafe_sdk import TypeSafeClient

    with TypeSafeClient(timeout=30.0) as client:
        return client.system_one(state, questions)


def pick_move(
    board: chess.Board,
    strategy: str = "best_this_turn",
    *,
    system_one: SystemOneFn | None = None,
) -> Pick:
    """Ask Jev (or ``system_one``) to pick a legal move under ``strategy``."""
    moves = legal_moves(board)
    legal_n = len(list(board.legal_moves))
    if not moves:
        raise ValueError(f"no legal moves in {board.fen()}")

    if len(moves) == 1:
        return Pick(
            strategy=strategy,
            uci=moves[0]["uci"],
            san=moves[0]["san"],
            played_question="only_legal_move",
            legal_count=legal_n,
            skipped_api=True,
            note="Only one legal move. No TypeSafe request.",
        )

    criteria = criteria_from_moves(moves)
    questions = questions_for(strategy, criteria)
    state = position_state(board)
    fn = system_one or default_system_one
    t0 = time.perf_counter()
    result = fn(state, questions)
    latency = time.perf_counter() - t0
    answers, usage, model = _answers_from(result)

    order = list(questions.keys())
    played_q = order[0]
    if played_q not in answers:
        raise RuntimeError(f"TypeSafe response missing {played_q!r}; got {list(answers)}")
    uci = answers[played_q].choice
    legal_uci = {m["uci"] for m in moves}
    if uci not in legal_uci:
        # Typed Choice should make this impossible; still refuse to play an illegal move.
        raise RuntimeError(f"Jev returned {uci!r} which is not in the legal set")
    san = next(m["san"] for m in moves if m["uci"] == uci)
    return Pick(
        strategy=strategy,
        uci=uci,
        san=san,
        played_question=played_q,
        answers=answers,
        question_order=order,
        legal_count=legal_n,
        latency_s=latency,
        usage=usage,
        model=model,
        note=f"{strategy} played {san} from question {played_q}",
    )


def pick_uci(board: chess.Board, strategy: str = "best_this_turn", **kwargs: Any) -> chess.Move:
    return chess.Move.from_uci(pick_move(board, strategy, **kwargs).uci)
