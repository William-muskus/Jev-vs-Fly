"""Unit tests for Jev prompt strategies — no TypeSafe key required."""

from __future__ import annotations

import chess
import pytest

from experiments.board_state import criteria_from_moves, describe_move, legal_moves, position_state
from experiments.compare import compare_position, render_markdown, run
from experiments.positions import POSITIONS
from experiments.prompts import pick_move, questions_for


def test_opening_has_twenty_moves_and_e4_is_described():
    board = chess.Board()
    moves = legal_moves(board)
    assert len(moves) == 20
    e4 = next(m for m in moves if m["uci"] == "e2e4")
    assert e4["san"] == "e4"
    assert "two squares" in e4["description"]
    assert "central" in e4["description"]
    crit = criteria_from_moves(moves)
    assert crit["e2e4"].startswith("e4 —")


def test_mate_in_one_is_labelled():
    board = chess.Board()
    for san in ("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"):
        board.push_san(san)
    mate = chess.Move.from_uci("h5f7")
    assert "checkmate" in describe_move(board, mate)
    assert board.san(mate) == "Qxf7#"


def test_question_key_order_is_the_experimental_variable():
    crit = {"e2e4": "e4", "d2d4": "d4"}
    assert list(questions_for("best_this_turn", crit)) == ["best_this_turn"]
    assert list(questions_for("best_win_rate", crit)) == ["best_win_rate"]
    assert list(questions_for("both_turn_then_win", crit)) == ["best_this_turn", "best_win_rate"]
    assert list(questions_for("both_win_then_turn", crit)) == ["best_win_rate", "best_this_turn"]


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        questions_for("teleport", {"e2e4": "e4"})


class Scripted:
    """Deterministic stand-in for TypeSafe: plays a preferred UCI if legal, else the first option."""

    def __init__(self, prefer: str = "e2e4"):
        self.prefer = prefer
        self.calls: list[list[str]] = []

    def __call__(self, state, questions):
        self.calls.append(list(questions.keys()))
        answers = {}
        for qid, q in questions.items():
            keys = list(q.criteria.keys())
            choice = self.prefer if self.prefer in keys else keys[0]
            rest = [k for k in keys if k != choice]
            mass = 0.3 / len(rest) if rest else 0.0
            probs = {k: (0.7 if k == choice else mass) for k in keys}
            answers[qid] = {
                "type": "choice",
                "choice": choice,
                "probabilities": probs,
                "confidence": 0.62,
            }
        return {"model": "fake", "answers": answers, "usage": {"input_tokens": 11, "output_tokens": 2}}


def test_pick_move_plays_the_first_question_and_records_order():
    board = chess.Board()
    fake = Scripted("d2d4")
    both = pick_move(board, "both_win_then_turn", system_one=fake)
    assert fake.calls[0] == ["best_win_rate", "best_this_turn"]
    assert both.uci == "d2d4"
    assert both.played_question == "best_win_rate"
    assert set(both.answers) == {"best_win_rate", "best_this_turn"}

    turn = pick_move(board, "best_this_turn", system_one=Scripted("g1f3"))
    assert turn.uci == "g1f3" and turn.san == "Nf3"


def test_only_legal_move_skips_the_api():
    # Black king a8, White king b6 and knight c7: the only legal move is Kb8.
    board = chess.Board("k7/2N5/1K6/8/8/8/8/8 b - - 0 1")
    assert [m.uci() for m in board.legal_moves] == ["a8b8"]
    fake = Scripted("a8a7")
    pick = pick_move(board, "best_this_turn", system_one=fake)
    assert pick.skipped_api and pick.uci == "a8b8" and pick.san == "Kb8"
    assert fake.calls == []


def test_compare_with_scripted_client_writes_agreement_flags():
    fake = Scripted("e2e4")
    start = next(p for p in POSITIONS if p.id == "start")
    row = compare_position(start, system_one=fake)
    assert row["flags"]["solo_turn_equals_solo_win"] is True
    assert row["picks"]["both_turn_then_win"]["question_order"] == ["best_this_turn", "best_win_rate"]
    assert row["picks"]["both_win_then_turn"]["question_order"] == ["best_win_rate", "best_this_turn"]
    payload = run([start], system_one=fake)
    md = render_markdown(payload["positions"], payload["summary"])
    assert "best_this_turn" in md and start.title in md
    assert payload["summary"]["agreement"]["solo_turn_equals_solo_win"]["yes"] == 1


def test_position_state_is_named_json():
    state = position_state(chess.Board())
    assert state["side_to_move"] == "white"
    assert "r" in state["board"].lower() or "♖" in state["board"] or "♜" in state["board"]
    assert state["legal_move_count"] == 20
    assert all("uci" in m and "san" in m for m in state["legal_moves"])
