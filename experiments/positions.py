"""Curated positions for the prompt-strategy comparison.

Positions are built by playing moves rather than hand-typed FENs wherever we
can, so a mistyped FEN cannot make a test pass for the wrong reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class Position:
    id: str
    title: str
    why: str
    moves: tuple[str, ...] = ()
    fen: str | None = None

    def board(self) -> chess.Board:
        if self.fen:
            return chess.Board(self.fen)
        b = chess.Board()
        for san in self.moves:
            b.push_san(san)
        return b


POSITIONS: tuple[Position, ...] = (
    Position(
        id="start",
        title="Starting position",
        why="Twenty legal moves, all 'equal' on material. Does Jev prefer e4/d4/Nf3 over a3?",
    ),
    Position(
        id="open_e4e5",
        title="Open game after 1.e4 e5",
        why="A normal developing choice: Nf3, Nc3, Bc4, d4, f4…",
        moves=("e4", "e5"),
    ),
    Position(
        id="sicilian",
        title="Sicilian after 1.e4 c5",
        why="Does the win-rate question prefer Open Sicilian (Nf3) over quieter tries?",
        moves=("e4", "c5"),
    ),
    Position(
        id="scholars_threat",
        title="Scholar's-mate threat (White to play Qxf7#)",
        why="Mate in one is on the board. Every strategy should find Qxf7#.",
        moves=("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"),
    ),
    Position(
        id="hanging_queen",
        title="Queen hangs on f7",
        why="Qxf7+ drops the queen. A good pick avoids it; a bad one takes the bait.",
        moves=("e4", "e5", "Qh5", "Nc6"),
    ),
    Position(
        id="in_check",
        title="Black is in check (Bb5+)",
        why="Every legal move is an escape. The two questions should still be legal checks.",
        moves=("e4", "d5", "Bb5+"),
    ),
    Position(
        id="italian",
        title="Quiet Italian",
        why="A developed middlegame-shaped opening; many reasonable moves.",
        moves=("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"),
    ),
    Position(
        id="kq_endgame",
        title="King and queen vs king",
        why="Winning the game is a matter of not stalemating and driving the king to the edge.",
        fen="8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1",
    ),
)
