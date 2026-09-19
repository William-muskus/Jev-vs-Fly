"""Turn a python-chess board into the state Jev reads.

Jev is not a calculator (see TypeSafe's jev-1.13 jaggedness notes). Legality,
move generation and any numeric evaluation stay in code. The model only sees
an English board, a side to move, and a labelled list of legal moves.
"""

from __future__ import annotations

from typing import Any

import chess

# Choice questions accept at most 255 options (TypeSafe docs / published chess evals).
MAX_CHOICE_OPTIONS = 255

PIECE = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}
WIDE_CENTER = CENTER | {chess.C4, chess.C5, chess.F4, chess.F5}


def uci_of(move: chess.Move) -> str:
    return move.uci()


def describe_move(board: chess.Board, move: chess.Move) -> str:
    """A short English clause of what ``move`` does on ``board``. No search, no scores."""
    parts: list[str] = []
    piece = board.piece_at(move.from_square)
    name = PIECE[piece.piece_type] if piece else "piece"

    if board.is_kingside_castling(move):
        return "castle kingside, tucking the king behind the kingside pawns."
    if board.is_queenside_castling(move):
        return "castle queenside, tucking the king behind the queenside pawns."

    if board.is_en_passant(move):
        parts.append("captures a pawn en passant")
    elif board.is_capture(move):
        captured = board.piece_at(move.to_square)
        victim = PIECE[captured.piece_type] if captured else "piece"
        parts.append(f"captures the {victim}")

    if move.promotion:
        parts.append(f"promotes the pawn to a {PIECE[move.promotion]}")
    elif piece and piece.piece_type == chess.PAWN:
        from_rank = chess.square_rank(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        parts.append("advances a pawn two squares" if abs(to_rank - from_rank) == 2 else "advances a pawn")
    else:
        origin = chess.square_name(move.from_square)
        dest = chess.square_name(move.to_square)
        parts.append(f"moves the {name} from {origin} to {dest}")

    if move.to_square in CENTER:
        parts.append("occupies a central square")
    elif move.to_square in WIDE_CENTER:
        parts.append("plays into the broad centre")

    if board.gives_check(move):
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        parts.append("delivers checkmate" if mate else "gives check")

    return ", ".join(parts) + "."


def _priority(board: chess.Board, move: chess.Move) -> tuple[int, int, str]:
    """Higher is kept first when we have to trim to ``MAX_CHOICE_OPTIONS``."""
    score = 0
    if board.gives_check(move):
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        score += 8 if mate else 4
    if board.is_capture(move) or board.is_en_passant(move):
        score += 2
    if move.promotion:
        score += 3
    if board.is_castling(move):
        score += 1
    return (-score, -int(move.to_square in CENTER), move.uci())


def legal_moves(board: chess.Board, *, limit: int = MAX_CHOICE_OPTIONS) -> list[dict[str, str]]:
    """Legal moves as ``{uci, san, description}``, stable, capped at ``limit``."""
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: _priority(board, m))
    if len(moves) > limit:
        moves = moves[:limit]
    out = []
    for m in moves:
        out.append({"uci": m.uci(), "san": board.san(m), "description": describe_move(board, m)})
    return out


def ascii_board(board: chess.Board) -> str:
    """Unicode board from White's side, ranks labelled. Semantic, not a FEN dump."""
    return board.unicode(borders=True, empty_square="·")


def position_state(board: chess.Board) -> dict[str, Any]:
    """Named JSON state for a System One request. Every field is something Jev can read."""
    moves = legal_moves(board)
    history = [board.san(m) for m in board.move_stack]
    return {
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "fullmove_number": board.fullmove_number,
        "in_check": board.is_check(),
        "board": ascii_board(board),
        "fen": board.fen(),
        "recent_moves": history[-8:],
        "legal_moves": moves,
        "legal_move_count": len(list(board.legal_moves)),
    }


def criteria_from_moves(moves: list[dict[str, str]]) -> dict[str, str]:
    """Choice criteria: UCI key → SAN plus the English description."""
    return {m["uci"]: f"{m['san']} — {m['description']}" for m in moves}
