/**
 * Wizard chess, as played in the underground chamber.
 *
 * The laws of movement are ordinary chess (chess.js / FIDE). What is enchanted
 * is the capture: the piece that takes always destroys the piece it lands on.
 * There is no combat roll. The smash is the rule.
 */

import type { LedgerMove, PieceKind } from "./types";

export const WIZARD_RANKS: Record<PieceKind, string> = {
  k: "king",
  q: "queen",
  b: "bishop",
  n: "knight",
  r: "rook",
  p: "pawn",
};

/** English command the living pieces hear — "White knight to E5!" */
export function heraldCall(move: LedgerMove): string {
  const army = move.color === "w" ? "White" : "Black";
  const piece = WIZARD_RANKS[move.kind];
  const dest = move.to.toUpperCase();

  if (move.castle) {
    const side = move.to.startsWith("g") ? "kingside" : "queenside";
    return `${army} king, castle ${side}!`;
  }

  if (move.promotion) {
    const became = WIZARD_RANKS[move.promotion];
    const line = `${army} pawn to ${dest}, becomes a ${became}!`;
    if (move.mate) return `${line} Checkmate!`;
    if (move.check) return `${line} Check!`;
    return line;
  }

  if (move.capture) {
    const line = `${army} ${piece} takes on ${dest}!`;
    if (move.mate) return `${line} Checkmate!`;
    if (move.check) return `${line} Check!`;
    return line;
  }

  const line = `${army} ${piece} to ${dest}!`;
  if (move.mate) return `${line} Checkmate!`;
  if (move.check) return `${line} Check!`;
  return line;
}
