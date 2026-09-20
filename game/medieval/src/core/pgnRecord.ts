import type { Chess } from "chess.js";

import type { EndReason, GameResult } from "./types";

/** PGN Seven Tag Roster result, including `*` while the game is still live. */
export type PgnResult = "1-0" | "0-1" | "1/2-1/2" | "*";

export const PGN_EVENT = "Jev vs Fly wizard chess";
export const PGN_SITE = "The Great Hall";

/** Map a hall ending onto the PGN Result tag (and the movetext terminator). */
export function pgnResultTag(result: GameResult | null): PgnResult {
  if (!result) return "*";
  if (result.winner === "w") return "1-0";
  if (result.winner === "b") return "0-1";
  return "1/2-1/2";
}

/** UTC date in the PGN `YYYY.MM.DD` form. */
export function pgnDateTag(now = new Date()): string {
  const year = now.getUTCFullYear();
  const month = String(now.getUTCMonth() + 1).padStart(2, "0");
  const day = String(now.getUTCDate()).padStart(2, "0");
  return `${year}.${month}.${day}`;
}

/**
 * chess.js prints the Seven Tag Roster as `?` / `*` unless these are set, and
 * it always appends the Result tag as the movetext terminator — so a finished
 * mate used to read `[White "?"]` / `Qg2# *`. Stamp before every `.pgn()`.
 */
export function stampChessPgn(
  chess: Chess,
  options: {
    white: string;
    black: string;
    result: GameResult | null;
    round?: number;
    now?: Date;
  },
): string {
  chess.setHeader("Event", PGN_EVENT);
  chess.setHeader("Site", PGN_SITE);
  chess.setHeader("Date", pgnDateTag(options.now));
  chess.setHeader("Round", String(options.round ?? 1));
  chess.setHeader("White", options.white);
  chess.setHeader("Black", options.black);
  chess.setHeader("Result", pgnResultTag(options.result));
  if (options.result) chess.setHeader("Termination", terminationTag(options.result.reason));
  else chess.removeHeader("Termination");
  return chess.pgn();
}

function terminationTag(reason: EndReason): string {
  return reason;
}
