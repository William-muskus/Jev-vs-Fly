/**
 * Layer 1 + 2: everything that must be exact.
 *
 * Jev is not a calculator and does not count reliably, so no arithmetic, no
 * legality and no tactics live above this file. This layer produces a shortlist
 * of moves that are all defensible, plus a natural-language description of what
 * each one does. Semantic words go up to the model; the numbers stay here.
 *
 * Performance note: `chess.moves({ verbose: true })` costs about 2.5ms a call in
 * chess.js 1.4 because it builds a before-and-after FEN for every move. The
 * search therefore runs on plain SAN strings and only asks for verbose detail at
 * the root, where there are a few dozen moves rather than a few hundred thousand.
 */

import { Chess, type Move, type Square } from "chess.js";

export const VALUE: Record<string, number> = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0 };
const MATE = 100000;

const PIECE_NAME: Record<string, string> = {
  p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king",
};

/**
 * Piece-square tables, read from rank 8 down to rank 1, white's point of view.
 *
 * These exist for one reason the probe made obvious: on material alone every
 * opening move scores identically, so the shortlist collapses to whatever move
 * generation happened to emit first and Jev only ever sees a3, a4, b3, b4.
 * A weak positional tiebreak is enough to put e4, d4 and Nf3 in front of it.
 * They are held at low weight on purpose. Generic positional taste belongs to
 * the search; the interesting taste is supposed to come from the persona.
 */
const PST_WEIGHT = 0.6;

// prettier-ignore
const PST: Record<string, number[]> = {
  p: [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
  ],
  n: [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
  ],
  b: [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
  ],
  r: [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
  ],
  q: [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
  ],
  k: [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0, 20, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
  ],
};

function pstIndex(square: string, color: "w" | "b"): number {
  const file = square.charCodeAt(0) - 97;
  const rankFromTop = 8 - Number(square[1]);
  return color === "w" ? rankFromTop * 8 + file : (7 - rankFromTop) * 8 + file;
}

/** Evaluation in centipawns from the point of view of the side to move. */
function evaluate(chess: Chess): number {
  let total = 0;
  for (const row of chess.board()) {
    for (const sq of row) {
      if (!sq) continue;
      const v = VALUE[sq.type] + PST_WEIGHT * PST[sq.type][pstIndex(sq.square, sq.color)];
      total += sq.color === "w" ? v : -v;
    }
  }
  return chess.turn() === "w" ? total : -total;
}

/** Material only, ignoring placement. Used for the plain-English balance line. */
function materialFor(chess: Chess, color: "w" | "b"): number {
  let total = 0;
  for (const row of chess.board()) {
    for (const sq of row) {
      if (!sq) continue;
      total += sq.color === color ? VALUE[sq.type] : -VALUE[sq.type];
    }
  }
  return total;
}

const QUIESCE_MAX_PLY = 4;

/**
 * Captures-only search at the leaves. Without it a depth-N search sees the
 * capture on the last ply and never the recapture, and the bot hangs pieces.
 */
function quiesce(chess: Chess, alpha: number, beta: number, ply: number, budget: number): number {
  const moves = chess.moves();
  if (moves.length === 0) return chess.inCheck() ? -MATE + ply : 0;

  const inCheck = chess.inCheck();
  let best = -Infinity;

  if (!inCheck) {
    best = evaluate(chess);
    if (best >= beta) return best;
    if (best > alpha) alpha = best;
    if (budget <= 0) return best;
  }

  const forced = inCheck ? moves : moves.filter((m) => m.includes("x") || m.includes("="));
  if (forced.length === 0) return best;

  for (const m of forced) {
    chess.move(m);
    const v = -quiesce(chess, -beta, -alpha, ply + 1, budget - 1);
    chess.undo();
    if (v > best) best = v;
    if (best > alpha) alpha = best;
    if (alpha >= beta) break;
  }
  return best;
}

function orderKey(san: string): number {
  let r = 0;
  if (san.includes("#")) r += 8;
  if (san.includes("=")) r += 4;
  if (san.includes("x")) r += 2;
  if (san.includes("+")) r += 1;
  return r;
}

function negamax(chess: Chess, depth: number, alpha: number, beta: number, ply: number): number {
  const moves = chess.moves();
  if (moves.length === 0) return chess.inCheck() ? -MATE + ply : 0;
  if (depth === 0) return quiesce(chess, alpha, beta, ply, QUIESCE_MAX_PLY);

  moves.sort((a, b) => orderKey(b) - orderKey(a));

  let best = -Infinity;
  for (const m of moves) {
    chess.move(m);
    const v = -negamax(chess, depth - 1, -beta, -alpha, ply + 1);
    chess.undo();
    if (v > best) best = v;
    if (best > alpha) alpha = best;
    if (alpha >= beta) break;
  }
  return best;
}

export interface Candidate {
  id: string;
  san: string;
  uci: string;
  /** Search score in centipawns from the mover's point of view. */
  cp: number;
  /** Centipawns given up against the best available move. Zero for the best. */
  loss: number;
  /** What this move does, in words. This is what Jev reads. */
  effect: string;
  forcedMate: boolean;
}

const CENTER = new Set(["d4", "e4", "d5", "e5"]);
const WIDE_CENTER = new Set(["c4", "c5", "d4", "d5", "e4", "e5", "f4", "f5"]);

function describe(chess: Chess, m: Move, cp: number, staticNow: number, mated: boolean): string {
  const parts: string[] = [];

  if (mated) {
    parts.push("delivers checkmate");
  } else if (cp > MATE / 2) {
    parts.push("forces checkmate within a few moves");
  } else {
    // Named buckets rather than raw centipawns. Jev reads semantic
    // representations far better than numeric ones.
    const delta = cp - staticNow;
    if (delta >= 280) parts.push("wins a piece or more");
    else if (delta >= 90) parts.push("wins a pawn");
    else if (delta > -45) parts.push("leaves the material balance unchanged");
    else if (delta > -280) parts.push("gives up a pawn");
    else parts.push("sacrifices a piece or more");
  }

  if (m.flags.includes("k")) parts.push("castles kingside and tucks the king away");
  if (m.flags.includes("q")) parts.push("castles queenside and tucks the king away");
  if (m.promotion) parts.push(`promotes a pawn to a ${PIECE_NAME[m.promotion]}`);
  if (m.captured && !m.promotion) parts.push(`captures a ${PIECE_NAME[m.captured]}`);
  if (!mated && m.san.includes("+")) parts.push("gives check");

  const homeRank = m.color === "w" ? "1" : "8";
  if ((m.piece === "n" || m.piece === "b") && m.from[1] === homeRank) {
    parts.push("brings a new piece off the back rank");
  }
  if (CENTER.has(m.to)) parts.push("occupies a central square");
  else if (WIDE_CENTER.has(m.to)) parts.push("plays into the broad centre");
  if (m.piece === "k" && !m.flags.includes("k") && !m.flags.includes("q")) {
    parts.push("moves the king by hand, giving up the right to castle");
  }
  if (m.piece === "p" && !m.captured && Math.abs(Number(m.to[1]) - Number(m.from[1])) === 2) {
    parts.push("advances a pawn two squares");
  }

  if (!mated) {
    chess.move(m);
    const attacked = chess.isAttacked(m.to as Square, chess.turn());
    chess.undo();
    if (attacked && m.piece !== "p") {
      parts.push(`leaves the ${PIECE_NAME[m.piece]} on a square the opponent attacks`);
    }
  }

  return parts.join(", ") + ".";
}

export interface SearchOptions {
  depth?: number;
  marginCp?: number;
  maxCandidates?: number;
}

export interface Shortlist {
  candidates: Candidate[];
  /** Set when code alone settles the move and no Jev request is warranted. */
  forced: Candidate | null;
  legalCount: number;
  searchMs: number;
}

export function shortlist(chess: Chess, opts: SearchOptions = {}): Shortlist {
  const depth = opts.depth ?? 2;
  const marginCp = opts.marginCp ?? 90;
  const maxCandidates = opts.maxCandidates ?? 6;
  const started = Date.now();

  const legal = chess.moves({ verbose: true }) as Move[];
  if (legal.length === 0) {
    return { candidates: [], forced: null, legalCount: 0, searchMs: Date.now() - started };
  }

  const staticNow = evaluate(chess);

  // Order root moves by a single static evaluation each. Nearly free, and a
  // good first move makes the contention window below prune hard.
  const ordered = legal
    .map((m) => {
      chess.move(m);
      const guess = -evaluate(chess);
      chess.undo();
      return { m, guess };
    })
    .sort((a, b) => b.guess - a.guess || orderKey(b.m.san) - orderKey(a.m.san))
    .map((x) => x.m);

  const scored: Candidate[] = [];
  let bestSoFar = -Infinity;

  for (const m of ordered) {
    // Only moves within `marginCp` of the best can reach the shortlist, so any
    // move provably worse than that needs a bound, not an exact score. Searching
    // each root move against that bound rather than against an open window is
    // what makes this finish in well under a second instead of half a minute.
    const floor = bestSoFar === -Infinity ? -Infinity : bestSoFar - marginCp - 1;

    chess.move(m);
    const mated = chess.moves().length === 0 && chess.inCheck();
    const cp = mated ? MATE : -negamax(chess, depth - 1, -Infinity, -floor, 1);
    chess.undo();

    if (cp > bestSoFar) bestSoFar = cp;

    scored.push({
      id: "",
      san: m.san,
      uci: m.from + m.to + (m.promotion ?? ""),
      cp: Math.round(cp),
      loss: 0,
      effect: describe(chess, m, cp, staticNow, mated),
      forcedMate: mated,
    });

    if (mated) break; // nothing beats mate; stop searching
  }

  scored.sort((a, b) => b.cp - a.cp);
  const bestCp = scored[0].cp;
  for (const c of scored) c.loss = Math.round(bestCp - c.cp);

  // Mate on the board is not a matter of taste.
  if (scored[0].forcedMate) {
    scored[0].id = "m1";
    return {
      candidates: [scored[0]], forced: scored[0],
      legalCount: legal.length, searchMs: Date.now() - started,
    };
  }

  const kept = scored.filter((c) => c.loss <= marginCp).slice(0, maxCandidates);
  kept.forEach((c, i) => (c.id = `c${i + 1}`));

  return {
    candidates: kept,
    forced: kept.length === 1 ? kept[0] : null,
    legalCount: legal.length,
    searchMs: Date.now() - started,
  };
}

export type Phase = "opening" | "middlegame" | "endgame";

export function phaseOf(chess: Chess): Phase {
  let nonPawn = 0;
  let home = 0;
  for (const row of chess.board()) {
    for (const sq of row) {
      if (!sq) continue;
      if (sq.type !== "p" && sq.type !== "k") nonPawn += VALUE[sq.type];
      if ((sq.type === "n" || sq.type === "b") && (sq.square[1] === "1" || sq.square[1] === "8")) home++;
    }
  }
  if (nonPawn <= 1400) return "endgame";
  const fullmove = Number(chess.fen().split(" ")[5] ?? 1);
  if (fullmove <= 12 && home >= 4) return "opening";
  return "middlegame";
}

export interface PositionBrief {
  side_to_move: string;
  phase: Phase;
  material_balance: string;
  my_king_is_in_check: boolean;
  move_number: number;
  recent_moves: string[];
  [key: string]: string | number | boolean | string[];
}

/** A literal read of the position. Nothing here asks the model to count. */
export function positionBrief(chess: Chess): PositionBrief {
  const mat = materialFor(chess, chess.turn());
  const balance =
    mat >= 280 ? "I am a piece or more ahead"
    : mat >= 90 ? "I am a pawn ahead"
    : mat > -90 ? "material is level"
    : mat > -280 ? "I am a pawn down"
    : "I am a piece or more down";

  return {
    side_to_move: chess.turn() === "w" ? "white" : "black",
    phase: phaseOf(chess),
    material_balance: balance,
    my_king_is_in_check: chess.inCheck(),
    move_number: Number(chess.fen().split(" ")[5] ?? 1),
    recent_moves: chess.history().slice(-6),
  };
}
