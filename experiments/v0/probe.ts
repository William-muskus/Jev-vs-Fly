/**
 * Offline check of everything below the model. No network, no API key.
 *
 * This is the layer that has to be right before Jev is worth adding: if the
 * shortlist already contains a blunder, no amount of good taste above it helps.
 *
 * Positions are built by playing moves rather than by hand-written FENs,
 * because a mistyped FEN produces a test that passes for the wrong reason.
 */

import { Chess } from "chess.js";
import { shortlist, positionBrief, type Shortlist } from "./tactics.js";

interface Case {
  name: string;
  moves: string[];
  marginCp?: number;
  expect: (o: Shortlist) => string | null;
}

const sans = (o: Shortlist) => (o.forced ? [o.forced] : o.candidates).map((c) => c.san).join(" ");

const CASES: Case[] = [
  {
    name: "Mate in one is taken, and no request is spent",
    moves: ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"],
    expect: (o) => (o.forced?.san === "Qxf7#" ? null : `expected forced Qxf7#, got ${sans(o)}`),
  },
  {
    name: "A free pawn is taken",
    moves: ["d4", "e5"],
    expect: (o) => (o.candidates[0]?.san === "dxe5" ? null : `expected dxe5 first, got ${sans(o)}`),
  },
  {
    name: "A check that hangs the queen never reaches the shortlist",
    moves: ["e4", "e5", "Qh5", "Nc6"],
    expect: (o) =>
      o.candidates.some((c) => c.san.startsWith("Qxf7"))
        ? `shortlist contains Qxf7+, which drops the queen: ${sans(o)}`
        : null,
  },
  {
    name: "In check, every candidate is a real escape",
    moves: ["e4", "d5", "Bb5+"],
    expect: (o) => (o.candidates.length >= 2 ? null : `expected several escapes, got ${sans(o)}`),
  },
  {
    name: "Opening shortlist reaches real opening moves, not just a3 and b3",
    moves: [],
    expect: (o) =>
      o.candidates.some((c) => ["e4", "d4", "Nf3", "c4", "Nc3", "e3", "d3", "g3"].includes(c.san))
        ? null
        : `shortlist is all filler: ${sans(o)}`,
  },
  {
    name: "A wide margin still filters out the outright blunders",
    moves: ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"],
    marginCp: 250,
    expect: (o) =>
      o.candidates.some((c) => c.san === "Bxf7+" && c.loss > 250)
        ? `kept a move past the margin: ${sans(o)}`
        : null,
  },
];

let failures = 0;
let slowest = 0;
const t0 = Date.now();

for (const tc of CASES) {
  const chess = new Chess();
  for (const m of tc.moves) chess.move(m);

  const out = shortlist(chess, { depth: 2, marginCp: tc.marginCp ?? 90, maxCandidates: 6 });
  slowest = Math.max(slowest, out.searchMs);
  const problem = tc.expect(out);

  console.log(`\n${problem ? "FAIL" : "ok  "}  ${tc.name}`);
  console.log(`      ${out.legalCount} legal, searched in ${out.searchMs}ms`);
  if (problem) { failures++; console.log(`      >>> ${problem}`); }
  for (const c of out.forced ? [out.forced] : out.candidates) {
    console.log(`      ${(c.id || "--").padEnd(3)} ${c.san.padEnd(7)} loss=${String(c.loss).padStart(4)}  ${c.effect}`);
  }
}

console.log(`\n--- state Jev would receive at move 1 ---`);
console.log(JSON.stringify(positionBrief(new Chess()), null, 2));
console.log(`\n${failures ? `${failures} FAILING` : "all passing"}, slowest search ${slowest}ms, total ${Date.now() - t0}ms`);
process.exit(failures ? 1 : 0);
