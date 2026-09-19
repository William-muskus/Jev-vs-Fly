/**
 * Layer 4: policy. Raw judgments come back from Jev untouched; every weight,
 * threshold and comparison is applied here, in code, where it can be changed
 * without asking the model anything again.
 */

import { Chess } from "chess.js";
import { TypeSafeClient } from "@typesafe-ai/sdk";
import { shortlist, positionBrief, type Candidate } from "./tactics.js";
import type { Persona } from "./personas.js";
import { judge, type JevJudgment } from "./jev.js";

export interface ScoredCandidate extends Candidate {
  fit: number;
  aggression: number;
  looseness: number;
  total: number;
}

export interface Decision {
  san: string;
  uci: string;
  source: "only-legal-move" | "forced-mate" | "jev" | "low-confidence-fallback" | "search-fallback";
  note: string;
  table: ScoredCandidate[];
  confidence?: number;
  underPressure?: number;
  usage?: { input_tokens: number; output_tokens: number };
}

function combine(
  candidates: Candidate[],
  persona: Persona,
  j: JevJudgment,
): ScoredCandidate[] {
  const defensive = j.underPressure;
  const wAggr = persona.weights.aggression * (1 - 0.7 * defensive);
  const wLoose = persona.weights.looseness + 0.8 * defensive;

  const scored = candidates.map((c) => {
    const fit = j.fit[c.id] ?? 0;
    const aggression = j.aggression[c.id] ?? 0;
    const looseness = j.looseness[c.id] ?? 0;
    const total =
      persona.weights.fit * fit +
      wAggr * aggression -
      wLoose * looseness -
      persona.weights.material * (c.loss / 100);
    return { ...c, fit, aggression, looseness, total };
  });

  scored.sort((a, b) => b.total - a.total);
  return scored;
}

export async function chooseMove(
  chess: Chess,
  persona: Persona,
  client: TypeSafeClient | null,
): Promise<Decision> {
  const list = shortlist(chess, {
    depth: persona.searchDepth,
    marginCp: persona.marginCp,
    maxCandidates: 6,
  });

  if (list.candidates.length === 0) throw new Error("No legal moves.");

  if (list.forced) {
    const c = list.forced;
    return {
      san: c.san,
      uci: c.uci,
      source: c.forcedMate ? "forced-mate" : "only-legal-move",
      note: c.forcedMate
        ? "Mate is on the board. No judgment required, and no request spent."
        : "Only one move survives the material filter. No request spent.",
      table: [{ ...c, fit: 1, aggression: 0, looseness: 0, total: 1 }],
    };
  }

  const engineBest = list.candidates.reduce((a, b) => (a.loss <= b.loss ? a : b));

  if (!client) {
    return {
      san: engineBest.san,
      uci: engineBest.uci,
      source: "search-fallback",
      note: "No TypeSafe client configured. Playing the search's own pick.",
      table: list.candidates.map((c) => ({ ...c, fit: 0, aggression: 0, looseness: 0, total: -c.loss })),
    };
  }

  let j: JevJudgment;
  try {
    j = await judge(client, {
      candidates: list.candidates,
      persona,
      position: positionBrief(chess),
    });
  } catch (err) {
    return {
      san: engineBest.san,
      uci: engineBest.uci,
      source: "search-fallback",
      note: `TypeSafe request failed (${(err as Error).message}). Playing the search's own pick.`,
      table: list.candidates.map((c) => ({ ...c, fit: 0, aggression: 0, looseness: 0, total: -c.loss })),
    };
  }

  const table = combine(list.candidates, persona, j);

  // Confidence describes how concentrated the distribution is, not permission
  // to act. When the options are genuinely close, several of them are often
  // fine and a spread distribution is not a failure. This gate exists so a
  // flat distribution hands the decision back to the search rather than
  // picking a near-arbitrary winner by a third decimal place.
  if (j.fitConfidence < persona.minConfidence) {
    return {
      san: engineBest.san,
      uci: engineBest.uci,
      source: "low-confidence-fallback",
      note: `Choice confidence ${j.fitConfidence.toFixed(2)} is below the ${persona.name} threshold of ${persona.minConfidence}. Deferring to the search.`,
      table,
      confidence: j.fitConfidence,
      underPressure: j.underPressure,
      usage: j.usage,
    };
  }

  const winner = table[0];
  return {
    san: winner.san,
    uci: winner.uci,
    source: "jev",
    note: `${persona.name} plays ${winner.san}, conceding ${winner.loss} centipawns against the search's ${engineBest.san}.`,
    table,
    confidence: j.fitConfidence,
    underPressure: j.underPressure,
    usage: j.usage,
  };
}
