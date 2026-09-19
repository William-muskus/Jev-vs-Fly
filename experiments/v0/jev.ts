/**
 * Layer 3: the one place the model is asked anything.
 *
 * Every question below goes in a single request. They are independent
 * judgments over the same state, they run in parallel, and none of them can
 * see another's answer, which is exactly the shape this needs. Batching also
 * keeps the per-move cost close to one request instead of 2N + 2.
 *
 * Nothing here asks Jev to calculate, count, or compare numbers. It is asked
 * what a move means and whether that meaning matches an intention.
 */

import { TypeSafeClient, choice, noul, score, type Questions } from "@typesafe-ai/sdk";
import type { Candidate, PositionBrief } from "./tactics.js";
import type { Persona } from "./personas.js";

const AGGRESSION_LEVELS = [
  "A quiet move. It improves a piece, completes development, or holds the position " +
    "steady, and makes no threat the opponent has to answer.",
  "A constructive move. It gains space or prepares an attack for later, but the " +
    "opponent is free to carry on with their own plan this move.",
  "An active move. It creates a direct threat against an enemy piece, an enemy pawn, " +
    "or the squares around the enemy king, and the opponent has to respond to it.",
  "A committal attacking move. It commits material or the safety of my own king to an " +
    "attack and relies on the initiative to pay that back.",
] as const;

const LOOSENESS_LEVELS = [
  "The move keeps my own position tight. No new weak squares, no new weak pawns, and " +
    "the cover in front of my king is untouched.",
  "The move makes a small concession, such as one slightly awkward piece or one pawn " +
    "that is now a little harder to defend.",
  "The move noticeably loosens my position. It opens a line towards my own king or " +
    "leaves a weak square that the opponent can use for a long time.",
  "The move leaves my king or a key defender badly exposed, and depends on the " +
    "opponent failing to find the punishment.",
] as const;

export interface JevJudgment {
  /** Jev's relative pick across the shortlist, keyed by candidate id. */
  fit: Record<string, number>;
  fitConfidence: number;
  /** 0 to 1, normalised from the 0 to 3 rubric. */
  aggression: Record<string, number>;
  looseness: Record<string, number>;
  /** Probability that the position calls for defending rather than playing on. */
  underPressure: number;
  usage: { input_tokens: number; output_tokens: number };
}

export interface JudgeInput {
  candidates: Candidate[];
  persona: Persona;
  position: PositionBrief;
}

export async function judge(client: TypeSafeClient, input: JudgeInput): Promise<JevJudgment> {
  const { candidates, persona, position } = input;

  const candidateState: Record<string, { move: string; effect: string }> = {};
  const pickCriteria: Record<string, string> = {};
  for (const c of candidates) {
    candidateState[c.id] = { move: c.san, effect: c.effect };
    pickCriteria[c.id] = `Play ${c.san}. This move ${c.effect}`;
  }

  const state = {
    plan: persona.plan,
    position,
    candidates: candidateState,
  };

  const questions: Questions = {
    pick: choice(
      {
        task: "Choose the move that best serves the intention written in `plan`.",
        context:
          "`position` describes the position. Every option below is a legal move that has " +
          "already been checked for material soundness, so choose on intention and meaning, " +
          "not on whether the move loses material.",
      },
      pickCriteria,
    ),

    under_pressure: noul(
      "Reading `position` and the recent moves in `position.recent_moves`, is my own king " +
        "under a direct attack that I should spend this move answering?",
      {
        true: "There is an immediate threat against my king or the squares next to it that " +
          "will do damage if I ignore it this move.",
        false: "There is no immediate threat against my king. I am free to carry on with my " +
          "own plan.",
      },
    ),
  };

  for (const c of candidates) {
    questions[`aggression_${c.id}`] = score(
      `The move \`candidates.${c.id}.move\` does the following: \`candidates.${c.id}.effect\`. ` +
        `How much does that move attack the opponent?`,
      AGGRESSION_LEVELS,
    );
    questions[`looseness_${c.id}`] = score(
      `The move \`candidates.${c.id}.move\` does the following: \`candidates.${c.id}.effect\`. ` +
        `How much does that move weaken my own position?`,
      LOOSENESS_LEVELS,
    );
  }

  const result = await client.systemOne({ state, questions });
  const a = result.answers as Record<string, any>;

  const fit: Record<string, number> = {};
  const aggression: Record<string, number> = {};
  const looseness: Record<string, number> = {};

  for (const c of candidates) {
    fit[c.id] = a.pick?.probabilities?.[c.id] ?? 0;
    aggression[c.id] = (a[`aggression_${c.id}`]?.score ?? 0) / (AGGRESSION_LEVELS.length - 1);
    looseness[c.id] = (a[`looseness_${c.id}`]?.score ?? 0) / (LOOSENESS_LEVELS.length - 1);
  }

  return {
    fit,
    fitConfidence: a.pick?.confidence ?? 0,
    aggression,
    looseness,
    underPressure: a.under_pressure?.noul ?? 0,
    usage: result.usage,
  };
}
