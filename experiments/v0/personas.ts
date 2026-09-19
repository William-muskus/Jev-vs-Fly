/**
 * A persona is two separate things kept deliberately apart:
 *
 *   plan    - prose the model reads. Changing it changes the judgment.
 *   weights - numbers code applies to the model's answers. Changing these
 *             reruns nothing, because the evidence and the questions are
 *             unchanged. That separation is the whole point of scoring
 *             dimensions once and letting policy live in code.
 */

export interface Persona {
  name: string;
  /** Read by Jev. Written as an intention, not as an evaluation function. */
  plan: string;
  weights: {
    /** Trust in Jev's relative pick across the shortlist. */
    fit: number;
    /** Appetite for moves that create threats. */
    aggression: number;
    /** Penalty for moves that loosen our own position. Higher is more cautious. */
    looseness: number;
    /** Penalty per unit of material conceded against the engine's best move. */
    material: number;
  };
  /**
   * Below this Choice confidence, code stops deferring and plays the
   * lowest-material-loss move instead. Confidence summarises how concentrated
   * the distribution is, not whether acting is correct, so this is a threshold
   * to tune on real games rather than a guarantee.
   */
  minConfidence: number;
  /** Centipawns of material the persona will concede for a move it likes. */
  marginCp: number;
  searchDepth: number;
}

export const PERSONAS: Record<string, Persona> = {
  romantic: {
    name: "Romantic",
    plan:
      "I play in the style of the 1850s attacking masters. I want my pieces pointing at " +
      "the enemy king as early as possible. I will give up a pawn, and sometimes a piece, " +
      "to open lines towards the king or to keep the opponent from ever getting settled. " +
      "I would rather play a move that asks the opponent a hard question than a move that " +
      "quietly improves my worst piece. A slow manoeuvring game is a game I am losing.",
    weights: { fit: 1.0, aggression: 0.85, looseness: 0.1, material: 0.35 },
    minConfidence: 0.3,
    marginCp: 240,
    searchDepth: 2,
  },

  solid: {
    name: "Solid",
    plan:
      "I play for a position with no weaknesses. I castle early, I keep my pawn structure " +
      "intact, and I do not create targets. I prefer a move that improves my worst-placed " +
      "piece over a move that starts a fight. I am happy to trade into a slightly better " +
      "endgame. I only attack when the opponent has already given me something concrete " +
      "to attack, and I never open lines near my own king to do it.",
    weights: { fit: 1.0, aggression: 0.1, looseness: 0.9, material: 0.9 },
    minConfidence: 0.35,
    marginCp: 60,
    searchDepth: 2,
  },

  hustler: {
    name: "Coffeehouse hustler",
    plan:
      "I play to make the position messy and to set traps. I want moves that look wrong " +
      "but are hard to refute over the board, and moves that give my opponent a chance to " +
      "go badly astray. I value a cheap threat that costs my opponent time even when a " +
      "quiet move is objectively better. I avoid clean simplifying trades, because a clear " +
      "position is a position where the better player wins.",
    weights: { fit: 1.1, aggression: 0.6, looseness: 0.25, material: 0.45 },
    minConfidence: 0.28,
    marginCp: 170,
    searchDepth: 2,
  },

  /** No plan bias at all. Use this to measure what Jev contributes over the search. */
  neutral: {
    name: "Neutral",
    plan:
      "I have no stylistic preference. I want the move that is objectively best for the " +
      "position in front of me, judged on piece activity, king safety, and pawn structure.",
    weights: { fit: 1.0, aggression: 0.0, looseness: 0.35, material: 0.7 },
    minConfidence: 0.35,
    marginCp: 70,
    searchDepth: 2,
  },
};
