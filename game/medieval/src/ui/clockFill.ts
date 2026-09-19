/**
 * Visual fill for the two accumulating AI clocks.
 *
 * There is no time control on a Jev vs Fly autoplay, so a bar cannot drain
 * toward zero. Each army's meter is therefore the share of the longer clock:
 * the leader sits at 100% and the other fills as it catches up. The stacked
 * duel bar uses {@link clockShare} of the battle total, so the two segments
 * always add to one once any thinking has happened.
 */

/** Fill of one army's clock against the longer of the two. Leader is 100%. */
export function clockFill(sideMs: number, peerMs: number): number {
  const cap = Math.max(0, sideMs, peerMs);
  if (cap <= 0) return 0;
  return Math.min(1, Math.max(0, sideMs / cap));
}

/** Share of the battle's total clock. Both sides sum to 1 once time has run. */
export function clockShare(sideMs: number, totalMs: number): number {
  if (totalMs <= 0) return 0;
  return Math.min(1, Math.max(0, sideMs / totalMs));
}

/** CSS width percent, one decimal so a half-second tick is visible. */
export function clockFillPercent(sideMs: number, peerMs: number): number {
  return Math.round(clockFill(sideMs, peerMs) * 1000) / 10;
}
