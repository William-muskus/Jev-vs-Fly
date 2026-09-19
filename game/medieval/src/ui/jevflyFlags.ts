/**
 * Query flags for a Jev vs Fly autoplay.
 *
 * `cinema=1` hides the HUD for a board-only capture. Default is off so
 * `?autoplay=1` still shows Jev, Fruit Fly, the herald, and the PGN verdict.
 */
export const JEV_PLAYER_NAME = "Jev";
export const FLY_PLAYER_NAME = "Fruit Fly";

export function cinemaEnabled(search = typeof window !== "undefined" ? window.location.search : ""): boolean {
  return new URLSearchParams(search).get("cinema") === "1";
}

/** Wall-clock and PGN names for the two AIs. */
export function jevFlySideNames(jevWhite = true): { w: string; b: string } {
  return jevWhite
    ? { w: JEV_PLAYER_NAME, b: FLY_PLAYER_NAME }
    : { w: FLY_PLAYER_NAME, b: JEV_PLAYER_NAME };
}
