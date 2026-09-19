/**
 * Query flags for a Jev vs Fly autoplay.
 *
 * `cinema=1` hides the HUD for a board-only capture. Default is off so
 * `?autoplay=1` still shows Jev, Fly, the herald, and the PGN verdict.
 */
export function cinemaEnabled(search = typeof window !== "undefined" ? window.location.search : ""): boolean {
  return new URLSearchParams(search).get("cinema") === "1";
}
