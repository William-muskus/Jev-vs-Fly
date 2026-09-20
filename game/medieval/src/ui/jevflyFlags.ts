import type { Faction } from "../core/types";
import type { QualityPreset } from "../scene/quality";
import { QUALITY_ORDER } from "../scene/quality";

/**
 * Query flags for a Jev vs Fly autoplay.
 *
 * `cinema=1` hides the HUD for a board-only capture. Default is off so
 * `?autoplay=1` still shows Jev, Fruit Fly, the herald, and the PGN verdict.
 */
export const JEV_PLAYER_NAME = "Jev";
export const FLY_PLAYER_NAME = "Fruit Fly";
export const HUMAN_PLAYER_NAME = "You";

/** Who the human faces on the Computer tab. */
export type ComputerOpponent = "jev" | "fly";

export function cinemaEnabled(search = typeof window !== "undefined" ? window.location.search : ""): boolean {
  return new URLSearchParams(search).get("cinema") === "1";
}

/** Optional `?quality=low|medium|high|ultra` override for autoplay and debugging. */
export function qualityPresetFromSearch(
  search = typeof window !== "undefined" ? window.location.search : "",
): QualityPreset | null {
  const q = new URLSearchParams(search).get("quality");
  return QUALITY_ORDER.includes(q as QualityPreset) ? (q as QualityPreset) : null;
}

/** Wall-clock and PGN names for the two AIs. */
export function jevFlySideNames(jevWhite = true): { w: string; b: string } {
  return jevWhite
    ? { w: JEV_PLAYER_NAME, b: FLY_PLAYER_NAME }
    : { w: FLY_PLAYER_NAME, b: JEV_PLAYER_NAME };
}

/** Wall-clock names when a human takes one banner against Jev or the fly. */
export function vsComputerSideNames(
  opponent: ComputerOpponent,
  playerColor: Faction,
): { w: string; b: string } {
  const ai = opponent === "jev" ? JEV_PLAYER_NAME : FLY_PLAYER_NAME;
  return playerColor === "w"
    ? { w: HUMAN_PLAYER_NAME, b: ai }
    : { w: ai, b: HUMAN_PLAYER_NAME };
}

/** True when Jev sat on either banner — the victory parchment then shows spend. */
export function jevWasPlayer(sideNames?: { w: string; b: string } | null): boolean {
  return sideNames?.w === JEV_PLAYER_NAME || sideNames?.b === JEV_PLAYER_NAME;
}
