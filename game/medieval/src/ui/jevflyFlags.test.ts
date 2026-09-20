import { describe, expect, it } from "vitest";

import {
  cinemaEnabled,
  FLY_PLAYER_NAME,
  HUMAN_PLAYER_NAME,
  JEV_PLAYER_NAME,
  jevFlySideNames,
  jevWasPlayer,
  qualityPresetFromSearch,
  vsComputerSideNames,
} from "./jevflyFlags";

describe("cinemaEnabled", () => {
  it("keeps the HUD on default autoplay", () => {
    expect(cinemaEnabled("")).toBe(false);
    expect(cinemaEnabled("?autoplay=1")).toBe(false);
    expect(cinemaEnabled("?autoplay=1&cinema=0")).toBe(false);
  });

  it("hides the HUD only when cinema=1", () => {
    expect(cinemaEnabled("?cinema=1")).toBe(true);
    expect(cinemaEnabled("?autoplay=1&cinema=1")).toBe(true);
  });
});

describe("qualityPresetFromSearch", () => {
  it("reads a graphics preset from the query string", () => {
    expect(qualityPresetFromSearch("")).toBeNull();
    expect(qualityPresetFromSearch("?autoplay=1")).toBeNull();
    expect(qualityPresetFromSearch("?quality=low")).toBe("low");
    expect(qualityPresetFromSearch("?autoplay=1&quality=low")).toBe("low");
    expect(qualityPresetFromSearch("?quality=potato")).toBeNull();
  });
});
describe("jevFlySideNames", () => {
  it("labels the wall clock Jev and Fruit Fly", () => {
    expect(jevFlySideNames(true)).toEqual({ w: JEV_PLAYER_NAME, b: FLY_PLAYER_NAME });
    expect(jevFlySideNames(false)).toEqual({ w: FLY_PLAYER_NAME, b: JEV_PLAYER_NAME });
    expect(FLY_PLAYER_NAME).toBe("Fruit Fly");
  });
});

describe("vsComputerSideNames", () => {
  it("puts You on the chosen banner against Jev or the fly", () => {
    expect(vsComputerSideNames("jev", "w")).toEqual({ w: HUMAN_PLAYER_NAME, b: JEV_PLAYER_NAME });
    expect(vsComputerSideNames("jev", "b")).toEqual({ w: JEV_PLAYER_NAME, b: HUMAN_PLAYER_NAME });
    expect(vsComputerSideNames("fly", "w")).toEqual({ w: HUMAN_PLAYER_NAME, b: FLY_PLAYER_NAME });
    expect(vsComputerSideNames("fly", "b")).toEqual({ w: FLY_PLAYER_NAME, b: HUMAN_PLAYER_NAME });
  });
});

describe("jevWasPlayer", () => {
  it("is true only when Jev sat on a banner", () => {
    expect(jevWasPlayer({ w: "Jev", b: "Fruit Fly" })).toBe(true);
    expect(jevWasPlayer({ w: "You", b: "Jev" })).toBe(true);
    expect(jevWasPlayer({ w: "You", b: "Fruit Fly" })).toBe(false);
    expect(jevWasPlayer(null)).toBe(false);
  });
});
