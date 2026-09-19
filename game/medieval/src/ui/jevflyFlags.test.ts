import { describe, expect, it } from "vitest";

import { cinemaEnabled, FLY_PLAYER_NAME, JEV_PLAYER_NAME, jevFlySideNames } from "./jevflyFlags";

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

describe("jevFlySideNames", () => {
  it("labels the wall clock Jev and Fruit Fly", () => {
    expect(jevFlySideNames(true)).toEqual({ w: JEV_PLAYER_NAME, b: FLY_PLAYER_NAME });
    expect(jevFlySideNames(false)).toEqual({ w: FLY_PLAYER_NAME, b: JEV_PLAYER_NAME });
    expect(FLY_PLAYER_NAME).toBe("Fruit Fly");
  });
});
