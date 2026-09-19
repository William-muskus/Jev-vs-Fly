import { describe, expect, it } from "vitest";

import { cinemaEnabled } from "./jevflyFlags";

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
