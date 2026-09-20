import { describe, expect, it } from "vitest";

import { MainMenu, type MatchConfig } from "./MainMenu";

describe("MainMenu", () => {
  it("exports a component Vite can transform", () => {
    expect(typeof MainMenu).toBe("function");
  });

  it("carries a Computer-tab opponent on MatchConfig", () => {
    const vsJev: MatchConfig = {
      mode: "ai",
      difficulty: "medium",
      playerColor: "w",
      clockMinutes: null,
      opponent: "jev",
    };
    const vsFly: MatchConfig = {
      ...vsJev,
      opponent: "fly",
      flyDifficulty: "superfly",
      playerColor: "b",
    };
    expect(vsJev.opponent).toBe("jev");
    expect(vsFly.opponent).toBe("fly");
    expect(vsFly.flyDifficulty).toBe("superfly");
  });

  it("does not use MatchConfig for the Jev vs Fly watch duel", () => {
    const watch: MatchConfig = {
      mode: "demo",
      difficulty: "medium",
      playerColor: "w",
      clockMinutes: null,
    };
    expect(watch.opponent).toBeUndefined();
  });
});
