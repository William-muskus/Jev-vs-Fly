import { describe, expect, it } from "vitest";

import { GameController } from "./gameController";

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

describe("countdown clock publishes", () => {
  it("does not rebuild the HUD ten times a second while the printed time is unchanged", async () => {
    const controller = new GameController();
    let publishes = 0;
    controller.on("state", () => {
      publishes += 1;
    });
    controller.start({ mode: "hotseat", difficulty: "easy", playerColor: "w", clockMinutes: 5 });
    const afterStart = publishes;
    await wait(350);
    expect(publishes - afterStart).toBeLessThanOrEqual(2);
    controller.dispose();
  });
});
