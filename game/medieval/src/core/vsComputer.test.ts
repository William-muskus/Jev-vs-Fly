import { describe, expect, it } from "vitest";

import { GameController } from "./gameController";

function waitFor(predicate: () => boolean, timeoutMs = 4000): Promise<void> {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const tick = (): void => {
      if (predicate()) {
        resolve();
        return;
      }
      if (Date.now() - started > timeoutMs) {
        reject(new Error("timed out waiting for the board"));
        return;
      }
      setTimeout(tick, 10);
    };
    tick();
  });
}

describe("vs computer custom movers", () => {
  it("lets a human play against a mover on the other banner", async () => {
    const controller = new GameController();
    controller.setThinkFloorMs(0);
    const fens: string[] = [];
    controller.setMovers(
      {
        b: async (fen) => {
          fens.push(fen);
          return { from: "e7", to: "e5", promotion: null, score: 0, depth: 0 };
        },
      },
      { w: "You", b: "Fruit Fly" },
    );
    controller.start({ mode: "ai", difficulty: "easy", playerColor: "w", clockMinutes: null });
    expect(controller.getSnapshot().sideNames).toEqual({ w: "You", b: "Fruit Fly" });
    expect(controller.getSnapshot().turn).toBe("w");
    expect(fens).toEqual([]);

    await controller.tryMove("e2", "e4");
    await waitFor(
      () => controller.getSnapshot().sanList.length >= 2 && !controller.getSnapshot().thinking,
    );

    expect(fens).toHaveLength(1);
    expect(controller.getSnapshot().sanList.join(" ")).toMatch(/e4/);
    expect(controller.getSnapshot().sanList.join(" ")).toMatch(/e5/);
    expect(controller.getSnapshot().turn).toBe("w");
    controller.dispose();
  });

  it("asks the computer mover first when the human takes Obsidian", async () => {
    const controller = new GameController();
    controller.setThinkFloorMs(0);
    let asked = 0;
    controller.setMovers(
      {
        w: async () => {
          asked += 1;
          return { from: "e2", to: "e4", promotion: null, score: 0, depth: 0 };
        },
      },
      { w: "Jev", b: "You" },
    );
    controller.start({ mode: "ai", difficulty: "easy", playerColor: "b", clockMinutes: null });
    await waitFor(() => asked === 1 && controller.getSnapshot().turn === "b");
    expect(controller.getSnapshot().sideNames).toEqual({ w: "Jev", b: "You" });
    expect(controller.getSnapshot().sanList.join(" ")).toMatch(/e4/);
    controller.dispose();
  });
});
