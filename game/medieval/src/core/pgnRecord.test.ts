import { Chess } from "chess.js";
import { describe, expect, it } from "vitest";

import { GameController } from "./gameController";
import { PGN_EVENT, PGN_SITE, pgnDateTag, pgnResultTag, stampChessPgn } from "./pgnRecord";

describe("pgnResultTag", () => {
  it("maps a hall ending onto the PGN result", () => {
    expect(pgnResultTag(null)).toBe("*");
    expect(pgnResultTag({ winner: "w", reason: "checkmate" })).toBe("1-0");
    expect(pgnResultTag({ winner: "b", reason: "checkmate" })).toBe("0-1");
    expect(pgnResultTag({ winner: null, reason: "stalemate" })).toBe("1/2-1/2");
    expect(pgnResultTag({ winner: null, reason: "plycap" })).toBe("1/2-1/2");
  });
});

describe("pgnDateTag", () => {
  it("prints UTC YYYY.MM.DD", () => {
    expect(pgnDateTag(new Date("2026-09-20T12:00:00Z"))).toBe("2026.09.20");
  });
});

describe("stampChessPgn", () => {
  it("replaces chess.js placeholder tags and the trailing star", () => {
    const chess = new Chess();
    chess.move("f3");
    chess.move("e5");
    chess.move("g4");
    chess.move("Qh4");
    const pgn = stampChessPgn(chess, {
      white: "Jev",
      black: "Fruit Fly",
      result: { winner: "b", reason: "checkmate" },
      now: new Date("2026-09-20T12:00:00Z"),
    });
    expect(pgn).toContain(`[Event "${PGN_EVENT}"]`);
    expect(pgn).toContain(`[Site "${PGN_SITE}"]`);
    expect(pgn).toContain('[Date "2026.09.20"]');
    expect(pgn).toContain('[White "Jev"]');
    expect(pgn).toContain('[Black "Fruit Fly"]');
    expect(pgn).toContain('[Result "0-1"]');
    expect(pgn).toContain('[Termination "checkmate"]');
    expect(pgn).toContain("Qh4#");
    expect(pgn.trim().endsWith("0-1")).toBe(true);
    expect(pgn).not.toContain('[White "?"]');
    expect(pgn).not.toMatch(/# \*$/);
  });
});

describe("GameController PGN snapshot", () => {
  it("stamps names and 0-1 after fool's mate", async () => {
    const controller = new GameController();
    controller.setThinkFloorMs(0);
    controller.setMovers({}, { w: "Jev", b: "Fruit Fly" });
    controller.start({ mode: "hotseat", difficulty: "easy", playerColor: "w", clockMinutes: null });
    expect(await controller.tryMove("f2", "f3")).toBe(true);
    expect(await controller.tryMove("e7", "e5")).toBe(true);
    expect(await controller.tryMove("g2", "g4")).toBe(true);
    expect(await controller.tryMove("d8", "h4")).toBe(true);
    const snapshot = controller.getSnapshot();
    expect(snapshot.status).toBe("over");
    expect(snapshot.result).toEqual({ winner: "b", reason: "checkmate" });
    expect(snapshot.pgn).toContain('[White "Jev"]');
    expect(snapshot.pgn).toContain('[Black "Fruit Fly"]');
    expect(snapshot.pgn).toContain('[Result "0-1"]');
    expect(snapshot.pgn).toContain("Qh4#");
    expect(snapshot.pgn.trim().endsWith("0-1")).toBe(true);
    expect(snapshot.pgn).not.toContain('[White "?"]');
    expect(snapshot.pgn).not.toMatch(/\*\s*$/);
    controller.dispose();
  });

  it("stamps 1-0 when Ivory resigns", () => {
    const controller = new GameController();
    controller.setMovers({}, { w: "You", b: "Jev" });
    controller.start({ mode: "ai", difficulty: "easy", playerColor: "w", clockMinutes: null });
    controller.resign();
    const snapshot = controller.getSnapshot();
    expect(snapshot.result?.reason).toBe("resignation");
    expect(snapshot.pgn).toContain('[White "You"]');
    expect(snapshot.pgn).toContain('[Black "Jev"]');
    expect(snapshot.pgn).toContain('[Result "0-1"]');
    expect(snapshot.pgn).toContain('[Termination "resignation"]');
    controller.dispose();
  });
});
