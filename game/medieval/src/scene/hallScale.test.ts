import { describe, expect, it } from "vitest";

import { TILE } from "./board";
import { PIECE_HEIGHT, PIECE_HEIGHT_SCALE } from "./pieces";
import { BOARD_REACH } from "./viewport";

describe("wizard hall scale", () => {
  it("stands the army three times taller than the original Staunton tokens", () => {
    expect(PIECE_HEIGHT_SCALE).toBe(3);
    expect(PIECE_HEIGHT.p).toBeCloseTo(0.78 * 3);
    expect(PIECE_HEIGHT.k).toBeCloseTo(1.12 * 3);
    expect(PIECE_HEIGHT.k).toBeGreaterThan(PIECE_HEIGHT.p);
  });

  it("gives the 3× army a square wide enough for a Cults plinth", () => {
    expect(TILE).toBeGreaterThan(1.8);
    expect(TILE).toBeLessThan(2.5);
    expect(BOARD_REACH).toBeGreaterThan(TILE * 4);
  });
});
