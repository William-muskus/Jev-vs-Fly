import { describe, expect, it } from "vitest";

import type { LedgerMove } from "./types";
import { heraldCall } from "./wizardChess";

function move(partial: Partial<LedgerMove> & Pick<LedgerMove, "kind" | "from" | "to" | "san">): LedgerMove {
  return {
    ply: 0,
    number: 1,
    color: "w",
    capture: false,
    castle: false,
    promotion: null,
    check: false,
    mate: false,
    ...partial,
  };
}

describe("heraldCall", () => {
  it("commands a quiet developing move", () => {
    expect(heraldCall(move({ kind: "n", from: "g1", to: "f3", san: "Nf3" }))).toBe("White knight to F3!");
  });

  it("names a smash capture", () => {
    expect(
      heraldCall(move({ color: "b", kind: "q", from: "d8", to: "d4", san: "Qxd4", capture: true })),
    ).toBe("Black queen takes on D4!");
  });

  it("calls checkmate on the capturing blow", () => {
    expect(
      heraldCall(move({ kind: "q", from: "h5", to: "f7", san: "Qxf7#", capture: true, check: true, mate: true })),
    ).toBe("White queen takes on F7! Checkmate!");
  });

  it("calls a kingside castle", () => {
    expect(heraldCall(move({ kind: "k", from: "e1", to: "g1", san: "O-O", castle: true }))).toBe(
      "White king, castle kingside!",
    );
  });
});
