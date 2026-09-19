import { describe, expect, it } from "vitest";

import { clockFill, clockFillPercent, clockShare } from "./clockFill";

describe("clockFill", () => {
  it("is empty before either army has thought", () => {
    expect(clockFill(0, 0)).toBe(0);
  });

  it("sits the leader at 100% and the trailer as a share of that", () => {
    expect(clockFill(8_000, 2_000)).toBe(1);
    expect(clockFill(2_000, 8_000)).toBe(0.25);
  });

  it("treats equal clocks as both full", () => {
    expect(clockFill(5_000, 5_000)).toBe(1);
  });

  it("ignores a negative peer so a running clock still fills", () => {
    expect(clockFill(4_000, -10)).toBe(1);
  });
});

describe("clockShare", () => {
  it("is empty before the battle has a total", () => {
    expect(clockShare(0, 0)).toBe(0);
    expect(clockShare(1_000, 0)).toBe(0);
  });

  it("splits the total between the two armies", () => {
    expect(clockShare(3_000, 10_000)).toBe(0.3);
    expect(clockShare(7_000, 10_000)).toBe(0.7);
  });

  it("clamps a side that somehow exceeds the total", () => {
    expect(clockShare(12_000, 10_000)).toBe(1);
  });
});

describe("clockFillPercent", () => {
  it("rounds to a tenth of a percent for the CSS width", () => {
    expect(clockFillPercent(1_000, 3_000)).toBe(33.3);
  });
});
