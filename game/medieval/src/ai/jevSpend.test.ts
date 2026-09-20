import { describe, expect, it } from "vitest";

import {
  addJevUsage,
  emptyJevUsage,
  formatJevSpend,
  formatJevUsd,
  jevUsd,
  parseJevUsage,
} from "./jevSpend";

describe("parseJevUsage", () => {
  it("reads token counts and ignores junk", () => {
    expect(parseJevUsage({ input_tokens: 11, output_tokens: 2 })).toEqual({
      input_tokens: 11,
      output_tokens: 2,
    });
    expect(parseJevUsage(null)).toEqual(emptyJevUsage());
    expect(parseJevUsage({ input_tokens: "12.9" })).toEqual({ input_tokens: 12, output_tokens: 0 });
  });
});

describe("jevUsd", () => {
  it("bills input tokens at $0.042 per million", () => {
    expect(jevUsd(1_000_000)).toBeCloseTo(0.042);
    expect(jevUsd(28_000)).toBeCloseTo(0.001176);
    expect(jevUsd(0)).toBe(0);
  });
});

describe("formatJevSpend", () => {
  it("shows dollars and the token total", () => {
    expect(formatJevUsd(0)).toBe("$0.00");
    expect(formatJevSpend({ input_tokens: 28_000, output_tokens: 400 })).toBe(
      "Jev · $0.0012 · 28,400 tokens",
    );
    expect(addJevUsage(emptyJevUsage(), { input_tokens: 11, output_tokens: 2 })).toEqual({
      input_tokens: 11,
      output_tokens: 2,
    });
  });
});
