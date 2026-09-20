import { describe, expect, it } from "vitest";

import { formatElapsed } from "./elapsedFormat";

describe("formatElapsed", () => {
  it("prints m:ss and h:mm:ss", () => {
    expect(formatElapsed(0)).toBe("0:00");
    expect(formatElapsed(1_250)).toBe("0:01");
    expect(formatElapsed(125_000)).toBe("2:05");
    expect(formatElapsed(3_661_000)).toBe("1:01:01");
  });
});
