import { describe, expect, it } from "vitest";

import { FlyBrainView } from "./FlyBrainView";

describe("FlyBrainView", () => {
  it("exports a component Vite can transform", () => {
    expect(typeof FlyBrainView).toBe("function");
  });
});
