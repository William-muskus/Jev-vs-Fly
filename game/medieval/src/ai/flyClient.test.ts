import { describe, expect, it } from "vitest";

import { isFlyStreamMessage } from "./flyClient";

describe("isFlyStreamMessage", () => {
  it("keeps live activity from resolving a pending move", () => {
    expect(isFlyStreamMessage("live")).toBe(true);
    expect(isFlyStreamMessage("thought")).toBe(true);
    expect(isFlyStreamMessage("thinking")).toBe(true);
    expect(isFlyStreamMessage("move")).toBe(false);
    expect(isFlyStreamMessage("eval")).toBe(false);
    expect(isFlyStreamMessage("error")).toBe(false);
  });
});
