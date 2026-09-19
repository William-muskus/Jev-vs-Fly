import { describe, expect, it } from "vitest";

import { isGlbMagic } from "./wizardRoster";

describe("isGlbMagic", () => {
  it("accepts a glTF-binary header", () => {
    expect(isGlbMagic(new Uint8Array([0x67, 0x6c, 0x54, 0x46, 0, 0, 0, 0]))).toBe(true);
  });

  it("rejects HTML that Vite serves for a missing mesh", () => {
    const html = new TextEncoder().encode("<!doctype html>");
    expect(isGlbMagic(html)).toBe(false);
  });
});
