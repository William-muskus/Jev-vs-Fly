import { describe, expect, it } from "vitest";

import { isGlbMagic, wizardYawRadians } from "./wizardRoster";

describe("isGlbMagic", () => {
  it("accepts a glTF-binary header", () => {
    expect(isGlbMagic(new Uint8Array([0x67, 0x6c, 0x54, 0x46, 0, 0, 0, 0]))).toBe(true);
  });

  it("rejects HTML that Vite serves for a missing mesh", () => {
    const html = new TextEncoder().encode("<!doctype html>");
    expect(isGlbMagic(html)).toBe(false);
  });
});

describe("wizardYawRadians", () => {
  it("defaults to a half turn so Cults sculpts face the enemy", () => {
    expect(wizardYawRadians("")).toBeCloseTo(Math.PI);
    expect(wizardYawRadians("?autoplay=1")).toBeCloseTo(Math.PI);
  });

  it("reads degrees from wizardYaw", () => {
    expect(wizardYawRadians("?wizardYaw=90")).toBeCloseTo(Math.PI / 2);
    expect(wizardYawRadians("?wizardYaw=0")).toBe(0);
    expect(wizardYawRadians("?wizardYaw=270")).toBeCloseTo((3 * Math.PI) / 2);
  });
});
