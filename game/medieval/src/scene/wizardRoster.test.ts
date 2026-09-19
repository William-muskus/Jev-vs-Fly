import { describe, expect, it } from "vitest";

import { isGlbMagic, wizardStandEuler, wizardYawRadians } from "./wizardRoster";

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
  it("defaults to no extra yaw", () => {
    expect(wizardYawRadians("")).toBe(0);
    expect(wizardYawRadians("?autoplay=1")).toBe(0);
  });

  it("reads degrees from wizardYaw", () => {
    expect(wizardYawRadians("?wizardYaw=90")).toBeCloseTo(Math.PI / 2);
    expect(wizardYawRadians("?wizardYaw=0")).toBe(0);
    expect(wizardYawRadians("?wizardYaw=180")).toBeCloseTo(Math.PI);
  });
});

describe("wizardStandEuler", () => {
  it("leaves an already upright sculpt alone", () => {
    expect(wizardStandEuler({ x: 2, y: 10, z: 3 }, "")).toEqual({ x: 0, y: 0, z: 0 });
  });

  it("tips a Z-up print onto +Y", () => {
    const e = wizardStandEuler({ x: 2, y: 3, z: 12 }, "");
    expect(e.x).toBeCloseTo(-Math.PI / 2);
    expect(e.z).toBe(0);
  });

  it("can be skipped with wizardUp=none", () => {
    expect(wizardStandEuler({ x: 2, y: 3, z: 12 }, "?wizardUp=none")).toEqual({ x: 0, y: 0, z: 0 });
  });

  it("forces the Z-up tip with wizardUp=z even if Y looks taller", () => {
    const e = wizardStandEuler({ x: 2, y: 12, z: 3 }, "?wizardUp=z");
    expect(e.x).toBeCloseTo(-Math.PI / 2);
    expect(e.z).toBe(0);
  });

  it("tips an X-up print onto +Y", () => {
    const e = wizardStandEuler({ x: 12, y: 2, z: 3 }, "");
    expect(e.x).toBe(0);
    expect(e.z).toBeCloseTo(Math.PI / 2);
  });
});
