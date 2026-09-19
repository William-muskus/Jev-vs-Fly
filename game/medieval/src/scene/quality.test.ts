import { describe, expect, it } from "vitest";

import { QUALITY_SETTINGS } from "./quality";

describe("low graphics", () => {
  it("drops the standing-crowd work that still ran when idle clips were off", () => {
    const low = QUALITY_SETTINGS.low;
    expect(low.idleAnimations).toBe(false);
    expect(low.contactShadows).toBe(false);
    expect(low.troopCount).toBe(0);
    expect(low.smokeCount).toBe(0);
    expect(low.ashCount).toBe(0);
    expect(low.campfires).toBe(0);
    expect(low.dustCount).toBe(0);
    expect(low.captureParticles).toBeLessThanOrEqual(8);
    expect(low.postFx).toBe(false);
    expect(low.shadows).toBe(false);
  });
});
