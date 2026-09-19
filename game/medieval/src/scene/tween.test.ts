import { describe, expect, it } from "vitest";

import { TweenManager } from "./tween";

describe("TweenManager timeScale", () => {
  it("finishes twice as fast at scale 2", async () => {
    const tweens = new TweenManager();
    tweens.setTimeScale(2);
    let value = 0;
    const done = tweens.to({
      duration: 1,
      easing: (t) => t,
      onUpdate: (t) => {
        value = t;
      },
    });
    tweens.update(0.5);
    expect(value).toBeCloseTo(1, 5);
    await done;
  });

  it("holds the authored tempo at scale 1", async () => {
    const tweens = new TweenManager();
    let value = 0;
    const done = tweens.to({
      duration: 1,
      easing: (t) => t,
      onUpdate: (t) => {
        value = t;
      },
    });
    tweens.update(0.5);
    expect(value).toBeCloseTo(0.5, 5);
    tweens.update(0.5);
    expect(value).toBeCloseTo(1, 5);
    await done;
  });
});
