import { describe, expect, it } from "vitest";

import { flyAnatomyFromReady, flyThoughtFromReply } from "./flyAnatomy";

describe("flyAnatomyFromReady", () => {
  it("is empty until the worker has sent a sample", () => {
    expect(flyAnatomyFromReady({})).toBeNull();
  });

  it("keeps the connectome sample the canvas plots", () => {
    const sample = { idx: [0], xy: [0.1, 0.2], cls: [1] };
    const silhouette = { xy: [0, 0, 1, 1], cls: [1, 1] };
    const anatomy = flyAnatomyFromReady({ sample, silhouette, legend: ["optic"], features: { steps: 16 } });
    expect(anatomy?.legend).toEqual(["optic"]);
    expect(anatomy?.steps).toBe(16);
    expect(anatomy?.sample).toBe(sample);
  });
});

describe("flyThoughtFromReply", () => {
  it("is empty without activity", () => {
    expect(flyThoughtFromReply({})).toBeNull();
  });

  it("carries a recurrent trace for replay", () => {
    const trace = new Float32Array([0.1, 0.4, 0.8, 0.2]);
    const thought = flyThoughtFromReply({ trace, traceSteps: 2, activitySample: new Float32Array([0.8, 0.2]) });
    expect(thought?.traceSteps).toBe(2);
    expect(thought?.trace).toBe(trace);
  });
});
