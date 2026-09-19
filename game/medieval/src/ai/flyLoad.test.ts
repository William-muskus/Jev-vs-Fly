import { describe, expect, it } from "vitest";

import { explainFlyLoadFailure, flyGpuEnabled, flyWorkerCrashMessage, isFlyWorkerCrash } from "./flyLoad";

describe("flyGpuEnabled", () => {
  it("keeps WebGPU off on default autoplay", () => {
    expect(flyGpuEnabled("")).toBe(false);
    expect(flyGpuEnabled("?autoplay=1")).toBe(false);
    expect(flyGpuEnabled("?flyGpu=0")).toBe(false);
  });

  it("opts in only when flyGpu=1", () => {
    expect(flyGpuEnabled("?flyGpu=1")).toBe(true);
    expect(flyGpuEnabled("?autoplay=1&flyGpu=true")).toBe(true);
  });
});

describe("flyWorkerCrashMessage", () => {
  it("falls back when the browser leaves the error empty", () => {
    expect(flyWorkerCrashMessage({})).toBe("fly worker crashed");
    expect(flyWorkerCrashMessage({ message: "" })).toBe("fly worker crashed");
  });

  it("keeps a real parser message", () => {
    expect(flyWorkerCrashMessage({ message: "Unexpected token '<'", filename: "/engine/worker.js", lineno: 1 })).toBe(
      "Unexpected token '<' in /engine/worker.js:1",
    );
  });
});

describe("explainFlyLoadFailure", () => {
  it("points at python -m game.server when the API is down", () => {
    expect(explainFlyLoadFailure(new Error("Failed to fetch"))).toMatch(/python -m game\.server/);
    expect(explainFlyLoadFailure(new Error("cannot fetch /model/brain.json (502)"))).toMatch(/python -m game\.server/);
  });

  it("tells the user to retry after a dead worker", () => {
    expect(isFlyWorkerCrash(new Error("fly worker crashed"))).toBe(true);
    expect(explainFlyLoadFailure(new Error("fly worker crashed"))).toMatch(/python -m game\.server/);
  });
});
