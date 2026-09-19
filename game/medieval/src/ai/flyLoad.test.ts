import { describe, expect, it } from "vitest";

import { explainFlyLoadFailure, flyGpuEnabled, flyWorkerCrashMessage, isFlyWorkerCrash, isFlyWorkerScript } from "./flyLoad";

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
    expect(explainFlyLoadFailure(new Error("fly worker crashed"))).toMatch(/sync-engine/);
  });

  it("does not blame the Python API when the worker file is a path", () => {
    expect(explainFlyLoadFailure(new Error("Unexpected token '.' in /engine/worker.js:1"))).toMatch(/sync-engine/);
    expect(explainFlyLoadFailure(new Error("Unexpected token '.' in /engine/worker.js:1"))).not.toMatch(/python -m game\.server/);
  });
});

describe("isFlyWorkerScript", () => {
  it("accepts the real worker module", () => {
    expect(isFlyWorkerScript("// worker.js — Web Worker\nimport { Chess } from '../vendor/chess.js';\n")).toBe(true);
  });

  it("rejects a Windows git-symlink path file", () => {
    expect(isFlyWorkerScript("../../../../web/engine/worker.js\n")).toBe(false);
  });
});
