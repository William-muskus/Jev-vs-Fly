import type { EngineMove } from "./aiClient";
import { flyAnatomyFromReady, flyThoughtFromReply, type FlyAnatomy, type FlyThought } from "./flyAnatomy";
import {
  explainFlyLoadFailure,
  flyGpuEnabled,
  flyWorkerCrashMessage,
  isFlyWorkerCrash,
  preflightFlyApi,
} from "./flyLoad";
import type { PieceKind, SquareId } from "../core/types";

export type { FlyAnatomy, FlyThought } from "./flyAnatomy";

export type FlyDifficulty = "larva" | "fly" | "superfly";

interface FlyReply {
  type: string;
  id?: number;
  move?: string | null;
  message?: string;
  done?: number;
  total?: number;
  activitySample?: Float32Array | null;
  trace?: Float32Array | null;
  traceSteps?: number;
  step?: number;
  steps?: number;
  sample?: FlyAnatomy["sample"];
  silhouette?: FlyAnatomy["silhouette"];
  legend?: string[];
  features?: { steps?: number };
}

const LOAD_TIMEOUT_MS = 180_000;

/** Worker packets that must not resolve a pending `move` / `eval` request. */
export function isFlyStreamMessage(type: string): boolean {
  return type === "live" || type === "thought" || type === "thinking" || type === "progress" || type === "backend";
}

/**
 * The FlyWire connectome worker from this repo's `web/engine`.
 * Loaded from /engine/worker.js; the blob is served at /model/ by the Python app.
 */
export class FlyClient {
  private worker: Worker | null = null;
  private ready: Promise<void> | null = null;
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: FlyReply) => void; reject: (e: Error) => void }>();
  difficulty: FlyDifficulty = "fly";
  anatomy: FlyAnatomy | null = null;
  onAnatomy: ((anatomy: FlyAnatomy) => void) | null = null;
  onThought: ((thought: FlyThought) => void) | null = null;
  onThinking: ((thinking: boolean) => void) | null = null;
  /** Live sampled activity while a forward/search is still running. Not React state. */
  onLive: ((sample: Float32Array | number[], step: number, steps: number) => void) | null = null;
  private liveListeners = new Set<(sample: Float32Array | number[], step: number, steps: number) => void>();

  subscribeLive(fn: (sample: Float32Array | number[], step: number, steps: number) => void): () => void {
    this.liveListeners.add(fn);
    return () => {
      this.liveListeners.delete(fn);
    };
  }

  private dispatchLive(sample: Float32Array | number[], step: number, steps: number): void {
    if (typeof document !== "undefined") {
      document.documentElement.dataset.flyLive = `${step + 1}/${steps}`;
    }
    try {
      this.onLive?.(sample, step, steps);
    } catch {
      /* a viewer throw must not stall the worker pump */
    }
    for (const fn of this.liveListeners) {
      try {
        fn(sample, step, steps);
      } catch {
        /* ignore */
      }
    }
    this.ackLive();
  }

  /** Release the worker's next timestep only after this sample has been applied. */
  private ackLive(): void {
    try {
      this.worker?.postMessage({ type: "live-ack" });
    } catch {
      /* worker gone */
    }
  }

  load(baseUrl = "/model/"): Promise<void> {
    if (!this.ready) {
      this.ready = this.loadBrain(baseUrl).catch((err) => {
        this.ready = null;
        this.teardown();
        throw new Error(explainFlyLoadFailure(err));
      });
    }
    return this.ready;
  }

  private async loadBrain(baseUrl: string): Promise<void> {
    await preflightFlyApi(baseUrl);
    const wantGpu = flyGpuEnabled();
    try {
      await this.boot(baseUrl, wantGpu);
    } catch (err) {
      if (wantGpu && isFlyWorkerCrash(err)) {
        this.teardown();
        await this.boot(baseUrl, false);
        return;
      }
      throw err;
    }
  }

  private boot(baseUrl: string, gpu: boolean): Promise<void> {
    this.teardown();
    return new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(0);
        reject(new Error("fly brain load timed out"));
      }, LOAD_TIMEOUT_MS);
      this.pending.set(0, {
        resolve: (msg) => {
          clearTimeout(timer);
          const anatomy = flyAnatomyFromReady(msg);
          if (anatomy) {
            this.anatomy = anatomy;
            this.onAnatomy?.(anatomy);
          }
          resolve();
        },
        reject: (err) => {
          clearTimeout(timer);
          reject(err);
        },
      });
      const worker = new Worker("/engine/worker.js", { type: "module", name: "fly-brain" });
      this.worker = worker;
      worker.onmessage = (ev: MessageEvent<FlyReply>) => {
        const msg = ev.data;
        if (msg.type === "progress" || msg.type === "backend") return;
        if (msg.type === "live") {
          if (msg.activitySample) {
            this.dispatchLive(
              msg.activitySample,
              msg.step ?? 0,
              msg.steps && msg.steps > 0 ? msg.steps : 1,
            );
          }
          return;
        }
        if (msg.type === "thinking") {
          this.onThinking?.(true);
          return;
        }
        if (msg.type === "thought") {
          const thought = flyThoughtFromReply(msg);
          if (thought) this.onThought?.(thought);
          return;
        }
        if (msg.type === "ready") {
          const p = this.pending.get(0);
          this.pending.delete(0);
          p?.resolve(msg);
          return;
        }
        if (msg.id && this.pending.has(msg.id)) {
          const p = this.pending.get(msg.id)!;
          this.pending.delete(msg.id);
          if (msg.type === "error") p.reject(new Error(msg.message || "fly error"));
          else p.resolve(msg);
        }
      };
      const fail = (err: Error): void => {
        for (const p of this.pending.values()) p.reject(err);
        this.pending.clear();
      };
      worker.onerror = (ev) => {
        fail(new Error(flyWorkerCrashMessage(ev)));
      };
      worker.onmessageerror = () => {
        fail(new Error("fly worker crashed"));
      };
      // Full worker sample (2048 neurons) and silhouette (6000). Do not thin
      // those arrays here — the hall canvas draws every point, in one ink.
      worker.postMessage({ type: "load", baseUrl, gpu });
    });
  }

  private teardown(): void {
    if (!this.worker) return;
    this.worker.onmessage = null;
    this.worker.onerror = null;
    this.worker.onmessageerror = null;
    this.worker.terminate();
    this.worker = null;
    for (const p of this.pending.values()) p.reject(new Error("fly worker restarted"));
    this.pending.clear();
  }

  async bestMove(fen: string, historyUci: string[]): Promise<EngineMove | null> {
    await this.load();
    const id = this.nextId++;
    this.onThinking?.(true);
    let sawThought = false;
    const prevThought = this.onThought;
    this.onThought = (thought) => {
      sawThought = true;
      prevThought?.(thought);
    };
    try {
      const msg = await new Promise<FlyReply>((resolve, reject) => {
        const timer = setTimeout(() => {
          this.pending.delete(id);
          reject(new Error("fly timed out"));
        }, 90_000);
        this.pending.set(id, {
          resolve: (v) => {
            clearTimeout(timer);
            resolve(v);
          },
          reject: (e) => {
            clearTimeout(timer);
            reject(e);
          },
        });
        this.worker!.postMessage({
          type: "move",
          id,
          fen,
          moves: historyUci,
          difficulty: this.difficulty,
          trace: true,
        });
      });
      if (!sawThought) {
        const thought = flyThoughtFromReply(msg);
        if (thought) prevThought?.(thought);
      }
      if (!msg.move) return null;
      const uci = msg.move;
      return {
        from: uci.slice(0, 2) as SquareId,
        to: uci.slice(2, 4) as SquareId,
        promotion: (uci[4] as PieceKind | undefined) ?? null,
        score: 0,
        depth: 0,
      };
    } finally {
      this.onThought = prevThought;
      this.onThinking?.(false);
    }
  }
}

export const flyClient = new FlyClient();
