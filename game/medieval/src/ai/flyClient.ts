import type { EngineMove } from "./aiClient";
import {
  explainFlyLoadFailure,
  flyGpuEnabled,
  flyWorkerCrashMessage,
  isFlyWorkerCrash,
  preflightFlyApi,
} from "./flyLoad";
import type { PieceKind, SquareId } from "../core/types";

type FlyDifficulty = "larva" | "fly" | "superfly";

interface FlyReply {
  type: string;
  move?: string | null;
  message?: string;
}

const LOAD_TIMEOUT_MS = 180_000;

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
        resolve: () => {
          clearTimeout(timer);
          resolve();
        },
        reject: (err) => {
          clearTimeout(timer);
          reject(err);
        },
      });
      const worker = new Worker("/engine/worker.js", { type: "module", name: "fly-brain" });
      this.worker = worker;
      worker.onmessage = (ev: MessageEvent<FlyReply & { id?: number }>) => {
        const msg = ev.data;
        if (msg.type === "progress" || msg.type === "thinking" || msg.type === "backend") return;
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
      this.worker!.postMessage({ type: "move", id, fen, moves: historyUci, difficulty: this.difficulty });
    });
    if (!msg.move) return null;
    const uci = msg.move;
    return {
      from: uci.slice(0, 2) as SquareId,
      to: uci.slice(2, 4) as SquareId,
      promotion: (uci[4] as PieceKind | undefined) ?? null,
      score: 0,
      depth: 0,
    };
  }
}

export const flyClient = new FlyClient();
