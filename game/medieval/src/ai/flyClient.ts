import type { EngineMove } from "./aiClient";
import type { PieceKind, SquareId } from "../core/types";

type FlyDifficulty = "larva" | "fly" | "superfly";

interface FlyReply {
  type: string;
  move?: string | null;
  message?: string;
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

  load(baseUrl = "/model/"): Promise<void> {
    if (this.ready) return this.ready;
    this.worker = new Worker("/engine/worker.js", { type: "module" });
    this.worker.onmessage = (ev: MessageEvent<FlyReply & { id?: number }>) => {
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
    this.worker.onerror = (ev) => {
      for (const p of this.pending.values()) p.reject(new Error(ev.message || "fly worker crashed"));
      this.pending.clear();
    };
    this.ready = new Promise<void>((resolve, reject) => {
      this.pending.set(0, {
        resolve: () => resolve(),
        reject,
      });
      this.worker!.postMessage({ type: "load", baseUrl });
    });
    return this.ready;
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
