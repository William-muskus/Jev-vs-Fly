import type { EngineMove } from "./aiClient";
import type { PieceKind, SquareId } from "../core/types";

function uciToMove(uci: string): EngineMove {
  const from = uci.slice(0, 2) as SquareId;
  const to = uci.slice(2, 4) as SquareId;
  const promo = uci[4] as PieceKind | undefined;
  return { from, to, promotion: promo ?? null, score: 0, depth: 0 };
}

/** Server-side Jev: TypeSafe Choice over the legal list. The API key stays on the server. */
export async function jevBestMove(fen: string, strategy = "best_this_turn"): Promise<EngineMove | null> {
  let last: Error = new Error("Jev request failed");
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const res = await fetch("/api/jev-move", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ fen, strategy }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || body.message || `Jev HTTP ${res.status}`);
      }
      const data = (await res.json()) as { uci?: string };
      if (!data.uci) return null;
      return uciToMove(data.uci);
    } catch (err) {
      last = err instanceof Error ? err : new Error(String(err));
      if (attempt === 2) break;
      await new Promise((resolve) => setTimeout(resolve, 800 * 2 ** attempt));
    }
  }
  throw last;
}
