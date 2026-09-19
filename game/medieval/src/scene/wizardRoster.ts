import type { PieceKind } from "../core/types";

const RANKS: PieceKind[] = ["k", "q", "b", "n", "r", "p"];

let pending: Promise<Set<PieceKind>> | null = null;

/** Which converted wizard GLBs are on disk. Empty means use procedural stone. */
export function wizardGlbKinds(): Promise<Set<PieceKind>> {
  if (!pending) {
    pending = fetch("/models/wizard/manifest.json")
      .then((res) => (res.ok ? res.json() : { kinds: [] }))
      .then((body: { kinds?: string[] }) => {
        const kinds = new Set<PieceKind>();
        for (const kind of body.kinds ?? []) {
          if (RANKS.includes(kind as PieceKind)) kinds.add(kind as PieceKind);
        }
        return kinds;
      })
      .catch(() => new Set<PieceKind>());
  }
  return pending;
}
