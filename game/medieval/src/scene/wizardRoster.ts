import type { PieceKind } from "../core/types";

const RANKS: PieceKind[] = ["k", "q", "b", "n", "r", "p"];

let pending: Promise<Set<PieceKind>> | null = null;

/** glTF-binary magic (`glTF`) — Vite's SPA fallback returns HTML with status 200. */
export function isGlbMagic(bytes: ArrayBuffer | Uint8Array): boolean {
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  return view.length >= 4 && view[0] === 0x67 && view[1] === 0x6c && view[2] === 0x54 && view[3] === 0x46;
}

function looksLikeHtml(contentType: string): boolean {
  const type = contentType.toLowerCase();
  return type.includes("html") || type.includes("javascript") || type.startsWith("text/");
}

async function probeKind(kind: PieceKind): Promise<boolean> {
  const url = `/models/wizard/${kind}.glb`;
  try {
    const head = await fetch(url, { method: "HEAD" });
    if (!head.ok) return false;
    const type = head.headers.get("content-type") ?? "";
    if (looksLikeHtml(type)) return false;
    if (type.toLowerCase().includes("gltf") || type.toLowerCase().includes("octet-stream") || type.toLowerCase().includes("model")) {
      return true;
    }
    const peek = await fetch(url, { headers: { Range: "bytes=0-3" } });
    if (!peek.ok) return false;
    if (looksLikeHtml(peek.headers.get("content-type") ?? "")) return false;
    return isGlbMagic(await peek.arrayBuffer());
  } catch {
    return false;
  }
}

/** Test helper: drop the cached roster so a later fetch runs again. */
export function resetWizardRoster(): void {
  pending = null;
}

/**
 * Yaw applied to Cults GLBs so local +Z (the hall's "front") matches the sculpt.
 * nbauchat's set faces the camera as exported; 180° turns them toward the enemy.
 * Override live with `?wizardYaw=90` (degrees) without reconverting.
 */
export function wizardYawRadians(search = typeof window !== "undefined" ? window.location.search : ""): number {
  const raw = new URLSearchParams(search).get("wizardYaw");
  if (raw === null || raw === "") return Math.PI;
  const deg = Number(raw);
  if (!Number.isFinite(deg)) return Math.PI;
  return (deg * Math.PI) / 180;
}

/**
 * Which converted wizard GLBs are actually being served.
 *
 * The tracked `manifest.json` used to list `kinds: []`, and `git pull` would
 * overwrite the converter's real list while leaving the gitignored `.glb`
 * files on disk — the hall then kept the stone army. We sniff the six meshes
 * themselves and ignore HTML 200s from the SPA fallback.
 */
export function wizardGlbKinds(): Promise<Set<PieceKind>> {
  if (!pending) {
    pending = Promise.all(RANKS.map(async (kind) => ((await probeKind(kind)) ? kind : null))).then(
      (found) => new Set(found.filter((kind): kind is PieceKind => kind !== null)),
    );
  }
  return pending;
}
