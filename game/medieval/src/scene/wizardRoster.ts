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
 * Yaw around world up. Cults STLs already face the enemy in XY; default 0.
 * Override with `?wizardYaw=90` (degrees) without reconverting.
 */
export function wizardYawRadians(search = typeof window !== "undefined" ? window.location.search : ""): number {
  const raw = new URLSearchParams(search).get("wizardYaw");
  if (raw === null || raw === "") return 0;
  const deg = Number(raw);
  if (!Number.isFinite(deg)) return 0;
  return (deg * Math.PI) / 180;
}

export interface AxisSize {
  x: number;
  y: number;
  z: number;
}

/**
 * Print-bed STLs are Z-up, so they lie on the hall's XZ board. Rotate the
 * tallest axis onto +Y. `?wizardUp=none` skips this; `?wizardUp=z` forces the
 * Z-up correction.
 */
export function wizardStandEuler(
  size: AxisSize,
  search = typeof window !== "undefined" ? window.location.search : "",
): { x: number; y: number; z: number } {
  const raw = (new URLSearchParams(search).get("wizardUp") ?? "").toLowerCase();
  if (raw === "none" || raw === "0") return { x: 0, y: 0, z: 0 };
  if (raw === "z") return { x: -Math.PI / 2, y: 0, z: 0 };
  if (raw === "x") return { x: 0, y: 0, z: Math.PI / 2 };
  if (size.y >= size.x && size.y >= size.z) return { x: 0, y: 0, z: 0 };
  if (size.z >= size.x && size.z >= size.y) return { x: -Math.PI / 2, y: 0, z: 0 };
  return { x: 0, y: 0, z: Math.PI / 2 };
}

/**
 * Extra sit after the Z-up tip, in board squares. The tip rotates around the
 * GLB origin, so uncentered prints pick up a Y (up) and Z (along the file) shift
 * until we re-measure. These flags nudge that sit without reconverting.
 * `?wizardLift=0.1` raises; `?wizardPush=-0.2` slides toward White's back rank.
 */
export function wizardNudge(search = typeof window !== "undefined" ? window.location.search : ""): {
  y: number;
  z: number;
} {
  const params = new URLSearchParams(search);
  const y = Number(params.get("wizardLift") ?? "0");
  const z = Number(params.get("wizardPush") ?? "0");
  return {
    y: Number.isFinite(y) ? y : 0,
    z: Number.isFinite(z) ? z : 0,
  };
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
