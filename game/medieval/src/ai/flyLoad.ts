/**
 * How the hall loads the FlyWire worker.
 *
 * The 3D scene already holds a WebGL context. Asking the connectome worker to
 * also spin up WebGPU on Windows often kills that worker with an empty
 * `onerror` — the banner then says "fly worker crashed". JS mode is slower but
 * shares the GPU with the hall. Pass `?flyGpu=1` to opt back in.
 */

const API_HINT = "From the repo root, venv on: python -m game.server";

export function flyGpuEnabled(search = typeof window !== "undefined" ? window.location.search : ""): boolean {
  const raw = new URLSearchParams(search).get("flyGpu");
  return raw === "1" || raw === "true";
}

export function flyWorkerCrashMessage(ev: { message?: string; filename?: string; lineno?: number }): string {
  const where = ev.filename ? ` in ${ev.filename}${ev.lineno ? `:${ev.lineno}` : ""}` : "";
  const detail = `${ev.message ?? ""}${where}`.trim();
  return detail || "fly worker crashed";
}

export function isFlyWorkerCrash(err: unknown): boolean {
  const raw = err instanceof Error ? err.message : String(err);
  return /fly worker crashed|worker crashed/i.test(raw);
}

export function explainFlyLoadFailure(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err);
  if (/python -m game\.server/i.test(raw)) return raw;
  if (
    /\b(502|503|504)\b/.test(raw) ||
    /econnrefused|failed to fetch|networkerror|not running|did not answer \/health|brain\.json/i.test(raw)
  ) {
    return `${raw}. ${API_HINT}`;
  }
  if (isFlyWorkerCrash(err)) {
    return `${raw}. ${API_HINT}, then click Jev vs Fly again (or hard-reload).`;
  }
  return raw;
}

/** Fail fast with a readable message when the Python API is down. */
export async function preflightFlyApi(baseUrl = "/model/"): Promise<void> {
  const root = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  let health: Response;
  try {
    health = await fetch("/health", { cache: "no-cache" });
  } catch {
    throw new Error(`the Jev vs Fly API is not running on port 8766. ${API_HINT}`);
  }
  if (!health.ok) {
    throw new Error(`the Jev vs Fly API did not answer /health (${health.status}). ${API_HINT}`);
  }
  let header: Response;
  try {
    header = await fetch(`${root}brain.json`, { cache: "no-cache" });
  } catch {
    throw new Error(`could not reach ${root}brain.json. ${API_HINT}`);
  }
  if (!header.ok) {
    throw new Error(`could not fetch ${root}brain.json (${header.status}). ${API_HINT}`);
  }
}
