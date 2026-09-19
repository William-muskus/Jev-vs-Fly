/**
 * How the hall loads the FlyWire worker.
 *
 * The 3D scene already holds a WebGL context. Asking the connectome worker to
 * also spin up WebGPU on Windows often kills that worker with an empty
 * `onerror` — the banner then says "fly worker crashed". JS mode is slower but
 * shares the GPU with the hall. Pass `?flyGpu=1` to opt back in.
 *
 * A second Windows failure mode: `public/engine/*.js` used to be git symlinks.
 * Without Developer Mode those files are a relative path, not JavaScript, so
 * the worker dies after /health and /model/brain.json have already succeeded.
 */

const API_HINT = "From the repo root, venv on: python -m game.server";
const ENGINE_HINT =
  "In game/medieval run: npm run sync-engine   then restart npm run dev and hard-reload (Ctrl+Shift+R).";

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

/** True when the response body is the worker module, not a broken symlink path. */
export function isFlyWorkerScript(source: string): boolean {
  const head = source.trimStart().slice(0, 160);
  if (/^(?:import\s|export\s|\/\/|\/\*)/.test(head)) return true;
  return false;
}

export function explainFlyLoadFailure(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err);
  if (/python -m game\.server|npm run sync-engine/i.test(raw)) return raw;
  if (
    /\b(502|503|504)\b/.test(raw) ||
    /econnrefused|failed to fetch|networkerror|not running|did not answer \/health|brain\.json/i.test(raw)
  ) {
    return `${raw}. ${API_HINT}`;
  }
  if (/unexpected token|not javascript|broken windows git symlink/i.test(raw)) {
    return `${raw}. ${ENGINE_HINT}`;
  }
  if (isFlyWorkerCrash(err)) {
    return `${raw}. If /health is already 200, this is the worker script, not the API. ${ENGINE_HINT}`;
  }
  return raw;
}

/** Fail fast when the Python API is down or the worker file is not JavaScript. */
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
  await preflightFlyWorker();
}

async function preflightFlyWorker(): Promise<void> {
  let res: Response;
  try {
    res = await fetch("/engine/worker.js", { cache: "no-cache" });
  } catch {
    throw new Error(`could not fetch /engine/worker.js. ${ENGINE_HINT}`);
  }
  const text = await res.text();
  if (!res.ok) {
    throw new Error(`could not fetch /engine/worker.js (${res.status}). ${ENGINE_HINT}`);
  }
  if (!isFlyWorkerScript(text)) {
    throw new Error(
      "the fly worker is not JavaScript (broken Windows git symlink). " + ENGINE_HINT,
    );
  }
}
