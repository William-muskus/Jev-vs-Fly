/**
 * Copy the FlyWire worker (and chess.js) into this Vite app as real files.
 *
 * `public/engine/*.js` used to be git symlinks into `web/engine`. Windows clones
 * without `core.symlinks=true` get a 40-byte path file instead of JavaScript, so
 * `new Worker("/engine/worker.js")` dies as "fly worker crashed" after /health
 * and /model/brain.json have already succeeded.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const medieval = path.resolve(here, "..");
const repo = path.resolve(medieval, "../..");

export function syncFlyEngine(destRoot = path.join(medieval, "public")) {
  const srcEngine = path.join(repo, "web", "engine");
  const destEngine = path.join(destRoot, "engine");
  const srcChess = path.join(repo, "web", "vendor", "chess.js");
  const destChess = path.join(destRoot, "vendor", "chess.js");
  fs.mkdirSync(destEngine, { recursive: true });
  fs.mkdirSync(path.dirname(destChess), { recursive: true });
  for (const name of fs.readdirSync(srcEngine)) {
    if (!name.endsWith(".js")) continue;
    writeCopy(path.join(destEngine, name), path.join(srcEngine, name));
  }
  writeCopy(destChess, srcChess);
}

function writeCopy(dest, src) {
  try {
    fs.lstatSync(dest);
    fs.unlinkSync(dest);
  } catch {
    // missing is fine
  }
  fs.copyFileSync(src, dest);
}

const invoked = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invoked) {
  const dest = process.argv[2]
    ? path.resolve(process.cwd(), process.argv[2])
    : path.join(medieval, "public");
  syncFlyEngine(dest);
  console.log(`synced fly engine → ${dest}`);
}
