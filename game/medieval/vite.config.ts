import fs from "fs";
import path from "path";

import type { IncomingMessage, ServerResponse } from "node:http";

import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin, type ProxyOptions, type ViteDevServer } from "vite";

const PYTHON_API = "http://127.0.0.1:8766";
const API_DOWN = JSON.stringify({
  detail: "Jev vs Fly API is not running on port 8766. From the repo root, venv on: python -m game.server",
});

/** Same-origin proxy to the FastAPI app, with a JSON 502 when that process is down. */
function pythonApiProxy(): ProxyOptions {
  return {
    target: PYTHON_API,
    configure(proxy) {
      proxy.on("error", (_err, _req, res) => {
        if (!res || !("writeHead" in res)) return;
        const httpRes = res as ServerResponse;
        if (httpRes.headersSent) return;
        httpRes.writeHead(502, { "Content-Type": "application/json" });
        httpRes.end(API_DOWN);
      });
    },
  };
}

/**
 * Serve /engine/* and /vendor/chess.js from ../../web as real files.
 * Git-symlink copies under public/ become a path-file on many Windows clones,
 * so the fly worker dies as "fly worker crashed" before it can fetch the blob.
 */
function flyEngineFromRepo(): Plugin {
  const engineDir = path.resolve(__dirname, "../../web/engine");
  const chess = path.resolve(__dirname, "../../web/vendor/chess.js");

  const fileFor = (url: string): string | null => {
    const clean = url.split("?")[0];
    if (clean === "/vendor/chess.js") return chess;
    if (!clean.startsWith("/engine/")) return null;
    const name = path.basename(clean);
    if (!name.endsWith(".js") || name.includes("..")) return null;
    const file = path.join(engineDir, name);
    return fs.existsSync(file) ? file : null;
  };

  const send = (res: ServerResponse, file: string): void => {
    res.statusCode = 200;
    res.setHeader("Content-Type", "text/javascript; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache");
    fs.createReadStream(file).pipe(res);
  };

  const middleware = (req: IncomingMessage, res: ServerResponse, next: () => void): void => {
    const file = fileFor(req.url ?? "");
    if (!file) {
      next();
      return;
    }
    send(res, file);
  };

  const attach = (server: ViteDevServer): void => {
    server.middlewares.use(middleware);
  };

  return {
    name: "fly-engine-from-repo",
    configureServer: attach,
    configurePreviewServer: attach,
    closeBundle() {
      const dist = path.resolve(__dirname, "dist");
      const destEngine = path.join(dist, "engine");
      const destVendor = path.join(dist, "vendor");
      fs.mkdirSync(destEngine, { recursive: true });
      fs.mkdirSync(destVendor, { recursive: true });
      for (const name of fs.readdirSync(engineDir)) {
        if (!name.endsWith(".js")) continue;
        fs.copyFileSync(path.join(engineDir, name), path.join(destEngine, name));
      }
      fs.copyFileSync(chess, path.join(destVendor, "chess.js"));
    },
  };
}

/** Missing wizard meshes must 404, not fall through to index.html (status 200). */
function wizardGlb404(): Plugin {
  return {
    name: "wizard-glb-404",
    configureServer(server: ViteDevServer) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? "").split("?")[0];
        if (!url.startsWith("/models/wizard/") || !url.endsWith(".glb")) {
          next();
          return;
        }
        const file = path.join(server.config.root, "public", url.replace(/^\//, ""));
        if (!fs.existsSync(file)) {
          res.statusCode = 404;
          res.setHeader("Content-Type", "text/plain");
          res.end("wizard glb not found");
          return;
        }
        next();
      });
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    host: "127.0.0.1",
    port: 8080,
    open: "/?autoplay=1",
    hmr: {
      overlay: false,
    },
    fs: {
      allow: [path.resolve(__dirname, "../..")],
    },
    proxy: {
      "/api": pythonApiProxy(),
      // Trailing slash so `/models/wizard/*.glb` stays on Vite (public/), not the fly-brain proxy.
      "/model/": pythonApiProxy(),
      "/health": pythonApiProxy(),
    },
  },
  plugins: [react(), flyEngineFromRepo(), wizardGlb404()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  // Expose both VITE_* (Vite default) and EXPO_PUBLIC_* (Rork's cross-platform
  // public-env convention, written by tools like getOrCreateAuthConfig).
  envPrefix: ["VITE_", "EXPO_PUBLIC_"],
});
