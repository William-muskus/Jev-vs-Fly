import fs from "fs";
import path from "path";

import type { ServerResponse } from "node:http";

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
  plugins: [react(), wizardGlb404()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  // Expose both VITE_* (Vite default) and EXPO_PUBLIC_* (Rork's cross-platform
  // public-env convention, written by tools like getOrCreateAuthConfig).
  envPrefix: ["VITE_", "EXPO_PUBLIC_"],
});
