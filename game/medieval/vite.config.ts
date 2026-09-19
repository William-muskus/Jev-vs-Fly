import fs from "fs";
import path from "path";

import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin, type ViteDevServer } from "vite";

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
export default defineConfig(({ mode }) => ({
  server: {
    host: "127.0.0.1",
    port: 8080,
    open: "/?autoplay=1",
    hmr: {
      overlay: false,
    },
    proxy: {
      "/api": "http://127.0.0.1:8766",
      // Trailing slash so `/models/wizard/*.glb` stays on Vite (public/), not the fly-brain proxy.
      "/model/": "http://127.0.0.1:8766",
      "/health": "http://127.0.0.1:8766",
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
}));
