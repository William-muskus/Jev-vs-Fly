# web/ — play chess against the fly brain, in the browser

A static site (ES modules, no build step, no CDN) that runs the exact network trained in Python:
the FlyWire connectome as a recurrent net, evaluated inside a Web Worker — on the GPU through WebGPU
compute shaders when the browser has it, otherwise with plain typed-array CSR kernels (same math,
automatic fallback). Nothing but the network picks moves.

```
index.html / style.css / app.js   landing, loading screen, game UI, party mode, share card, leaderboard, sounds
board.js                          dependency-free SVG chess board (drag / click, legal dots, promotion picker)
engine/loader.js                  fetch brain.json + brain.flyb(.gz), stream progress, gunzip, Cache API, parse (§8)
engine/flybrain.js                FlyBrain.forward(): SPEC §4 dynamics in pure JS; softmax helpers
engine/flybrain-gpu.js            FlyBrainGPU: the same forward pass as WGSL compute shaders (async forward), plus
                                  f32 JS mirrors of every kernel (emulate*) so the shader math is testable under node
engine/mcts.js                    PUCT MCTS (c_puct 1.5) for "superfly"; every evaluation is the brain (sync or async)
engine/worker.js                  Web Worker protocol (load / move / eval), difficulty semantics (§9), backend choice + fallback
engine/encoding.js                board -> planes, move <-> index (shared with Python)          [other module]
vendor/chess.js                   chess.js 1.4.0 (BSD-2)                                        [other module]
assets/fly.svg                    the fly (own artwork; pieces are drawn in board.js)
model/                            gitignored: brain.json + brain.flyb + brain.flyb.gz from `fly export-web`
test/flybrain.test.mjs            node --test: forward vs. reference, loader layout, MCTS on a stub brain
test/flybrain-gpu.test.mjs        node --test: WGSL kernel math (f32 emulation) vs. float64 reference; async MCTS
test/parity.test.mjs              node --test: JS engine == Python reference on the exported brain (SPEC §8)
test/browser/gpu-parity.mjs       node script (not --test): real model in headless Chromium, WebGPU vs JS parity,
                                  latency, worker backends and device-loss fallback
```

## Run locally

```bash
fly export-web --run <run-name>      # writes web/model/brain.json, brain.flyb, brain.flyb.gz
scripts/serve-web.sh                 # http://localhost:8000  (python -m http.server on web/)
```

A model must be served from `model/` next to `index.html`. Without it the landing page explains
what to do. The first visit downloads the `.gz` blob (streamed, gunzipped in the worker) and caches
the decoded buffer in the Cache API, keyed by `run_name` + `exported_at` + `blob_sha256`; later visits load from cache.
Blob requests carry the same version tag as a query string (`brain.flyb.gz?v=…`) and are fetched with
`cache: 'no-cache'`, and every blob (downloaded or cached) is checked against the header's `total_bytes` and
`blob_sha256`, so a re-deploy never pairs the new `brain.json` with a stale `brain.flyb` from an HTTP cache.

## Worker protocol

```
→ {type:'load', baseUrl, gpu?}                            baseUrl absolute (app passes new URL('model/', location)); gpu:false = JS only
← {type:'progress', loaded, total, phase, n, nnz, runName}
← {type:'ready', header, legend, sample:{idx, xy, cls}, silhouette:{xy, cls}, fromCache, bytes, backend, gpu}
                                                          backend: 'webgpu' | 'js'; gpu: {timestamps} or {error: why not}
→ {type:'move', id, fen, moves:[uci…], difficulty}       moves = full game history (repetition plane)
← {type:'thinking', id, done, total}                      superfly only, every 10 simulations
← {type:'move', id, move, san, policyTop:[{uci,san,p}], value, activitySample:Float32Array(2048), thinkMs, sims, stepMs, backend}
→ {type:'eval', id, fen, moves}
← {type:'eval', id, value, policyTop, activitySample, backend}
→ {type:'bench', id, reps?, backend?, activity?}          per-forward latency of one engine
← {type:'bench', id, forwardMs, stepMs, backend, reps}
← {type:'backend', backend:'js', reason}                  unsolicited: the WebGPU device was lost, JS answers from now on
← {type:'error', id?, message}
```

**Backends.** On `load` the worker always builds the plain-JS `FlyBrain` and then tries `FlyBrainGPU.create`
(skipped with `gpu:false`, bounded to 20 s): WebGPU adapter + device, the CSR / heads uploaded once
(~55 MB of f32 storage buffers), one compute pipeline per kernel. Every forward then goes through one
wrapper: WebGPU while it is alive, JS otherwise. A GPU failure (device lost, out of memory, shader
error) is reported once as `{type:'backend'}` and the JS engine answers that request and all later ones;
`backend` in each reply says which engine produced it, and the page shows it in the specimen line
("engine WebGPU" / "engine JS"). Where WebGPU is missing (Firefox without the flag, `--disable-gpu`,
insecure origins) `ready` carries `gpu: {error}` and everything runs in JS as before.

`FlyBrainGPU.forward(x, {activity})` is asynchronous (one command buffer: injection, all timesteps
with ping-pong hidden-state buffers, both heads; one `mapAsync` readback of 4168 logits + the value
hidden layer + optionally the full final state, 537 KB). `mcts.js` accepts either kind of brain:
`step()` returns a boolean for `FlyBrain` and a promise for `FlyBrainGPU`; `runMCTSAsync` awaits
only when it is handed a promise, so the JS path is unchanged. The recurrent SpMV runs rows bucketed
by synapse count (1 / 8 / 64 lanes per row, `LANE_BUCKETS`) so the ~150 rows with thousands of
inputs do not serialise a whole timestep.

`value` is always from the side to move's perspective at the root (= the fly, when it is asked to move).
`activitySample` holds the final-step activity of 2048 fixed neurons (indices chosen deterministically at
load; their 2-D connectome positions and super-class are sent once in `ready`).

Difficulties (SPEC §9): **larva** samples the legal-masked policy at temperature 1.2; **fly** takes the
top-3 policy moves, plays each, asks the value head how the opponent likes the result and keeps the move
that is worst for them; **superfly** runs 200 PUCT simulations.

## Tests

```bash
node --test web/test/flybrain.test.mjs web/test/flybrain-gpu.test.mjs
for f in web/*.js web/engine/*.js; do node --check "$f"; done
node web/test/browser/gpu-parity.mjs        # needs web/model/ and a Chromium with WebGPU (CHROME=… to pick one)
```

The test builds a tiny random brain as a real `.flyb` blob, parses it with the loader and checks
`FlyBrain.forward` against a float64 reference for relu/tanh/gelu, linear and MLP value heads,
then runs MCTS with a constant-policy stub brain on chess.js (mate-in-one must be found).
`flybrain-gpu.test.mjs` checks the WGSL kernels through their f32 JS mirrors (`emulateInject` /
`emulateStep` / `emulateMatvec` / `emulateForward`, `Math.fround` after every operation, same
summation order as the shader lanes) against the float64 reference and `FlyBrain` for
relu / tanh / gelu / gelu-tanh / satrelu and both value heads, and that MCTS gives the same search
with a promise-returning brain. `test/browser/gpu-parity.mjs` serves `web/` with the real model,
starts headless Chromium (`--headless=new --enable-unsafe-webgpu --ignore-gpu-blocklist
--enable-features=Vulkan --use-angle=vulkan`; `GPU_ANGLE=swiftshader` for a software run) and
compares `FlyBrainGPU` with `FlyBrain` on 5 positions (|Δlogit| ≤ 1e-2 on all 4168 logits,
|Δvalue| ≤ 1e-2, same legal argmax), then drives `worker.js`: backend report, moves at every
difficulty, bench, and a forced device loss followed by moves from the JS fallback.
Cross-language parity with Python: `node --test web/test/parity.test.mjs` loads `web/model/` with the loader and checks every logit / value against `tests/vectors/model.json` (written by `fly export-web` from the same blob); it skips when either file is missing.

## Performance

Measured with Node 22 on a desktop CPU for a synthetic full-size brain (134k neurons, 2.7M synapses,
random column pattern — a worst case for cache locality): ≈3.5 ms per recurrent timestep, ≈37 ms per
forward pass at 8 steps including the 4168×2048 policy head. Superfly (200 simulations) therefore
takes ≈7–8 s per move; the UI shows the simulation count ticking.

WebGPU, exported `fly2` (134,209 neurons, 2,700,513 connections, 16 satrelu steps), headless Chromium 149
on an RTX 5080 via Vulkan: 0.09–0.16 ms per recurrent step (timestamp queries), 2.5–3.3 ms per forward
including submit + readback (≈2.5 ms without the 537 KB activity readback), against 79–88 ms in plain JS
in the same browser (4.5 ms per step) — ≈24×. Superfly (100 simulations) answers in ≈0.4 s instead of ≈8 s.
GPU vs JS on 5 positions: |Δlogit| ≤ 1.9e-4, |Δvalue| ≤ 1.1e-6, |Δactivity| ≤ 3e-6, same argmax.

## Deploy

`scripts/deploy-pages.sh` copies `web/` (with `web/model/`) into a temporary worktree on the
`gh-pages` branch, commits and pushes. Every file must stay below GitHub's 100 MB limit.
Each deploy is a single snapshot commit that replaces the branch (force push) so the ~75 MB of
model blobs are not accumulated in history; `KEEP_HISTORY=1` appends instead. Re-running with an
unchanged `web/` is a no-op (`BUILD.txt` is keyed on the source commit and `brain.json`'s
`exported_at`, not on the wall clock).

## Credits

FlyWire connectome — Dorkenwald et al. 2024 and Schlegel et al. 2024 (*Nature*), data CC BY-NC 4.0.
chess.js — BSD-2. Piece and fly artwork were drawn for this project.
