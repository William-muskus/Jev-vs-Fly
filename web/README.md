# web/ — play chess against the fly brain, in the browser

A static site (ES modules, no build step, no CDN) that runs the exact network trained in Python:
the FlyWire connectome as a recurrent net, evaluated with plain typed-array CSR kernels inside a
Web Worker. Nothing but the network picks moves.

```
index.html / style.css / app.js   landing, loading screen, game UI, party mode, share card, leaderboard, sounds
board.js                          dependency-free SVG chess board (drag / click, legal dots, promotion picker)
engine/loader.js                  fetch brain.json + brain.flyb(.gz), stream progress, gunzip, Cache API, parse (§8)
engine/flybrain.js                FlyBrain.forward(): SPEC §4 dynamics in pure JS; softmax helpers
engine/mcts.js                    PUCT MCTS (c_puct 1.5) for "superfly"; every evaluation is the brain
engine/worker.js                  Web Worker protocol (load / move / eval), difficulty semantics (§9)
engine/encoding.js                board -> planes, move <-> index (shared with Python)          [other module]
vendor/chess.js                   chess.js 1.4.0 (BSD-2)                                        [other module]
assets/fly.svg                    the fly (own artwork; pieces are drawn in board.js)
model/                            gitignored: brain.json + brain.flyb + brain.flyb.gz from `fly export-web`
test/flybrain.test.mjs            node --test: forward vs. reference, loader layout, MCTS on a stub brain
test/parity.test.mjs              node --test: JS engine == Python reference on the exported brain (SPEC §8)
```

## Run locally

```bash
fly export-web --run <run-name>      # writes web/model/brain.json, brain.flyb, brain.flyb.gz
scripts/serve-web.sh                 # http://localhost:8000  (python -m http.server on web/)
```

A model must be served from `model/` next to `index.html`. Without it the landing page explains
what to do. The first visit downloads the `.gz` blob (streamed, gunzipped in the worker) and caches
the decoded buffer in the Cache API, keyed by `run_name` + `exported_at`; later visits load from cache.

## Worker protocol

```
→ {type:'load', baseUrl}                                  baseUrl absolute (app passes new URL('model/', location))
← {type:'progress', loaded, total, phase, n, nnz, runName}
← {type:'ready', header, legend, sample:{idx, xy, cls}, silhouette:{xy, cls}, fromCache, bytes}
→ {type:'move', id, fen, moves:[uci…], difficulty}       moves = full game history (repetition plane)
← {type:'thinking', id, done, total}                      superfly only, every 10 simulations
← {type:'move', id, move, san, policyTop:[{uci,san,p}], value, activitySample:Float32Array(2048), thinkMs, sims, stepMs}
→ {type:'eval', id, fen, moves}
← {type:'eval', id, value, policyTop, activitySample}
← {type:'error', id?, message}
```

`value` is always from the side to move's perspective at the root (= the fly, when it is asked to move).
`activitySample` holds the final-step activity of 2048 fixed neurons (indices chosen deterministically at
load; their 2-D connectome positions and super-class are sent once in `ready`).

Difficulties (SPEC §9): **larva** samples the legal-masked policy at temperature 1.2; **fly** takes the
top-3 policy moves, plays each, asks the value head how the opponent likes the result and keeps the move
that is worst for them; **superfly** runs 200 PUCT simulations.

## Tests

```bash
node --test web/test/flybrain.test.mjs
for f in web/*.js web/engine/*.js; do node --check "$f"; done
```

The test builds a tiny random brain as a real `.flyb` blob, parses it with the loader and checks
`FlyBrain.forward` against a float64 reference for relu/tanh/gelu, linear and MLP value heads,
then runs MCTS with a constant-policy stub brain on chess.js (mate-in-one must be found).
Cross-language parity with Python: `node --test web/test/parity.test.mjs` loads `web/model/` with the loader and checks every logit / value against `tests/vectors/model.json` (written by `fly export-web` from the same blob); it skips when either file is missing.

## Performance

Measured with Node 22 on a desktop CPU for a synthetic full-size brain (134k neurons, 2.7M synapses,
random column pattern — a worst case for cache locality): ≈3.5 ms per recurrent timestep, ≈37 ms per
forward pass at 8 steps including the 4168×2048 policy head. Superfly (200 simulations) therefore
takes ≈7–8 s per move; the UI shows the simulation count ticking.

## Deploy

`scripts/deploy-pages.sh` copies `web/` (with `web/model/`) into a temporary worktree on the
`gh-pages` branch, commits and pushes. Every file must stay below GitHub's 100 MB limit.

## Credits

FlyWire connectome — Dorkenwald et al. 2024 and Schlegel et al. 2024 (*Nature*), data CC BY-NC 4.0.
chess.js — BSD-2. Piece and fly artwork were drawn for this project.
