# fly-chess — Architecture & Interface Specification

This document is the contract between all modules. Every implementer must follow it exactly;
if something here is impossible, change the SPEC *and* every dependent module, never silently diverge.

## 0. Principles

1. **The fly brain really plays.** The player is a recurrent neural network whose neurons and
   synapses are the real FlyWire adult *Drosophila melanogaster* connectome. Nothing else picks moves
   (no hidden engine, no random-move fallback presented as "the fly"). The browser runs the exact
   same network as Python (bit-for-bit up to float rounding), verified by cross-language test vectors.
2. **Connectome-faithful parameterisation.** Synapse *existence* comes from the connectome and is
   never learned. Synapse *sign* comes from the presynaptic neurotransmitter (Dale's law) and is
   never learned. Only synaptic *strength magnitudes* (initialised from synapse counts), per-neuron
   biases / time constants, and the sensory-input and motor-readout projections are learned.
3. **One shared encoding.** Board → planes and move ↔ index are defined once here and implemented
   identically in `flychess/chessenv/encoding.py` and `web/engine/encoding.js`.
4. **Everything runs from the CLI**: `fly download`, `fly build-brain`, `fly train`, `fly play`,
   `fly dashboard`, `fly export-web`, `fly eval`, and from Python: `flychess.train(...)`, `flychess.play(...)`.

## 1. Repository layout

```
fly-chess/
  pyproject.toml            # uv/hatch project, console script `fly`
  README.md                 # user documentation (quickstart, science, credits)
  docs/SPEC.md              # this file
  flychess/
    __init__.py             # exports train(), play(), load_brain(), __version__
    cli.py                  # argparse CLI, subcommands listed in §9
    paths.py                # DATA_DIR / RUNS_DIR / CACHE_DIR resolution (env FLYCHESS_HOME)
    connectome/
      download.py           # download FlyWire tables (Zenodo/Codex mirrors), verify sizes
      load.py               # parse tables -> Connectome dataclass (pandas -> numpy)
      graph.py              # subgraph selection, neuron role assignment, CSR build -> BrainGraph
    chessenv/
      encoding.py           # board->planes, move<->index (§3)
      env.py                # ChessEnv (reset/step/legal_mask/result), perspective handling
    model/
      spmm.py               # SpMM autograd.Function (CSR fwd, transposed-CSR bwd, chunked dW)
      flybrain.py           # FlyBrain nn.Module (§4)
      config.py             # BrainConfig dataclass, yaml (de)serialisation
    data/
      lichess.py            # download monthly PGN .zst, stream-parse, filter, write shards
      shards.py             # npz/parquet shard format (§5), PositionDataset, collate
    train/
      metrics.py            # MetricsLogger: JSONL appends to runs/<run>/metrics.jsonl (§7)
      imitation.py          # stage 1 supervised trainer
      mcts.py               # batched MCTS using FlyBrain (policy+value)
      selfplay.py           # stage 2 self-play generation + training loop
      trainer.py            # train(config) orchestrating both stages, checkpoints (§6)
    eval/
      elo.py                # round-robin / vs-random / vs-previous-checkpoint Elo estimation
      opponents.py          # RandomPlayer, GreedyMaterialPlayer, optional Stockfish (if installed)
    play/
      engine.py             # FlyEngine: choose_move(board, difficulty) using FlyBrain (+MCTS)
      terminal.py           # rich terminal UI to play vs the fly
      local_web.py          # `fly play --gui`: serves web/ locally with the current brain
    dashboard/
      server.py             # FastAPI + websocket, tails metrics.jsonl, serves static/
      static/index.html, dashboard.js, dashboard.css
    export/
      web.py                # export trained FlyBrain -> web/model/brain.flyb (+ brain.json) (§8)
      testvectors.py        # writes tests/vectors/*.json for JS parity tests
  web/                      # static website (GitHub Pages / any static host)
    index.html, style.css, app.js
    engine/encoding.js, engine/flybrain.js, engine/mcts.js, engine/worker.js, engine/loader.js
    vendor/chess.js         # vendored (MIT)
    assets/                 # piece SVGs, fly sprites, sounds
    model/                  # gitignored; produced by `fly export-web`
    test/parity.test.mjs    # node test: JS engine == Python vectors
  tests/                    # pytest
  scripts/                  # helper shell scripts (deploy pages, full pipeline)
  data/  runs/              # gitignored: downloaded data, training runs
```

## 2. Connectome (`flychess/connectome`)

### 2.1 Source files (FlyWire public release, Codex snapshot 783, CC-BY 4.0)
Downloaded into `data/connectome/`:
- `connections.csv.gz` — columns `pre_root_id, post_root_id, neuropil, syn_count, nt_type`.
  One row per (pre, post, neuropil); only pairs with >= 5 synapses. Aggregate over neuropil → (pre, post, syn_count, nt_type).
- `classification.csv.gz` — `root_id, flow, super_class, class, sub_class, cell_type, hemibrain_type, hemilineage, side, nerve`.
- `neurons.csv.gz` — `root_id, group, nt_type, nt_type_score, da_avg, ser_avg, gaba_avg, glut_avg, ach_avg, oct_avg, ...`.
- `cell_stats.csv.gz` or `coordinates.csv.gz` — soma / representative position for visualisation.
`download.py` tries the URLs in `SOURCES` in order, verifies size, and prints manual instructions
if all fail (the user can also drop the files in `data/connectome/`). `fly download --connectome`.

### 2.2 `Connectome` (load.py)
```python
@dataclass
class Connectome:
    root_ids: np.ndarray[int64]      # (N,) sorted unique neuron ids
    super_class: np.ndarray[str]     # (N,)
    cell_class: np.ndarray[str]      # (N,)
    cell_type: np.ndarray[str]       # (N,)
    side: np.ndarray[str]            # (N,) 'left'/'right'/'center'/''
    nt_type: np.ndarray[str]         # (N,) presynaptic neurotransmitter: ACH/GABA/GLUT/DA/SER/OCT/''
    position: np.ndarray[float32]    # (N,3) nm, NaN if unknown
    pre: np.ndarray[int32]           # (E,) index into root_ids
    post: np.ndarray[int32]          # (E,)
    syn_count: np.ndarray[int32]     # (E,) summed over neuropils
    neuropil: np.ndarray[str] | None # optional dominant neuropil per edge
```
`load_connectome(data_dir) -> Connectome` caches a `connectome.npz` after first parse.

### 2.3 `BrainGraph` (graph.py) — what the model consumes
```python
@dataclass
class BrainGraph:
    n: int                            # neurons in the trained subgraph
    root_ids: np.ndarray[int64]       # (n,)
    csr_indptr: np.ndarray[int32]     # (n+1,)  rows = POST-synaptic neuron
    csr_indices: np.ndarray[int32]    # (nnz,)  PRE-synaptic neuron per synapse
    syn_count: np.ndarray[float32]    # (nnz,)
    sign: np.ndarray[int8]            # (nnz,) +1 / -1  from presynaptic nt_type
    input_idx: np.ndarray[int32]      # (n_in,)  sensory neurons that receive the board
    output_idx: np.ndarray[int32]     # (n_out,) descending/motor neurons read out by the heads
    super_class: np.ndarray[str]      # (n,)
    position: np.ndarray[float32]     # (n,3)
    meta: dict                        # selection parameters, counts per super_class, source hash
```
Sign rule (Dale's law, fly): `ACH → +1`, `GABA → -1`, `GLUT → -1` (glutamate is predominantly
inhibitory in the fly CNS via GluCl), `DA/SER/OCT/unknown → +1` (modulatory, treated excitatory).
Neurons with no annotated nt keep +1.

Selection (`build_brain_graph(conn, cfg: GraphConfig)`), `GraphConfig` fields:
- `region: 'full' | 'central'` — `central` drops `super_class in {optic, visual_projection, visual_centrifugal}`.
- `max_neurons: int | None` — if set, keep the top-k neurons by total synapse count (in+out),
  but *always* keep the input and output sets.
- `min_syn: int = 5`
- `input_super_classes: ['sensory', 'ascending']`, `max_inputs: 2048` — choose deterministically:
  sort candidates by (out-degree desc, root_id) and take the first `max_inputs`.
- `output_super_classes: ['descending', 'motor']`, `max_outputs: 2048` — same rule with in-degree.
- Remove self-loops; remove neurons with zero degree after filtering (except input/output sets).
- Rows must be sorted by (post, pre) — CSR canonical form. `graph.save(path)` / `BrainGraph.load(path)` as npz.
Default for the real run: `region='full', max_neurons=None` (whole brain, ~139k neurons, ~2.7M synapses).
A `tiny` config (`max_neurons=2000`) exists for tests/smoke runs. Tests also use a random `toy_graph(n, nnz)`
generator, which must NEVER be used outside `tests/`.

## 3. Chess encoding (`chessenv/encoding.py` ⟷ `web/engine/encoding.js`)

### 3.1 Perspective
All encodings are from the **side to move's perspective**: if black is to move, mirror the board
vertically (`chess.Board.mirror()` semantics: swap colours *and* flip ranks) so that the mover is
always "white" moving up the board. Move indices are likewise in mirrored coordinates when black moves.
Square numbering: `chess.Square` (a1=0 … h8=63, index = rank*8+file). Mirroring a square = `sq ^ 56`.

### 3.2 Board planes — `encode_board(board) -> float32[NUM_PLANES, 8, 8]`, `NUM_PLANES = 20`
Plane order (all values 0/1 unless noted), indexing `[plane, rank, file]`:
- 0–5: mover's P, N, B, R, Q, K
- 6–11: opponent's P, N, B, R, Q, K
- 12: mover can castle kingside (all ones) · 13: mover queenside · 14: opponent kingside · 15: opponent queenside
- 16: en-passant target square (single 1)
- 17: constant ones (bias plane)
- 18: halfmove clock / 100, clipped to 1 (all squares)
- 19: repetition: 1 if the current position already occurred before in the game (all squares), else 0
`FLAT_INPUT = NUM_PLANES * 64 = 1280`; flatten in C order (plane-major).

### 3.3 Move index — `move_to_index(move, board) -> int`, `index_to_move(idx, board) -> chess.Move`, `NUM_MOVES = 4168`
Work in perspective coordinates (mirror from/to squares when black to move):
- Non-promotion, or promotion to queen: `idx = from * 64 + to` (0…4095).
- Under-promotion (N, B, R): `idx = 4096 + (from_file * 3 + dir) * 3 + piece` where
  `dir ∈ {0: capture-left (to_file = from_file-1), 1: push (same file), 2: capture-right}`,
  `piece ∈ {0: knight, 1: bishop, 2: rook}`. from_file 0..7 → 72 indices (4096…4167).
`legal_move_mask(board) -> bool[NUM_MOVES]`. Castling is encoded as the king's from/to squares
(e1g1 etc., python-chess standard UCI, not king-takes-rook).

### 3.4 Value target
`+1` mover wins, `0` draw, `-1` mover loses. Model value head outputs `tanh` in the same convention.

### 3.5 Test vectors
`tests/vectors/encoding.json`: list of `{fen, planes_sha256, legal_indices:[...], moves:{uci: idx}}`.
Generated by Python, checked by both pytest and `web/test/parity.test.mjs`.

## 4. Model (`model/flybrain.py`)

```python
@dataclass
class BrainConfig:
    graph_path: str
    steps: int = 8            # recurrent timesteps per position
    alpha: float = 0.5        # initial leak (learned per neuron via sigmoid(logit))
    activation: str = 'relu'  # 'relu' | 'gelu' | 'tanh'
    input_dim: int = 1280
    num_moves: int = 4168
    weight_init_scale: float = 1.0   # multiplies log1p(syn_count) init
    dale: bool = True         # enforce signs
    dtype: str = 'float32'
```
Parameters:
- `syn_gain: (nnz,)` — `w = sign * softplus(syn_gain)`; init `syn_gain = inverse_softplus(scale * log1p(syn_count) / normaliser)`
  where `normaliser` makes the expected row sum of |w| ≈ 1 (spectral-radius-ish control; document formula in code).
- `bias: (n,)` init 0 · `leak_logit: (n,)` init logit(alpha).
- `w_in: (n_in, input_dim)` + `b_in: (n_in,)` — board planes injected into `input_idx` neurons each step.
- `policy_head: Linear(n_out, num_moves)`, `value_head: Linear(n_out, 1)` (or small MLP `n_out→256→1`), applied to
  the final-step activity of `output_idx` neurons.
Dynamics (batch-first `h: (B, n)` is fine; SpMM internally transposes to `(n, B)` if faster):
```
h_0 = 0
for t in range(steps):
    inp = zeros(B, n); inp[:, input_idx] = x @ w_in.T + b_in
    pre = SpMM(w, h_t) + bias + inp             # SpMM: out[b, post] = Σ_pre w[post,pre] * h[b, pre]
    h_{t+1} = (1 - a) * h_t + a * act(pre)      # a = sigmoid(leak_logit)
policy_logits = policy_head(h_T[:, output_idx]);  value = tanh(value_head(h_T[:, output_idx]))
```
`forward(x: (B,1280)) -> (policy_logits (B,4168), value (B,1), h_T (B,n) optional)`.
Provide `forward(x, return_activity=True)` for visualisation. Policy loss masks illegal moves with `-inf` before
cross-entropy. The module must be exportable: `state_dict()` + `BrainConfig` + graph path fully define the network.
`model/spmm.py` implements the custom autograd function benchmarked at 0.27 s per (B=256, T=8, full brain) step;
it precomputes the transposed CSR permutation once and computes `dW` in chunks of ≤ 1e6 synapses.

## 5. Data (`data/`)

- `fly download --games --months 2014-01,2014-02` downloads `lichess_db_standard_rated_<YYYY-MM>.pgn.zst` into `data/pgn/`.
- `data/lichess.py::build_shards(pgn_paths, out_dir, min_elo=1800, max_positions=None, skip_openings=4, workers=N)`
  streams games with `zstandard` + `chess.pgn`, keeps rated standard games where both players ≥ `min_elo`
  and the game has a decisive or drawn result, skips the first `skip_openings` plies, and writes shards of
  `SHARD_SIZE=262144` positions as `data/shards/<name>-NNNNN.npz` with arrays:
  `planes: uint8 (S, 20, 8, 8)` (planes 18/19 stored as uint8 0-255 / 0-1 and rescaled on load),
  `move: int16 (S,)`, `value: int8 (S,)`, `elo: int16 (S,)`, `ply: int16 (S,)`.
  Positions are shuffled globally across the month using a buffer before sharding.
- `data/shards.py::ShardDataset` (torch IterableDataset, shard-level shuffle, worker-sharded) and `collate`.
- Also provide `positions_from_pgn(path)` generator for tests.

## 6. Training (`train/`)

`fly train --run <name> [--stage imitation|selfplay|all] [--config cfg.yaml] [--resume]`.
`TrainConfig` (yaml): graph selection, BrainConfig, `batch_size=256`, `lr=1e-3` (AdamW, cosine schedule with
warmup), `epochs`, `value_loss_weight=1.0`, `grad_clip=1.0`, `amp=True` (bf16 autocast for the dense parts only;
SpMM stays fp32), `eval_every`, `checkpoint_every`, self-play settings (`games_per_iter`, `mcts_sims=64`,
`temperature`, `dirichlet_alpha`, `replay_buffer_size`), `seed`.

Checkpoints: `runs/<run>/ckpt-<step>.pt` + `runs/<run>/latest.pt` containing
`{'config': TrainConfig(asdict), 'brain_config': ..., 'graph_path': ..., 'model': state_dict, 'optim': ..., 'step': int, 'stage': str}`.
`flychess.load_brain(path_or_run) -> (FlyBrain, BrainGraph)`.

Stage 1 (imitation): cross-entropy on the human move (label smoothing 0.0) + MSE on value. Log every 20 steps.
Stage 2 (self-play): generate games with batched MCTS (`mcts.py`, all leaf evaluations batched through the GPU,
virtual loss, Dirichlet noise at root, temperature 1 for first 30 plies then 0), store (planes, π, z) in a replay
buffer, train on samples. Also periodically play `eval_games` against the previous checkpoint & RandomPlayer and log Elo.

Python API: `flychess.train(run='fly1', stage='all', **overrides)`.

## 7. Metrics (`train/metrics.py` ⟷ dashboard)

`runs/<run>/metrics.jsonl`: one JSON object per line, always with `"t": unix_time, "step": int, "kind": str`. Kinds:
- `train`: `{loss, policy_loss, value_loss, top1, top3, lr, pos_per_sec, gpu_mem_gb, epoch, stage}`
- `eval`: `{val_loss, val_top1, val_top3, val_value_mse}`
- `elo`: `{opponent, games, wins, draws, losses, elo_estimate}`
- `game`: `{pgn, result, moves: int, source: 'selfplay'|'eval'}` (a sample game, ≤ 1 per minute)
- `activity`: `{neuron_idx: [...], values: [...]}` sampled 2048 neurons' activity for a heatmap (≤ 1 per 30 s)
- `status`: `{message, stage, total_steps, eta_s}`
`runs/<run>/run.json` holds the static description (config, graph meta, start time).

## 8. Web model format (`export/web.py` ⟷ `web/engine/loader.js`)

`web/model/brain.json` (header) + `web/model/brain.flyb` (little-endian binary blob):
header lists `{name: string, dtype: 'f16'|'f32'|'i8', shape: [...], offset: int, length_bytes: int, scale?: float}`
for each array, in this order: `csr_indptr (i32)`, `csr_indices (i32)`, `w (f16, already signed = sign*softplus(gain))`,
`bias (f32)`, `alpha (f32, = sigmoid(leak_logit))`, `input_idx (i32)`, `output_idx (i32)`, `w_in (f16, [n_in,1280])`,
`b_in (f32)`, `policy_w (f16, [num_moves, n_out])`, `policy_b (f32)`, `value_w (f16, [1 or hidden, n_out])`, `value_b`,
(if MLP value head, also `value_w2`, `value_b2`), plus `positions (f16, [n,3])` normalised to [0,1] for the brain visualiser,
and `super_class (u8, [n])` with a legend in the header. Header also has `steps, activation, n, nnz, n_in, n_out, num_moves,
num_planes, run_name, train_steps, exported_at, elo_estimates`.
Offsets are 8-byte aligned. Whole file gzip-friendly. Target size ≤ 40 MB for the full brain.
JS `FlyBrain.forward(planesFloat32) -> {policy: Float32Array(4168), value: number, activity: Float32Array(n)}` must match
Python within `1e-2` on logits for the vectors in `tests/vectors/model.json` (`{fen, top5: [[idx, logit]], value}`),
generated by `export/testvectors.py` from the exported f16 weights (so both sides use identical rounded weights).

## 9. CLI (`cli.py`)

```
fly download [--connectome] [--games] [--months 2014-01,...]        # defaults: both
fly build-brain [--region full|central] [--max-neurons N] [--out data/brain/<name>.npz]
fly build-shards [--months ...] [--min-elo 1800] [--workers 16]
fly train --run NAME [--stage imitation|selfplay|all] [--config cfg.yaml] [--resume] [--steps N] [--tiny]
fly dashboard [--run NAME] [--port 8765]                            # http://localhost:8765
fly play [--run NAME | --ckpt path] [--difficulty larva|fly|superfly] [--color white|black] [--gui]
fly eval --run NAME [--games 50] [--opponent random|material|stockfish|<other-run>]
fly export-web --run NAME [--out web/model] [--quant f16|i8]
fly test-vectors --run NAME
```
Difficulty: `larva` = sample policy with temperature 1.2 (no search); `fly` = argmax policy + 1-ply value check;
`superfly` = MCTS 200 simulations. All three use the fly brain exclusively.

## 10. Dashboard (`dashboard/`)

`fly dashboard --run NAME` serves `static/index.html` at `/`, `GET /api/runs` (list runs), `GET /api/run/<name>`
(run.json + last 5000 metrics), `WS /ws/<name>` (pushes each new metrics line). UI: run selector; live charts
(loss / policy / value / top-k / lr / throughput / Elo) using a small vendored charting lib or hand-rolled canvas;
status bar (stage, step, ETA, GPU mem); latest self-play game rendered on a board (auto-replay); neuron activity
heatmap (positions from graph → 2D projection); log tail. Dark theme, fly-themed accents. No build step.

## 11. Website (`web/`)

Static, no build step, ES modules. `index.html` loads `app.js` → creates a Web Worker (`engine/worker.js`) that
fetches `model/brain.json` + `brain.flyb` with a progress bar ("growing the fly brain… 12 MB / 31 MB"), instantiates
`FlyBrain` (`engine/flybrain.js`: pure typed-array CSR SpMV, identical math to §4) and answers `{type:'move', fen,
difficulty}` with `{move, policyTop, value, activity(sampled), thinkMs}`. `engine/mcts.js` implements the same MCTS
as Python for `superfly`. Main thread: board UI (SVG, drag & click, legal-move dots, last-move highlight, promotion
picker), fly avatar with mood driven by the value head (confident / nervous / panicking / smug), live "fly brain"
canvas showing sampled neuron activity as glowing dots at connectome positions, move commentary from the
policy/value ("the fly is 87% sure about this one"), share-result card (canvas → PNG copy), local leaderboard
(localStorage), sound effects (optional toggle), party mode (pass-and-play tournament bracket vs. the fly, multiple
humans take turns, fastest win wins). Mobile-friendly. Footer credits FlyWire (CC-BY 4.0) with citations.
`scripts/deploy-pages.sh` publishes `web/` (with `web/model/`) to a `gh-pages` branch; model files ≤ 100 MB each.
