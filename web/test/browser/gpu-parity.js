// Browser side of web/test/browser/gpu-parity.mjs: loads the exported model from ../../model/,
// builds FlyBrain (plain JS) and FlyBrainGPU on the same arrays and compares them on a few
// positions, then times both. Exposes window.gpuParity(opts) → result object (JSON-serialisable).
import { loadBrain } from '../../engine/loader.js';
import { FlyBrain } from '../../engine/flybrain.js';
import { FlyBrainGPU, emulateForward } from '../../engine/flybrain-gpu.js';
import { Chess } from '../../vendor/chess.js';
import * as enc from '../../engine/encoding.js';

const POSITIONS = [
  { name: 'start', moves: [] },
  { name: 'italian', moves: ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5'] },
  { name: 'black to move', moves: ['d2d4', 'g8f6', 'c2c4', 'e7e6', 'b1c3', 'f8b4', 'd1c2'] },
  { name: 'middlegame', fen: 'r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 2 8' },
  { name: 'endgame', fen: '8/5pk1/6p1/8/3K4/8/5PPP/8 w - - 0 40' },
];

function encode(pos) {
  const chess = new Chess();
  if (pos.moves) for (const u of pos.moves) chess.move(enc.uciToMove(u));
  else chess.load(pos.fen);
  const moves = chess.moves({ verbose: true });
  return { x: enc.encodeBoard(chess), legal: moves.map((m) => enc.moveToIndex(m, chess)) };
}

const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const argmaxLegal = (p, legal) => legal.reduce((best, i) => (p[i] > p[best] ? i : best), legal[0]);
const log = (s) => { document.getElementById('out').textContent += s + '\n'; console.log(s); };

window.gpuParity = async ({ reps = 20, emulate = true } = {}) => {
  const model = await loadBrain(new URL('../../model/', location.href).href);
  const js = new FlyBrain(model);
  const t0 = performance.now();
  const gpu = await FlyBrainGPU.create(model);
  const createMs = performance.now() - t0;
  const adapterInfo = gpu.adapter?.info ? JSON.parse(JSON.stringify(gpu.adapter.info)) : null;
  log(`model ${model.header.run_name}: n=${js.n} nnz=${js.nnz} steps=${js.steps} act=${js.activation}; GPU ready in ${createMs.toFixed(0)} ms, timestamps=${gpu.timestamps}`);

  const positions = [];
  for (const pos of POSITIONS) {
    const { x, legal } = encode(pos);
    const a = js.forward(x);
    const aPolicy = a.policy.slice(), aActivity = a.activity.slice();
    const b = await gpu.forward(x);
    const r = {
      name: pos.name,
      dLogit: maxAbs(aPolicy, b.policy),
      dLogitLegal: Math.max(...legal.map((i) => Math.abs(aPolicy[i] - b.policy[i]))),
      dValue: Math.abs(a.value - b.value),
      dActivity: maxAbs(aActivity, b.activity),
      sameArgmax: argmaxLegal(aPolicy, legal) === argmaxLegal(b.policy, legal),
      valueJs: a.value, valueGpu: b.value,
      maxActivity: b.activity.reduce((m, v) => Math.max(m, v), 0),
    };
    if (emulate) {
      const e = emulateForward(model, x);
      r.dLogitEmu = maxAbs(e.policy, b.policy);
      r.dActivityEmu = maxAbs(e.activity, b.activity);
      r.dValueEmu = Math.abs(e.value - b.value);
    }
    positions.push(r);
    log(`${pos.name.padEnd(14)} |Δlogit| ${r.dLogit.toExponential(2)} (legal ${r.dLogitLegal.toExponential(2)})  |Δvalue| ${r.dValue.toExponential(2)}  |Δh| ${r.dActivity.toExponential(2)}  argmax ${r.sameArgmax ? 'same' : 'DIFFERENT'}` + (emulate ? `  vs emulation: logit ${r.dLogitEmu.toExponential(2)} h ${r.dActivityEmu.toExponential(2)}` : ''));
  }

  // latency (single forward, serial; the MCTS pattern)
  const { x } = encode(POSITIONS[1]);
  const time = async (fn, n) => {
    const ts = [];
    for (let i = 0; i < n; i++) { const t = performance.now(); await fn(); ts.push(performance.now() - t); }
    ts.sort((p, q) => p - q);
    return { median: ts[ts.length >> 1], min: ts[0], max: ts[ts.length - 1], mean: ts.reduce((s, v) => s + v, 0) / ts.length };
  };
  await gpu.forward(x); await gpu.forward(x);
  const gpuFull = await time(() => gpu.forward(x), reps);
  const gpuStepMs = gpu.lastStepMs;
  const gpuNoAct = await time(() => gpu.forward(x, { activity: false }), reps);
  const jsT = await time(() => js.forward(x), Math.max(3, Math.min(reps, 5)));
  const timing = { gpuForwardMs: gpuFull, gpuForwardNoActivityMs: gpuNoAct, gpuStepMs, jsForwardMs: jsT, jsStepMs: js.lastStepMs, createMs };
  log(`GPU forward: median ${gpuFull.median.toFixed(2)} ms (min ${gpuFull.min.toFixed(2)}, max ${gpuFull.max.toFixed(2)}), without activity readback ${gpuNoAct.median.toFixed(2)} ms; GPU step ${gpuStepMs.toFixed(3)} ms${gpu.timestamps ? ' (timestamp query)' : ' (estimate)'}`);
  log(`JS  forward: median ${jsT.median.toFixed(1)} ms, step ${js.lastStepMs.toFixed(2)} ms`);
  const limits = gpu.device.limits;
  return { header: { run_name: model.header.run_name, n: js.n, nnz: js.nnz, steps: js.steps, activation: js.activation }, adapterInfo, timestamps: gpu.timestamps, limits: { maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize, maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension }, positions, timing };
};
