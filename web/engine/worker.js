// worker.js — Web Worker hosting the fly brain. Every move sent back was chosen by the network
// (SPEC §9): larva = temperature-1.2 sample of the policy, fly = argmax policy with a 1-ply
// value-head check over the top-3 policy moves, superfly = 100-simulation PUCT MCTS (≈5 s per move on the full brain in plain JS; Python uses 200).
//
// Messages in:  {type:'load', baseUrl}
//               {type:'move', id, fen, moves:[uci...], difficulty:'larva'|'fly'|'superfly'}
//               {type:'eval', id, fen, moves:[uci...]}
//               {type:'cancel', id}     abandon the move request `id` (a running superfly search stops
//                                       within a few simulations; a queued one is skipped)
// Messages out: {type:'progress', loaded, total, phase, n, nnz, runName}
//               {type:'ready', header, sample:{idx, xy, cls}, silhouette:{xy, cls}, legend, fromCache, bytes}
//               {type:'thinking', id, done, total}            (superfly only, every 10 simulations)
//               {type:'move', id, move, san, policyTop, value, activitySample, thinkMs, sims, stepMs}
//               {type:'move', id, move:null, cancelled:true}  (reply to a cancelled request)
//               {type:'eval', id, value, policyTop, activitySample}
//               {type:'error', id?, message}
//
// Requests are handled strictly one at a time (they share one chess.js instance); `cancel` is the
// only message acted on immediately.

import { Chess } from '../vendor/chess.js';
import * as enc from './encoding.js';
import { loadBrain } from './loader.js';
import { FlyBrain, policyForLegal, sampleIndex, topK } from './flybrain.js';
import { runMCTSAsync, uciOf } from './mcts.js';

const SAMPLE_N = 2048;
const SILHOUETTE_N = 6000;
const DIFFICULTY = {
  larva: { kind: 'sample', temperature: 1.2 },
  fly: { kind: 'lookahead', topN: 3 },
  superfly: { kind: 'mcts', sims: 100 },
};

let brain = null;
let sampleIdx = null;         // Int32Array(SAMPLE_N) fixed at load
const chess = new Chess();

let queue = Promise.resolve();          // requests run one after another
let activeSearch = null;                // {id, cancelled} while a superfly search is in flight
const cancelledIds = new Set();         // cancelled requests not yet dispatched

self.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === 'cancel') {
    if (activeSearch && activeSearch.id === msg.id) activeSearch.cancelled = true;
    else { cancelledIds.add(msg.id); if (cancelledIds.size > 256) cancelledIds.clear(); }
    return;
  }
  queue = queue.then(() => dispatch(msg));
};

async function dispatch(msg) {
  try {
    if (msg.type === 'load') await handleLoad(msg);
    else if (msg.type === 'move') await handleMove(msg);
    else if (msg.type === 'eval') handleEval(msg);
    else if (msg.type === 'bench') handleBench(msg);
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: String(err && err.message || err) });
  }
}

async function handleLoad(msg) {
  // relative URLs are resolved against the page by app.js; resolve against this script otherwise
  const baseUrl = new URL(msg.baseUrl || '../model/', self.location?.href).href;
  const model = await loadBrain(baseUrl, (loaded, total, phase, header) => {
    self.postMessage({ type: 'progress', loaded, total, phase, n: header?.n, nnz: header?.nnz, runName: header?.run_name });
  });
  brain = new FlyBrain(model);
  const n = brain.n;
  sampleIdx = spreadSample(n, Math.min(SAMPLE_N, n), 0x9e3779b9);
  const silIdx = spreadSample(n, Math.min(SILHOUETTE_N, n), 0x85ebca6b);
  const legend = readLegend(model.header);
  self.postMessage({
    type: 'ready',
    header: model.header,
    fromCache: model.fromCache,
    bytes: model.bytes,
    legend,
    sample: { idx: sampleIdx, xy: projectXY(brain, sampleIdx), cls: classesOf(brain, sampleIdx) },
    silhouette: { xy: projectXY(brain, silIdx), cls: classesOf(brain, silIdx) },
  });
}

/** Deterministic, well-spread sample of k indices out of n (golden-ratio stride + hash jitter). */
function spreadSample(n, k, seed) {
  const idx = new Int32Array(k);
  const seen = new Set();
  let x = seed >>> 0;
  for (let i = 0; i < k; i++) {
    let v;
    do {
      x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;   // xorshift32
      v = x % n;
    } while (seen.has(v));
    seen.add(v);
    idx[i] = v;
  }
  idx.sort();
  return idx;
}

function projectXY(b, idx) {
  const xy = new Float32Array(idx.length * 2);
  if (!b.positions) { for (let i = 0; i < idx.length; i++) { xy[2 * i] = Math.random(); xy[2 * i + 1] = Math.random(); } return xy; }
  const P = b.positions;
  for (let i = 0; i < idx.length; i++) { xy[2 * i] = P[idx[i] * 3]; xy[2 * i + 1] = P[idx[i] * 3 + 1]; }
  return xy;
}

function classesOf(b, idx) {
  const c = new Uint8Array(idx.length);
  if (b.superClass) for (let i = 0; i < idx.length; i++) c[i] = b.superClass[idx[i]];
  return c;
}

function readLegend(header) {
  const raw = header.super_class_legend || header.super_classes || header.legend || null;
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  const out = [];
  for (const [k, v] of Object.entries(raw)) {
    if (Number.isFinite(+k)) out[+k] = v; else if (Number.isFinite(+v)) out[+v] = k;
  }
  return out;
}

/** Put the worker's chess instance into the requested game state (history matters for repetition). */
function setPosition(fen, moves) {
  chess.reset();
  let ok = false;
  if (Array.isArray(moves) && moves.length) {
    ok = true;
    for (const u of moves) {
      try { chess.move({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u[4] || undefined }); }
      catch { ok = false; break; }
    }
    if (ok && fen && chess.fen() !== fen) ok = false;
  }
  if (!ok) chess.load(fen);
}

function requireBrain() {
  if (!brain) throw new Error('brain not loaded');
}

function legalMoves() {
  const moves = chess.moves({ verbose: true });
  const idx = moves.map((m) => enc.moveToIndex(m, chess));
  return { moves, idx };
}

function policyTopFrom(moves, probs, k = 5) {
  const order = topK(probs, k);
  return order.map((i) => ({ uci: uciOf(moves[i]), san: moves[i].san, p: probs[i] }));
}

function activitySample(activity) {
  const out = new Float32Array(sampleIdx.length);
  for (let i = 0; i < sampleIdx.length; i++) out[i] = activity[sampleIdx[i]];
  return out;
}

async function handleMove(msg) {
  if (cancelledIds.delete(msg.id)) {
    self.postMessage({ type: 'move', id: msg.id, move: null, san: null, cancelled: true, policyTop: [], value: 0, activitySample: null, thinkMs: 0, sims: 0 });
    return;
  }
  requireBrain();
  const t0 = performance.now();
  setPosition(msg.fen, msg.moves);
  const cfg = DIFFICULTY[msg.difficulty] || DIFFICULTY.fly;
  const { moves, idx } = legalMoves();
  if (moves.length === 0) {
    self.postMessage({ type: 'move', id: msg.id, move: null, san: null, policyTop: [], value: 0, activitySample: null, thinkMs: 0, sims: 0 });
    return;
  }
  const x = enc.encodeBoard(chess);
  const res = brain.forward(x);
  const probs = policyForLegal(res.policy, idx, 1);
  const act = activitySample(res.activity);
  let chosen, value = res.value, sims = 0, note = '';

  if (cfg.kind === 'sample') {
    const pT = policyForLegal(res.policy, idx, cfg.temperature);
    chosen = moves[sampleIndex(pT)];
  } else if (cfg.kind === 'lookahead') {
    // top-N policy candidates; play each, ask the brain how the opponent likes it, keep the worst for them
    const cands = topK(probs, Math.min(cfg.topN, moves.length));
    let best = cands[0], bestScore = -Infinity;
    for (const i of cands) {
      chess.move(moves[i]);
      let oppValue;
      if (chess.isCheckmate()) oppValue = -1;
      else if (chess.isDraw() || enc.repetitionCount(chess) >= 3) oppValue = 0;
      else oppValue = brain.forward(enc.encodeBoard(chess)).value;
      chess.undo();
      const score = -oppValue;       // our value = negated opponent value
      if (score > bestScore + 1e-9) { bestScore = score; best = i; }
    }
    chosen = moves[best];
    value = bestScore;
    sims = cands.length;
  } else {
    activeSearch = { id: msg.id, cancelled: false };
    let out;
    try {
      out = await runMCTSAsync(brain, chess, enc, {
        sims: cfg.sims, cPuct: 1.5, dirichletAlpha: 0, temperature: 0, yieldEvery: 10,
        shouldStop: () => activeSearch.cancelled,
        onProgress: (done, total) => { if (done % 10 === 0) self.postMessage({ type: 'thinking', id: msg.id, done, total }); },
      });
    } finally {
      activeSearch = null;
    }
    if (out.stopped) {
      self.postMessage({ type: 'move', id: msg.id, move: null, san: null, cancelled: true, policyTop: [], value: 0, activitySample: null, thinkMs: performance.now() - t0, sims: out.sims });
      return;
    }
    chosen = out.moveObj;
    value = out.rootValue;
    sims = out.sims;
    note = out.visits.slice(0, 5).map((v) => `${v.uci}:${v.n}`).join(' ');
  }

  const thinkMs = performance.now() - t0;
  self.postMessage({
    type: 'move', id: msg.id,
    move: uciOf(chosen), san: chosen.san,
    policyTop: policyTopFrom(moves, probs, 5),
    value, activitySample: act, thinkMs, sims, stepMs: brain.lastStepMs, note,
  });
}

function handleEval(msg) {
  requireBrain();
  setPosition(msg.fen, msg.moves);
  const { moves, idx } = legalMoves();
  if (moves.length === 0) {
    self.postMessage({ type: 'eval', id: msg.id, value: 0, policyTop: [], activitySample: null });
    return;
  }
  const res = brain.forward(enc.encodeBoard(chess));
  const probs = policyForLegal(res.policy, idx, 1);
  self.postMessage({
    type: 'eval', id: msg.id, value: res.value,
    policyTop: policyTopFrom(moves, probs, 5), activitySample: activitySample(res.activity),
  });
}

function handleBench(msg) {
  requireBrain();
  chess.reset();
  const x = enc.encodeBoard(chess);
  const reps = msg.reps || 5;
  const t0 = performance.now();
  for (let i = 0; i < reps; i++) brain.forward(x);
  const ms = (performance.now() - t0) / reps;
  self.postMessage({ type: 'bench', id: msg.id, forwardMs: ms, stepMs: brain.lastStepMs });
}
