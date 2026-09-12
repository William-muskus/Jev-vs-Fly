// worker.js — Web Worker hosting the fly brain. Every move sent back was chosen by the network
// (SPEC §9): larva = temperature-1.2 sample of the policy, fly = argmax policy with a 1-ply
// value-head check over the top-3 policy moves, superfly = 200-simulation PUCT MCTS.
//
// Messages in:  {type:'load', baseUrl}
//               {type:'move', id, fen, moves:[uci...], difficulty:'larva'|'fly'|'superfly'}
//               {type:'eval', id, fen, moves:[uci...]}
// Messages out: {type:'progress', loaded, total, phase, n, nnz, runName}
//               {type:'ready', header, sample:{idx, xy, cls}, silhouette:{xy, cls}, legend, fromCache, bytes}
//               {type:'thinking', id, done, total}            (superfly only, every 10 simulations)
//               {type:'move', id, move, san, policyTop, value, activitySample, thinkMs, sims, stepMs}
//               {type:'eval', id, value, policyTop, activitySample}
//               {type:'error', id?, message}

import { Chess } from '../vendor/chess.js';
import * as enc from './encoding.js';
import { loadBrain } from './loader.js';
import { FlyBrain, policyForLegal, sampleIndex, topK } from './flybrain.js';
import { runMCTS, uciOf } from './mcts.js';

const SAMPLE_N = 2048;
const SILHOUETTE_N = 6000;
const DIFFICULTY = {
  larva: { kind: 'sample', temperature: 1.2 },
  fly: { kind: 'lookahead', topN: 3 },
  superfly: { kind: 'mcts', sims: 200 },
};

let brain = null;
let sampleIdx = null;         // Int32Array(SAMPLE_N) fixed at load
const chess = new Chess();

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'load') await handleLoad(msg);
    else if (msg.type === 'move') handleMove(msg);
    else if (msg.type === 'eval') handleEval(msg);
    else if (msg.type === 'bench') handleBench(msg);
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: String(err && err.message || err) });
  }
};

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

function handleMove(msg) {
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
      else if (chess.isDraw()) oppValue = 0;
      else oppValue = brain.forward(enc.encodeBoard(chess)).value;
      chess.undo();
      const score = -oppValue;       // our value = negated opponent value
      if (score > bestScore + 1e-9) { bestScore = score; best = i; }
    }
    chosen = moves[best];
    value = bestScore;
    sims = cands.length;
  } else {
    const out = runMCTS(brain, chess, enc, {
      sims: cfg.sims, cPuct: 1.5, dirichletAlpha: 0, temperature: 0,
      onProgress: (done, total) => { if (done % 10 === 0) self.postMessage({ type: 'thinking', id: msg.id, done, total }); },
    });
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
