// Browser side of the worker check in web/test/browser/gpu-parity.mjs: drives engine/worker.js on
// the real model — WebGPU backend reported in 'ready', moves at every difficulty, bench of both
// backends, then a simulated device loss and the automatic fall-back to the JS engine.
// Exposes window.workerGpu(opts) → result object.
const log = (s) => { document.getElementById('out').textContent += s + '\n'; console.log(s); };

class Client {
  constructor(url) {
    this.w = new Worker(url, { type: 'module' });
    this.pending = new Map(); this.id = 1; this.events = [];
    this.w.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === 'progress' || m.type === 'thinking') { this.events.push(m.type); return; }
      if (m.type === 'backend') { this.events.push(`backend:${m.backend}`); this.backendMsg = m; return; }
      if (m.type === 'ready') { this.ready?.(m); return; }
      if (m.type === 'error' && !m.id) { this.readyReject?.(new Error(m.message)); return; }
      const p = this.pending.get(m.id); this.pending.delete(m.id);
      if (!p) return;
      if (m.type === 'error') p.reject(new Error(m.message)); else p.resolve(m);
    };
  }
  load(opts = {}) {
    return new Promise((resolve, reject) => { this.ready = resolve; this.readyReject = reject; this.w.postMessage({ type: 'load', baseUrl: new URL('../../model/', location.href).href, ...opts }); });
  }
  request(msg) { const id = this.id++; return new Promise((resolve, reject) => { this.pending.set(id, { resolve, reject }); this.w.postMessage({ ...msg, id }); }); }
}

window.workerGpu = async () => {
  const out = {};
  const c = new Client(new URL('../../engine/worker.js', location.href));
  const ready = await c.load();
  out.readyBackend = ready.backend; out.gpuInfo = ready.gpu;
  log(`ready: backend=${ready.backend} gpu=${JSON.stringify(ready.gpu)} n=${ready.header.n}`);
  const start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  out.bench = {};
  for (const backend of ['webgpu', 'js']) {
    try {
      const b = await c.request({ type: 'bench', backend, reps: backend === 'js' ? 3 : 20 });
      out.bench[backend] = { forwardMs: b.forwardMs, stepMs: b.stepMs };
      log(`bench ${backend}: forward ${b.forwardMs.toFixed(2)} ms, step ${b.stepMs.toFixed(3)} ms`);
    } catch (err) { out.bench[backend] = { error: err.message }; log(`bench ${backend}: ${err.message}`); }
  }
  out.moves = {};
  for (const difficulty of ['larva', 'fly', 'superfly']) {
    const r = await c.request({ type: 'move', fen: start, moves: [], difficulty });
    out.moves[difficulty] = { move: r.move, backend: r.backend, thinkMs: r.thinkMs, sims: r.sims, stepMs: r.stepMs, sample: r.activitySample?.length, value: r.value, top: r.policyTop?.[0] };
    log(`${difficulty.padEnd(9)} ${r.san} (${r.move}) backend=${r.backend} think ${r.thinkMs.toFixed(0)} ms sims=${r.sims} value=${r.value.toFixed(3)} sample=${r.activitySample?.length}`);
  }
  const thinking = c.events.filter((e) => e === 'thinking').length;
  out.thinkingEvents = thinking;
  const ev = await c.request({ type: 'eval', fen: start, moves: [] });
  out.eval = { backend: ev.backend, value: ev.value, sample: ev.activitySample?.length };
  // the same evaluation through the JS engine must agree (both engines, same position)
  const jsClient = new Client(new URL('../../engine/worker.js', location.href));
  const jsReady = await jsClient.load({ gpu: false });
  const evJs = await jsClient.request({ type: 'eval', fen: start, moves: [] });
  out.evalJs = { backend: evJs.backend, readyBackend: jsReady.backend, value: evJs.value, gpuInfo: jsReady.gpu };
  out.evalAgree = Math.abs(ev.value - evJs.value) <= 1e-2 && ev.policyTop[0].uci === evJs.policyTop[0].uci && Math.abs(ev.policyTop[0].p - evJs.policyTop[0].p) <= 1e-2;
  log(`eval: gpu value ${ev.value.toFixed(5)} top ${ev.policyTop[0].uci} ${ev.policyTop[0].p.toFixed(4)} | js value ${evJs.value.toFixed(5)} top ${evJs.policyTop[0].uci} ${evJs.policyTop[0].p.toFixed(4)} (gpu:false load → ${jsReady.backend}, ${JSON.stringify(jsReady.gpu)})`);
  jsClient.w.terminate();
  // device loss → JS fallback
  const dbg = await c.request({ type: 'debug', op: 'lose-gpu' });
  out.afterLoseDebug = dbg.backend;
  const r2 = await c.request({ type: 'move', fen: start, moves: [], difficulty: 'fly' });
  out.afterLose = { move: r2.move, backend: r2.backend, backendMsg: c.backendMsg || null };
  log(`after lose-gpu: move ${r2.san} backend=${r2.backend} notice=${JSON.stringify(c.backendMsg)}`);
  const r3 = await c.request({ type: 'move', fen: start, moves: [], difficulty: 'superfly' });
  out.afterLoseSuperfly = { move: r3.move, backend: r3.backend, sims: r3.sims };
  log(`after lose-gpu: superfly ${r3.san} backend=${r3.backend} sims=${r3.sims}`);
  c.w.terminate();
  return out;
};
