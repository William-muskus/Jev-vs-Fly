// node web/test/browser/gpu-parity.mjs   (not part of `node --test`: needs a Chromium with WebGPU
// and the real model in web/model/). Serves web/, launches headless Chromium with WebGPU on the
// hardware adapter, opens test/browser/gpu-parity.html and checks FlyBrainGPU against FlyBrain:
//   |Δlogit| <= 1e-2 on every policy logit, |Δvalue| <= 1e-2, same legal argmax, on 5 positions,
// then prints the measured latencies. Exit code 1 on a mismatch, 2 when WebGPU could not start.
//
// Chromium is found like web/test/app.test.mjs (CHROME=/path/to/chrome overrides). Flags that make
// WebGPU work headless on Linux (Chromium 149, NVIDIA/AMD via Vulkan):
//   --headless=new --enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan --use-angle=vulkan
// (do NOT pass --disable-gpu / --use-angle=swiftshader-webgl; SwiftShader can be forced with
//  GPU_ANGLE=swiftshader for a software run). The page must be served over http: navigator.gpu is
// not exposed on about:blank.
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync, readdirSync, readFileSync, statSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { dirname, join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const WEB = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.flyb': 'application/octet-stream', '.gz': 'application/octet-stream' };
const TOL = 1e-2;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function findChrome() {
  const cands = [process.env.CHROME, process.env.CHROME_PATH, '/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable'];
  const pw = join(homedir(), '.cache', 'ms-playwright');
  if (existsSync(pw)) for (const d of readdirSync(pw).sort().reverse()) if (d.startsWith('chromium-')) cands.push(join(pw, d, 'chrome-linux64', 'chrome'), join(pw, d, 'chrome-linux', 'chrome'));
  return cands.find((c) => c && existsSync(c)) || null;
}

if (!existsSync(join(WEB, 'model', 'brain.json'))) { console.error('no web/model/brain.json — run `fly export-web` first'); process.exit(2); }
const CHROME = findChrome();
if (!CHROME) { console.error('no Chromium found (set CHROME=/path/to/chrome)'); process.exit(2); }

const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const file = join(WEB, path === '/' ? 'index.html' : path);
  if (!file.startsWith(WEB) || !existsSync(file) || statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
  if (req.method === 'HEAD') { res.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream', 'content-length': statSync(file).size }); res.end(); return; }
  res.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream' });
  res.end(readFileSync(file));
});

class Page {
  constructor(ws) {
    this.ws = ws; this.id = 0; this.pending = new Map(); this.console = [];
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && this.pending.has(m.id)) { this.pending.get(m.id)(m); this.pending.delete(m.id); }
      else if (m.method === 'Runtime.consoleAPICalled') this.console.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
      else if (m.method === 'Runtime.exceptionThrown') this.console.push('EXCEPTION ' + (m.params.exceptionDetails?.exception?.description || m.params.exceptionDetails?.text));
    };
  }
  send(method, params = {}) { return new Promise((r) => { const i = ++this.id; this.pending.set(i, r); this.ws.send(JSON.stringify({ id: i, method, params })); }); }
  async eval(expr) {
    const m = await this.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    if (m.result?.exceptionDetails) throw new Error(`page eval failed: ${m.result.exceptionDetails.exception?.description || m.result.exceptionDetails.text}`);
    return m.result?.result?.value;
  }
}

await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/`;
const profile = mkdtempSync(join(tmpdir(), 'flychess-gpu-'));
const port = 9400 + Math.floor(Math.random() * 400);
const angle = process.env.GPU_ANGLE || 'vulkan';
const flags = ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-features=Vulkan', `--use-angle=${angle}`,
  ...(angle === 'swiftshader' ? ['--enable-unsafe-swiftshader'] : []),
  '--hide-scrollbars', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, 'about:blank'];
console.log(`chromium: ${CHROME}\nflags: ${flags.slice(0, -3).join(' ')}`);
const chrome = spawn(CHROME, flags, { stdio: 'ignore' });
let code = 0;
try {
  let targets = null;
  for (let i = 0; i < 100 && !targets; i++) { try { targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); } catch { await sleep(100); } }
  if (!targets) throw new Error('chromium did not start');
  const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  const page = new Page(ws);
  await page.send('Runtime.enable'); await page.send('Page.enable');
  await page.send('Page.navigate', { url: base + 'test/browser/gpu-parity.html' });
  for (let i = 0; i < 200 && !(await page.eval('typeof window.gpuParity === "function"')); i++) await sleep(50);
  const gpuAvail = await page.eval('(async () => { if (!navigator.gpu) return "navigator.gpu undefined"; for (let i = 0; i < 4; i++) { const a = await navigator.gpu.requestAdapter(); if (a) return "ok"; await new Promise((r) => setTimeout(r, 300)); } return "requestAdapter() returned null"; })()');
  if (gpuAvail !== 'ok') { console.error(`WebGPU unavailable in headless Chromium: ${gpuAvail}`); code = 2; }
  else {
    const reps = Number(process.env.REPS || 30);
    const r = await page.eval(`window.gpuParity({ reps: ${reps}, emulate: ${process.env.NO_EMULATE ? 'false' : 'true'} })`);
    for (const line of page.console) console.log('  ' + line);
    console.log('\nadapter:', JSON.stringify(r.adapterInfo), 'limits:', JSON.stringify(r.limits), 'timestamps:', r.timestamps);
    const bad = r.positions.filter((p) => !(p.dLogit <= TOL && p.dValue <= TOL && p.sameArgmax));
    const worst = (k) => Math.max(...r.positions.map((p) => p[k] ?? 0));
    console.log(`\nworst over ${r.positions.length} positions: |Δlogit| ${worst('dLogit').toExponential(2)}  |Δvalue| ${worst('dValue').toExponential(2)}  |Δactivity| ${worst('dActivity').toExponential(2)}` + (r.positions[0].dLogitEmu !== undefined ? `  (GPU vs f32 emulation: |Δlogit| ${worst('dLogitEmu').toExponential(2)}, |Δactivity| ${worst('dActivityEmu').toExponential(2)})` : ''));
    const t = r.timing;
    console.log(`GPU forward median ${t.gpuForwardMs.median.toFixed(2)} ms (no activity readback ${t.gpuForwardNoActivityMs.median.toFixed(2)} ms; step ${t.gpuStepMs.toFixed(3)} ms) — JS forward median ${t.jsForwardMs.median.toFixed(1)} ms — ${(t.jsForwardMs.median / t.gpuForwardMs.median).toFixed(1)}x`);
    if (bad.length) { console.error(`\nPARITY FAILURE on: ${bad.map((p) => p.name).join(', ')} (tolerance ${TOL})`); code = 1; }
    else console.log(`\nPARITY OK (|Δlogit| <= ${TOL}, |Δvalue| <= ${TOL}, same argmax) on ${r.positions.length} positions`);

    // --- the worker: backend report, moves, bench, device loss → JS fallback
    console.log('\nworker (engine/worker.js on the real model):');
    page.console.length = 0;
    await page.send('Page.navigate', { url: base + 'test/browser/worker-gpu.html' });
    for (let i = 0; i < 200 && !(await page.eval('typeof window.workerGpu === "function"')); i++) await sleep(50);
    const w = await page.eval('window.workerGpu()');
    for (const line of page.console) console.log('  ' + line);
    const problems = [];
    if (w.readyBackend !== 'webgpu') problems.push(`ready.backend = ${w.readyBackend} (${JSON.stringify(w.gpuInfo)})`);
    for (const [d, m] of Object.entries(w.moves)) { if (!m.move) problems.push(`${d}: no move`); if (m.backend !== 'webgpu') problems.push(`${d}: backend ${m.backend}`); if (m.sample !== 2048) problems.push(`${d}: activity sample ${m.sample}`); }
    if (w.moves.superfly.sims !== 100 || w.thinkingEvents < 9) problems.push(`superfly: sims ${w.moves.superfly.sims}, thinking events ${w.thinkingEvents}`);
    if (!w.evalAgree) problems.push(`eval: webgpu and js workers disagree (${JSON.stringify(w.eval)} vs ${JSON.stringify(w.evalJs)})`);
    if (w.evalJs.readyBackend !== 'js') problems.push(`gpu:false load reported ${w.evalJs.readyBackend}`);
    if (w.afterLose.backend !== 'js' || !w.afterLose.move || !w.afterLose.backendMsg) problems.push(`after device loss: ${JSON.stringify(w.afterLose)}`);
    if (w.afterLoseSuperfly.backend !== 'js' || w.afterLoseSuperfly.sims !== 100) problems.push(`superfly after device loss: ${JSON.stringify(w.afterLoseSuperfly)}`);
    if (problems.length) { console.error('\nWORKER FAILURE:\n  ' + problems.join('\n  ')); code = 1; }
    else console.log(`\nWORKER OK: webgpu backend for larva/fly/superfly (superfly ${w.moves.superfly.thinkMs.toFixed(0)} ms for ${w.moves.superfly.sims} sims), bench webgpu ${w.bench.webgpu.forwardMs?.toFixed(2)} ms vs js ${w.bench.js.forwardMs?.toFixed(1)} ms, device loss → js fallback`);
  }
  ws.close();
} catch (err) {
  console.error(err.stack || String(err));
  code = code || 1;
} finally {
  chrome.kill('SIGKILL'); server.close();
  await sleep(200); rmSync(profile, { recursive: true, force: true });
}
process.exit(code);
