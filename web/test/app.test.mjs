// node --test web/test/app.test.mjs -- the real page in headless Chromium (raw CDP, no dependencies).
// Serves web/ plus a synthetic model and drives window.flychess. Skipped when no Chromium is found
// (set CHROME=/path/to/chrome to point at one).
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync, readdirSync, readFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { dirname, join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { synthModel } from './support/synthmodel.mjs';

const WEB = join(dirname(fileURLToPath(import.meta.url)), '..');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json', '.flyb': 'application/octet-stream' };

function findChrome() {
  const cands = [process.env.CHROME, process.env.CHROME_PATH, '/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable'];
  const pw = join(homedir(), '.cache', 'ms-playwright');
  if (existsSync(pw)) for (const d of readdirSync(pw)) if (d.startsWith('chromium')) cands.push(join(pw, d, 'chrome-linux64', 'chrome'), join(pw, d, 'chrome-linux', 'chrome'));
  return cands.find((c) => c && existsSync(c)) || null;
}
const CHROME = findChrome();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ static server (web/ + model)
const model = synthModel();
const serve = { model: true };
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (path.startsWith('/model/')) {
    if (!serve.model) { res.writeHead(404); res.end('no model'); return; }
    if (path === '/model/brain.json') { res.writeHead(200, { 'content-type': MIME['.json'] }); res.end(JSON.stringify(model.header)); return; }
    if (path === '/model/brain.flyb') { res.writeHead(200, { 'content-type': MIME['.flyb'], 'content-length': model.blob.byteLength }); res.end(model.blob); return; }
    res.writeHead(404); res.end(); return;
  }
  const file = join(WEB, path === '/' ? 'index.html' : path);
  if (!file.startsWith(WEB) || !existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream' });
  res.end(readFileSync(file));
});

// ------------------------------------------------------------------ minimal CDP client
class Page {
  constructor(ws) {
    this.ws = ws; this.id = 0; this.pending = new Map(); this.errors = [];
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && this.pending.has(m.id)) { this.pending.get(m.id)(m); this.pending.delete(m.id); }
      else if (m.method === 'Runtime.exceptionThrown') this.errors.push(m.params.exceptionDetails?.exception?.description || m.params.exceptionDetails?.text);
    };
  }
  send(method, params = {}) { return new Promise((r) => { const i = ++this.id; this.pending.set(i, r); this.ws.send(JSON.stringify({ id: i, method, params })); }); }
  /** evaluate a JS expression (may be a promise) and return its JSON value */
  async eval(expr) {
    const m = await this.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    if (m.result?.exceptionDetails) throw new Error(`page eval failed: ${m.result.exceptionDetails.exception?.description || m.result.exceptionDetails.text}\n${expr}`);
    return m.result?.result?.value;
  }
  async waitFor(expr, { timeout = 15000, what = expr } = {}) {
    const t0 = Date.now();
    while (Date.now() - t0 < timeout) { if (await this.eval(expr)) return; await sleep(40); }
    throw new Error(`timed out waiting for ${what}`);
  }
  click(sel) { return this.eval(`document.querySelector(${JSON.stringify(sel)}).click(), true`); }
  async key(key, { code = `Key${key.toUpperCase()}`, modifiers = 0 } = {}) {
    await this.send('Input.dispatchKeyEvent', { type: 'keyDown', key, code, text: modifiers ? undefined : key, modifiers });
    await this.send('Input.dispatchKeyEvent', { type: 'keyUp', key, code, modifiers });
  }
  async goto(url) {
    await this.send('Page.navigate', { url });
    await this.waitFor(`document.readyState === 'complete' && !!window.flychess`, { what: 'page load' });
  }
}

let chrome, page, base, profile;
const SNAP = `(() => { const a = window.flychess, $ = (i) => document.getElementById(i); return {
  status: $('status').textContent, error: $('status').classList.contains('error'), retry: !$('btn-retry').hidden,
  mood: $('avatar-wrap').dataset.mood, faded: $('commentary').classList.contains('fade'), commentary: $('commentary').textContent,
  thinking: a.state?.thinking, movable: a.board?.movable ?? null, undo: $('btn-undo').disabled, hist: a.chess.history(),
  top: $('who-top').textContent, bottom: $('who-bottom').textContent, orientation: a.board?.orientation,
  party: a.party ? { names: a.party.names, current: a.party.current, results: a.party.results.map((r) => r && r.name) } : null,
  partyBar: !$('party-bar').hidden, dead: a.brain.dead, sims: $('clk-sims').textContent }; })()`;
const HUMAN_TURN = `(() => { const a = window.flychess; return !!a.state && (!!a.state.result || (!a.state.thinking && a.chess.turn() === a.state.human)); })()`;

async function loadPage() {
  await page.goto(base);
  await page.waitFor(`!document.getElementById('btn-start').disabled`, { what: 'brain load', timeout: 30000 });
}
async function startGame({ difficulty = 'fly', color = 'white', name = 'Tester' } = {}) {
  if (await page.eval(`document.getElementById('result-modal').open`)) await page.eval(`document.getElementById('result-modal').close(), true`);
  if (await page.eval(`document.getElementById('screen-landing').hidden`)) await page.click('#brand');
  await page.eval(`document.querySelector('input[name=difficulty][value=${difficulty}]').checked = true; document.querySelector('input[name=color][value=${color}]').checked = true; document.getElementById('player-name').value = ${JSON.stringify(name)}; true`);
  await page.click('#btn-start');
  await page.waitFor(HUMAN_TURN, { what: 'first human turn' });
}
const humanMove = (from, to) => page.eval(`window.flychess.humanMove('${from}', '${to}'), true`);

before(async () => {
  if (!CHROME) return;
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  base = `http://127.0.0.1:${server.address().port}/`;
  profile = mkdtempSync(join(tmpdir(), 'flychess-test-'));
  const port = 9400 + Math.floor(Math.random() * 400);
  chrome = spawn(CHROME, ['--headless=new', '--no-sandbox', '--disable-gpu', '--hide-scrollbars', '--window-size=1400,900', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, 'about:blank'], { stdio: 'ignore' });
  let targets = null;
  for (let i = 0; i < 100 && !targets; i++) { try { targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); } catch { await sleep(100); } }
  assert.ok(targets, 'chromium did not start');
  const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  page = new Page(ws);
  await page.send('Runtime.enable'); await page.send('Page.enable');
});
after(async () => {
  page?.ws.close(); chrome?.kill('SIGKILL'); server.close();
  if (profile) await sleep(200).then(() => rmSync(profile, { recursive: true, force: true }));
});

const opts = { skip: CHROME ? false : 'no Chromium found (set CHROME=/path/to/chrome)' };

test('flip swaps the player labels and is ignored behind a modal / with Ctrl', opts, async () => {
  await loadPage();
  await startGame({ color: 'white' });
  let s = await page.eval(SNAP);
  assert.equal(s.orientation, 'white');
  assert.match(s.top, /the fly \(Fly\)/); assert.match(s.bottom, /Tester · to move/);
  await page.click('#btn-flip');
  s = await page.eval(SNAP);
  assert.equal(s.orientation, 'black');
  assert.match(s.top, /Tester · to move/, 'human label follows the pieces to the top');
  assert.match(s.bottom, /the fly \(Fly\)/);
  await page.key('f');                                                   // back to white
  assert.equal((await page.eval(SNAP)).orientation, 'white');
  await page.key('f', { modifiers: 2 });                                 // Ctrl+F is the browser's find
  assert.equal((await page.eval(SNAP)).orientation, 'white');
  await page.click('#btn-about');
  assert.equal(await page.eval(`document.getElementById('about-modal').open`), true);
  await page.key('f');
  assert.equal((await page.eval(SNAP)).orientation, 'white', 'f behind an open dialog must not flip');
  await page.eval(`document.getElementById('about-modal').close(), true`);
});

test('leaving via the landing page ends a party; a later solo game does not overwrite party results', opts, async () => {
  await page.eval(`document.getElementById('party-names').value = 'Alice\\nBob'; window.flychess.startParty(); true`);
  await page.waitFor(HUMAN_TURN);
  await page.click('#btn-resign');
  await page.waitFor(`document.getElementById('result-modal').open`);
  await page.click('#btn-next-player');
  await page.waitFor(HUMAN_TURN);
  await page.click('#btn-resign');
  await page.waitFor(`document.getElementById('result-modal').open`);
  let s = await page.eval(SNAP);
  assert.deepEqual(s.party.results, ['Alice', 'Bob']);
  await page.eval(`document.getElementById('result-modal').close(), true`);
  await page.click('#brand');
  s = await page.eval(SNAP);
  assert.equal(s.party, null, 'brand click ends the party'); assert.equal(s.partyBar, false);
  await startGame({ name: 'Carol' });
  s = await page.eval(SNAP);
  assert.equal(s.party, null); assert.equal(s.partyBar, false);
  await page.click('#btn-resign');
  await page.waitFor(`document.getElementById('result-modal').open`);
  assert.equal((await page.eval(SNAP)).party, null, 'finish() must not touch a party that has ended');
  assert.equal(await page.eval(`document.getElementById('btn-rematch').textContent`), 'Play again');
  await page.eval(`document.getElementById('result-modal').close(), true`);
});

test('a rejected fly move shows a persistent Retry instead of a stuck "thinking" board', opts, async () => {
  await startGame();
  await page.eval(`const a = window.flychess; a._realMove = a.brain.move; a.brain.move = () => Promise.reject(new Error('boom')); true`);
  await humanMove('e2', 'e4');
  await page.waitFor(`!document.getElementById('btn-retry').hidden`, { what: 'retry button' });
  let s = await page.eval(SNAP);
  assert.equal(s.status, 'The fly brain crashed.'); assert.equal(s.error, true);
  assert.equal(s.thinking, false); assert.equal(s.faded, false); assert.equal(s.mood, 'nervous');
  assert.match(s.commentary, /boom/);
  assert.equal(s.movable, null, 'it is still the fly\'s turn'); assert.equal(s.undo, true);
  assert.equal(await page.eval(`document.getElementById('toast').hidden`), true, 'no throwaway toast');
  await page.eval(`window.flychess.brain.move = window.flychess._realMove; true`);
  await page.click('#btn-retry');
  await page.waitFor(HUMAN_TURN, { what: 'fly reply after retry' });
  s = await page.eval(SNAP);
  assert.equal(s.hist.length, 2); assert.equal(s.retry, false); assert.equal(s.error, false); assert.equal(s.status, 'Your move.');
  assert.match(s.sims, /^1-ply × [1-3]$/, 'Fly level readout is a 1-ply check over N candidates');
  assert.equal(await page.eval(`document.querySelector('.clock').textContent.includes('search')`), true);
});

test('a dead worker is restarted by Retry and the game resumes', opts, async () => {
  await startGame();
  await humanMove('e2', 'e4');
  // kill the worker mid-move: terminate() fires no onerror, so simulate the crash report too
  await page.eval(`const b = window.flychess.brain; b.worker.terminate(); b.worker.onerror({ message: 'worker crashed' }); true`);
  await page.waitFor(`!document.getElementById('btn-retry').hidden`, { what: 'retry button' });
  let s = await page.eval(SNAP);
  assert.equal(s.dead, true); assert.equal(s.thinking, false); assert.equal(s.status, 'The fly brain crashed.');
  await page.click('#btn-retry');
  await page.waitFor(HUMAN_TURN, { what: 'fly reply after worker restart', timeout: 30000 });
  s = await page.eval(SNAP);
  assert.equal(s.dead, false); assert.equal(s.hist.length, 2); assert.equal(s.retry, false);
  assert.equal(s.hist[0], 'e4');
  await humanMove('d2', 'd4');                                           // the restarted worker keeps working
  await page.waitFor(HUMAN_TURN);
  assert.equal((await page.eval(SNAP)).hist.length, 4);
});

test('a failed download shows a visitor message with a working Retry', opts, async () => {
  serve.model = false;
  await page.goto(base);
  await page.waitFor(`!document.getElementById('load-error').hidden`, { what: 'load error' });
  const err = await page.eval(`document.getElementById('load-error').textContent`);
  assert.match(err, /Could not download the fly brain/);
  assert.match(err, /fly export-web/, 'the export hint is shown on localhost');
  assert.equal(await page.eval(`document.getElementById('loading').hidden`), true, 'progress bar hidden on failure');
  assert.equal(await page.eval(`document.getElementById('btn-start').disabled`), true);
  serve.model = true;
  await page.click('#btn-retry-load');
  await page.waitFor(`!document.getElementById('btn-start').disabled`, { what: 'brain load after retry', timeout: 30000 });
  assert.equal(await page.eval(`document.getElementById('load-error').hidden`), true);
  assert.equal(await page.eval(`document.getElementById('loading').hidden`), true);
  await startGame();                                                     // the restarted worker plays
  await humanMove('e2', 'e4');
  await page.waitFor(HUMAN_TURN);
  assert.equal((await page.eval(SNAP)).hist.length, 2);
  assert.deepEqual(page.errors, [], 'no uncaught page errors');
});
