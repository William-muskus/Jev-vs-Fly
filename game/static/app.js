// Jev vs Fly match UI. The fly brain runs in a worker (reused from web/engine);
// Jev's move is fetched from /api/jev-move so the TypeSafe key stays on the server.

import { Chess } from '/vendor/chess.js';
import { Board } from '/board.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const REQUEST_TIMEOUT_MS = 90_000;

class Brain {
  constructor() {
    this.pending = new Map();
    this.nextId = 1;
    this.ready = null;
    this.info = null;
    this.onProgress = () => {};
    this._spawn();
  }
  _spawn() {
    this.worker = new Worker('/engine/worker.js', { type: 'module' });
    this.worker.onmessage = (ev) => this._onMessage(ev.data);
    this.worker.onerror = (ev) => this._fail(new Error(ev.message || 'worker crashed'));
  }
  load(baseUrl = '/model/') {
    this.ready = new Promise((resolve, reject) => { this._resolveReady = resolve; this._rejectReady = reject; });
    this.worker.postMessage({ type: 'load', baseUrl });
    return this.ready;
  }
  _fail(err) {
    this._rejectReady?.(err);
    for (const [, p] of this.pending) { clearTimeout(p.timer); p.reject(err); }
    this.pending.clear();
  }
  _settle(id, fn) {
    const p = this.pending.get(id);
    if (!p) return;
    this.pending.delete(id); clearTimeout(p.timer); fn(p);
  }
  _onMessage(msg) {
    if (msg.type === 'progress') { this.onProgress(msg); return; }
    if (msg.type === 'ready') { this.info = msg; this._resolveReady?.(msg); return; }
    if (msg.type === 'thinking') return;
    if (msg.type === 'error') {
      if (msg.id && this.pending.has(msg.id)) this._settle(msg.id, (p) => p.reject(new Error(msg.message)));
      else this._fail(new Error(msg.message));
      return;
    }
    this._settle(msg.id, (p) => p.resolve(msg));
  }
  _request(payload) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this._settle(id, () => reject(new Error('fly brain timed out')));
      }, REQUEST_TIMEOUT_MS);
      this.pending.set(id, { resolve, reject, timer });
      this.worker.postMessage({ ...payload, id });
    });
  }
  move(fen, moves, difficulty) {
    return this._request({ type: 'move', fen, moves, difficulty, trace: false });
  }
}

const brain = new Brain();
const chess = new Chess();
const board = new Board($('board'), { orientation: 'white' });
board.setMovable(null);

let running = false;
let jevIsWhite = true;
const uciHistory = () => chess.history({ verbose: true }).map((m) => m.from + m.to + (m.promotion || ''));

function fmtBytes(n) {
  if (!n) return '0 B';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function setProbs(el, items) {
  el.innerHTML = '';
  for (const it of items.slice(0, 5)) {
    const li = document.createElement('li');
    const p = typeof it.p === 'number' ? it.p : (it[1] ?? 0);
    const label = it.san || it.uci || it[0];
    li.innerHTML = `<span>${label}</span><span>${(p * 100).toFixed(1)}%</span>`;
    el.appendChild(li);
  }
}

function outcomeText() {
  if (chess.isCheckmate()) return `Checkmate — ${chess.turn() === 'w' ? 'Black' : 'White'} wins.`;
  if (chess.isStalemate()) return 'Stalemate.';
  if (chess.isThreefoldRepetition()) return 'Draw by repetition.';
  if (chess.isInsufficientMaterial()) return 'Draw by insufficient material.';
  if (chess.isDraw()) return 'Draw.';
  return null;
}

function syncBoard() {
  board.setPosition(chess.fen());
  const hist = chess.history({ verbose: true });
  const last = hist[hist.length - 1];
  board.highlight({
    lastMove: last ? [last.from, last.to] : null,
    check: chess.inCheck() ? chess.fen().split(' ')[0].includes('k') && chess.turn() === 'b' ? 'e8'
      : chess.turn() === 'w' ? squareOfKing('w') : squareOfKing('b') : null,
  });
  if (chess.inCheck()) board.highlight({ lastMove: last ? [last.from, last.to] : null, check: squareOfKing(chess.turn()) });
}

function squareOfKing(color) {
  const b = chess.board();
  for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
    const p = b[r][f];
    if (p && p.type === 'k' && p.color === color) return 'abcdefgh'[f] + (8 - r);
  }
  return null;
}

function pushMoveList(san) {
  const ol = $('moves');
  const n = chess.history().length;
  if (n % 2 === 1) {
    const li = document.createElement('li');
    li.textContent = san;
    ol.appendChild(li);
  } else {
    const li = ol.lastElementChild;
    if (li) li.textContent += '  ' + san;
  }
}

function whoToMove() {
  const jevTurn = (chess.turn() === 'w') === jevIsWhite;
  $('turn').textContent = jevTurn ? 'Jev to move' : 'Fly to move';
  return jevTurn;
}

async function jevMove() {
  $('jev-status').textContent = 'asking TypeSafe…';
  $('jev-status').classList.add('thinking');
  const t0 = performance.now();
  const res = await fetch('/api/jev-move', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ fen: chess.fen(), strategy: $('strategy').value }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || data.message || 'Jev failed');
  const applied = chess.move({ from: data.uci.slice(0, 2), to: data.uci.slice(2, 4), promotion: data.uci[4] });
  if (!applied) throw new Error(`illegal Jev move ${data.uci}`);
  $('jev-status').classList.remove('thinking');
  $('jev-status').textContent = `plays ${data.san}  (${data.played_question})`;
  const ans = data.answers?.[data.played_question];
  if (ans?.top) setProbs($('jev-probs'), ans.top.map(([uci, p]) => ({ uci, p })));
  const ms = Math.round(performance.now() - t0);
  const tok = (data.usage?.input_tokens || 0) + (data.usage?.output_tokens || 0);
  $('jev-meta').textContent = `${ms} ms · ${tok} tokens · ${data.legal_count} legal`;
  return data.san;
}

async function flyMove() {
  $('fly-status').textContent = 'the connectome is thinking…';
  $('fly-status').classList.add('thinking');
  const msg = await brain.move(chess.fen(), uciHistory(), $('difficulty').value);
  if (!msg.move) throw new Error('fly returned no move');
  const applied = chess.move({ from: msg.move.slice(0, 2), to: msg.move.slice(2, 4), promotion: msg.move[4] });
  if (!applied) throw new Error(`illegal fly move ${msg.move}`);
  $('fly-status').classList.remove('thinking');
  $('fly-status').textContent = `plays ${msg.san}`;
  if (msg.policyTop) setProbs($('fly-probs'), msg.policyTop);
  $('fly-meta').textContent = `${Math.round(msg.thinkMs || 0)} ms · ${msg.backend || 'js'} · value ${(msg.value ?? 0).toFixed(2)}`;
  return msg.san;
}

async function playLoop() {
  running = true;
  $('result').hidden = true;
  while (running) {
    const over = outcomeText();
    if (over) {
      $('result').hidden = false;
      $('result').textContent = over;
      $('turn').textContent = over;
      running = false;
      break;
    }
    const maxPlies = Number(params.get('maxPlies') || 80);
    if (chess.history().length >= maxPlies) {
      $('result').hidden = false;
      $('result').textContent = `Stopped after ${maxPlies} plies.`;
      running = false;
      break;
    }
    whoToMove();
    const jevTurn = (chess.turn() === 'w') === jevIsWhite;
    try {
      const san = jevTurn ? await jevMove() : await flyMove();
      pushMoveList(san);
      syncBoard();
    } catch (err) {
      running = false;
      $('result').hidden = false;
      $('result').textContent = String(err.message || err);
      $('result').style.color = 'var(--red)';
      break;
    }
  }
}

function resetGame() {
  running = false;
  chess.reset();
  $('moves').innerHTML = '';
  $('jev-probs').innerHTML = '';
  $('fly-probs').innerHTML = '';
  $('jev-meta').textContent = '';
  $('fly-meta').textContent = '';
  $('result').hidden = true;
  $('result').style.color = '';
  syncBoard();
}

async function start() {
  jevIsWhite = $('jev-color').value === 'white';
  board.orientation = jevIsWhite ? 'white' : 'black';
  $('jev-side').textContent = jevIsWhite ? 'White' : 'Black';
  $('fly-side').textContent = jevIsWhite ? 'Black' : 'White';
  $('setup').querySelectorAll('select').forEach((el) => { el.disabled = true; });
  $('btn-start').disabled = true;
  $('arena').hidden = false;
  $('moves-wrap').hidden = false;
  resetGame();
  await playLoop();
  $('btn-start').disabled = false;
  $('setup').querySelectorAll('select').forEach((el) => { el.disabled = false; });
}

$('btn-start').addEventListener('click', start);
$('btn-new').addEventListener('click', start);
$('btn-pgn').addEventListener('click', async () => {
  await navigator.clipboard.writeText(chess.pgn());
  $('btn-pgn').textContent = 'Copied';
  setTimeout(() => { $('btn-pgn').textContent = 'Copy PGN'; }, 1200);
});

brain.onProgress = (msg) => {
  const frac = msg.total ? msg.loaded / msg.total : 0;
  $('bar-fill').style.width = `${Math.round(frac * 100)}%`;
  $('load-msg').textContent = `${msg.phase || 'loading'} · ${fmtBytes(msg.loaded)} / ${fmtBytes(msg.total)}`;
};

brain.load('/model/').then((info) => {
  $('btn-start').disabled = false;
  $('btn-start').textContent = 'Play Jev vs Fly';
  $('bar-fill').style.width = '100%';
  const n = info.header?.n || info.header?.neurons || '—';
  $('load-msg').textContent = `fly ready · ${n} neurons · ${info.backend || 'js'}`;
  if (params.get('autoplay') === '1' || params.get('autoplay') === 'true') {
    if (params.get('strategy')) $('strategy').value = params.get('strategy');
    if (params.get('difficulty')) $('difficulty').value = params.get('difficulty');
    if (params.get('jevColor')) $('jev-color').value = params.get('jevColor');
    start();
  }
}).catch((err) => {
  $('load-error').hidden = false;
  $('load-error').textContent = err.message || String(err);
});
