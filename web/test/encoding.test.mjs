// node --test web/test/  -- checks web/engine/encoding.js against tests/vectors/encoding.json
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Chess } from '../vendor/chess.js';
import {
  FLAT_INPUT, NUM_MOVES, encodeBoard, moveToIndex, indexToMove, legalMoveIndices, legalMoveMask,
  uciToMove, positionRepeated,
} from '../engine/encoding.js';

const here = dirname(fileURLToPath(import.meta.url));
const vectors = JSON.parse(readFileSync(join(here, '..', '..', 'tests', 'vectors', 'encoding.json'), 'utf8'));

function buildPosition(vec) {
  const chess = new Chess();
  if (vec.move_list) {
    for (const uci of vec.move_list) chess.move(uciToMove(uci));
    assert.equal(chess.fen(), vec.fen);
  } else {
    chess.load(vec.fen);
  }
  return chess;
}

function sha256LE(planes) {
  const buf = Buffer.alloc(planes.length * 4);
  for (let i = 0; i < planes.length; i++) buf.writeFloatLE(planes[i], i * 4);
  return createHash('sha256').update(buf).digest('hex');
}

test('vectors file is non-trivial', () => {
  assert.ok(vectors.length >= 30);
  assert.ok(vectors.some((v) => v.move_list));
  assert.ok(vectors.some((v) => v.repetition_plane === 1));
  assert.ok(vectors.some((v) => v.legal_indices.some((i) => i >= 4096)));
});

for (const [i, vec] of vectors.entries()) {
  const label = `#${i} ${vec.move_list ? vec.move_list.join(' ') : vec.fen}`;

  test(`planes ${label}`, () => {
    const chess = buildPosition(vec);
    const planes = encodeBoard(chess);
    assert.equal(planes.length, FLAT_INPUT);
    // sparse planes 0-17
    const expected = new Float32Array(FLAT_INPUT);
    for (const [p, r, f, v] of vec.planes_nonzero) expected[p * 64 + r * 8 + f] = v;
    expected.fill(Math.fround(vec.halfmove_plane), 18 * 64, 19 * 64);
    expected.fill(vec.repetition_plane, 19 * 64, 20 * 64);
    assert.deepEqual(Array.from(planes), Array.from(expected));
    assert.equal(positionRepeated(chess), vec.repetition_plane === 1);
    assert.equal(sha256LE(planes), vec.planes_sha256);
  });

  test(`moves ${label}`, () => {
    const chess = buildPosition(vec);
    assert.deepEqual(legalMoveIndices(chess), vec.legal_indices);
    const mask = legalMoveMask(chess);
    assert.equal(mask.length, NUM_MOVES);
    assert.equal(mask.reduce((a, b) => a + b, 0), vec.legal_indices.length);
    const seen = new Map();
    for (const move of chess.moves({ verbose: true })) {
      const uci = move.from + move.to + (move.promotion || '');
      const idx = moveToIndex(move, chess);
      assert.equal(idx, vec.moves[uci], `index of ${uci}`);
      assert.equal(moveToIndex(uci, chess), idx);
      assert.ok(!seen.has(idx), `duplicate index ${idx}`);
      seen.set(idx, uci);
      const back = indexToMove(idx, chess);
      assert.equal(back.uci, uci, `round trip of ${uci}`);
      assert.equal(chess.move(back).lan.length >= 4, true); // chess.js accepts the decoded move
      chess.undo();
    }
    assert.equal(seen.size, Object.keys(vec.moves).length);
  });
}
