// Synthetic `fly export-web` output for browser-level tests (same layout as SPEC §8, tiny sizes).
// Not a real fly: random CSR wiring and heads, just enough for worker.js to load and pick moves.
import { createHash } from 'node:crypto';
import { floatToHalf } from '../../engine/loader.js';

function rng(seed) { let x = seed >>> 0 || 1; return () => { x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; return x / 4294967296; }; }

/** @returns {{ header: object, blob: Buffer }} brain.json + brain.flyb bytes */
export function synthModel({ n = 64, nnz = 400, nIn = 6, nOut = 7, numMoves = 4168, inputDim = 1280, valueHidden = 4, steps = 3, seed = 1 } = {}) {
  const r = rng(seed);
  const rows = Array.from({ length: n }, () => new Set());
  let placed = 0;
  while (placed < nnz) {
    const post = Math.floor(r() * n), pre = Math.floor(r() * n);
    if (post === pre || rows[post].has(pre)) continue;
    rows[post].add(pre); placed++;
  }
  const indptr = new Int32Array(n + 1), indices = new Int32Array(nnz), w = new Float32Array(nnz);
  let e = 0;
  for (let i = 0; i < n; i++) {
    for (const c of [...rows[i]].sort((a, b) => a - b)) { indices[e] = c; w[e] = (r() - 0.4) * 0.6; e++; }
    indptr[i + 1] = e;
  }
  const perm = Array.from({ length: n }, (_, i) => i).sort(() => r() - 0.5);
  const inputIdx = Int32Array.from(perm.slice(0, nIn)).sort(), outputIdx = Int32Array.from(perm.slice(nIn, nIn + nOut)).sort();
  const f32 = (len, scale = 0.5) => Float32Array.from({ length: len }, () => (r() - 0.5) * scale);
  const arrays = [
    ['csr_indptr', 'i32', [n + 1], indptr], ['csr_indices', 'i32', [nnz], indices], ['w', 'f16', [nnz], w],
    ['bias', 'f32', [n], f32(n, 0.2)], ['alpha', 'f32', [n], Float32Array.from({ length: n }, () => 0.2 + 0.7 * r())],
    ['input_idx', 'i32', [nIn], inputIdx], ['output_idx', 'i32', [nOut], outputIdx],
    ['w_in', 'f16', [nIn, inputDim], f32(nIn * inputDim, 0.1)], ['b_in', 'f32', [nIn], f32(nIn)],
    ['policy_w', 'f16', [numMoves, nOut], f32(numMoves * nOut, 1)], ['policy_b', 'f32', [numMoves], f32(numMoves, 0.1)],
    ['value_w', 'f16', [valueHidden, nOut], f32(valueHidden * nOut, 1)], ['value_b', 'f32', [valueHidden], f32(valueHidden)],
    ['value_w2', 'f16', [1, valueHidden], f32(valueHidden, 1)], ['value_b2', 'f32', [1], f32(1)],
    ['positions', 'f16', [n, 3], Float32Array.from({ length: n * 3 }, () => r())],
    ['super_class', 'u8', [n], Uint8Array.from({ length: n }, () => Math.floor(r() * 4))],
  ];
  const specs = [], parts = [];
  let offset = 0;
  for (const [name, dtype, shape, data] of arrays) {
    let bytes;
    if (dtype === 'i32') bytes = new Uint8Array(Int32Array.from(data).buffer);
    else if (dtype === 'f32') bytes = new Uint8Array(Float32Array.from(data).buffer);
    else if (dtype === 'u8') bytes = Uint8Array.from(data);
    else { const u = new Uint16Array(data.length); for (let i = 0; i < data.length; i++) u[i] = floatToHalf(data[i]); bytes = new Uint8Array(u.buffer); }
    specs.push({ name, dtype, shape, offset, length_bytes: bytes.byteLength });
    parts.push([offset, bytes]);
    offset += Math.ceil(bytes.byteLength / 8) * 8;
  }
  const blob = Buffer.alloc(offset);
  for (const [off, bytes] of parts) blob.set(bytes, off);
  const header = {
    format: 'flyb', version: 1, run_name: 'synth', exported_at: '2026-01-01T00:00:00+00:00',
    n, nnz, steps, activation: 'relu', input_dim: inputDim, num_moves: numMoves, num_planes: 20,
    total_bytes: blob.byteLength, blob_sha256: createHash('sha256').update(blob).digest('hex'),
    super_class_legend: ['optic', 'central', 'sensory', 'descending'],
    arrays: specs,
  };
  return { header, blob };
}
