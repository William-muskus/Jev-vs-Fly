// flybrain-gpu.js — the fly brain forward pass on WebGPU (same math as flybrain.js / SPEC §4).
//
//   base = bias + inj                inj[input_idx[k]] = w_in[k]·x + b_in[k]       (inject shader, once per forward)
//   h_0 = 0 ; for t < steps:  h_{t+1}[i] = (1-α[i])·h_t[i] + α[i]·act(base[i] + Σ_e w[e]·h_t[col[e]])   (step shader)
//   policy = policy_w · h_T[output_idx] + policy_b                                  (matvec shader)
//   value  = tanh(value_w2 · act_v(value_w · h_T[output_idx] + value_b) + value_b2)  (matvec shader + 256-term dot on the CPU)
//
// All recurrent arithmetic is f32 on the GPU; one command buffer per forward encodes the injection,
// all `steps` recurrent updates (ping-pong hidden-state buffers) and both heads, then a single
// mapAsync reads back the policy logits, the value-head hidden layer and (optionally) the whole
// final state. The blob's neuron order (header.neuron_order = 'rcm') is used as-is, like FlyBrain.
//
// Row lengths of the connectome CSR span 0…5000+, so a thread-per-row SpMV is latency-bound on the
// few long rows. Rows are therefore bucketed once by length (LANE_BUCKETS) and each timestep runs
// one dispatch per bucket: 1 lane per short row, 8 lanes per medium row, 64 lanes (a whole
// workgroup) per long row, partial sums combined through workgroup memory. The buckets partition
// the rows, so the three dispatches write disjoint entries of h_out. The input projection
// (2048 × 1280) and the heads use the same 64-lane matvec kernel.
//
// `emulate*` are pure-JS, f32-rounded (Math.fround) mirrors of the WGSL, line by line, so the shader
// math is unit-testable under node (no WebGPU there); the browser test compares the real device
// against FlyBrain on the exported model. The only intended deviation between the emulation and the
// GPU is the compiler's freedom to fuse `s + w*h` into an FMA (one rounding instead of two).

const WG = 64;                // workgroup size of every kernel
const REDUCE_LANES = 64;      // lanes per matvec row (== WG)
const NONE = 0xffffffff;      // "not an input neuron" marker in the inverse input map
/** SpMV row buckets: rows with length <= maxLen (and above the previous bucket) get `lanes` threads each. */
export const LANE_BUCKETS = [{ lanes: 1, maxLen: 32 }, { lanes: 8, maxLen: 512 }, { lanes: 64, maxLen: Infinity }];

/** Lanes used for a row of `len` synapses (shared by the WGSL generator, the dispatcher and the emulation). */
export function lanesForRow(len) {
  for (const b of LANE_BUCKETS) if (len <= b.maxLen) return b.lanes;
  return LANE_BUCKETS[LANE_BUCKETS.length - 1].lanes;
}

/** Partition rows 0..n-1 into the lane buckets (ascending row ids inside each bucket). */
export function bucketRows(indptr, n) {
  const lists = LANE_BUCKETS.map(() => []);
  for (let i = 0; i < n; i++) {
    const len = indptr[i + 1] - indptr[i];
    lists[LANE_BUCKETS.findIndex((b) => len <= b.maxLen)].push(i);
  }
  return lists.map((rows, k) => ({ lanes: LANE_BUCKETS[k].lanes, rows: Uint32Array.from(rows) }));
}

// ---------------------------------------------------------------------------------------------
// activation: WGSL source and the f32 JS mirror

/** Resolve the activation kind of a header ('relu'|'tanh'|'gelu'|'gelu_tanh'|'satrelu'). */
export function activationKind(header, name = header.activation ?? 'relu') {
  if (name === 'gelu' && /^tanh$/i.test(String(header.gelu_approximate || ''))) return 'gelu_tanh';
  if (!['relu', 'tanh', 'gelu', 'gelu_tanh', 'satrelu'].includes(name)) throw new Error(`unknown activation ${name}`);
  return name;
}

/** WGSL helpers shared by every kernel: tanh via exp (no reliance on backend tanh polyfills), erf (A&S 7.1.26). */
const WGSL_MATH = `
fn tanh_(x: f32) -> f32 {
  let a = min(abs(x), 20.0);
  let e = exp(2.0 * a);
  let t = 1.0 - 2.0 / (e + 1.0);
  return select(-t, t, x >= 0.0);
}
fn erf_(x: f32) -> f32 {
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
  return select(-y, y, x >= 0.0);
}
`;

/** WGSL body of `fn NAME(v: f32) -> f32` for an activation kind. */
export function activationWGSL(kind, sat, fnName = 'act') {
  const body = {
    relu: 'return max(v, 0.0);',
    tanh: 'return tanh_(v);',
    gelu: 'return 0.5 * v * (1.0 + erf_(v * 0.7071067811865476));',
    gelu_tanh: 'return 0.5 * v * (1.0 + tanh_(0.7978845608028654 * (v + 0.044715 * v * v * v)));',
    satrelu: `return select(0.0, ${f32lit(sat)} * tanh_(v / ${f32lit(sat)}), v > 0.0);`,
    identity: 'return v;',
  }[kind];
  if (!body) throw new Error(`unknown activation ${kind}`);
  return `fn ${fnName}(v: f32) -> f32 { ${body} }\n`;
}

function f32lit(v) {
  const s = String(Number(v));
  return /[.e]/i.test(s) ? s : s + '.0';
}

const f = Math.fround;
function tanhF32(x) {
  const a = Math.min(Math.abs(x), 20);
  const e = f(Math.exp(f(2 * a)));
  const t = f(1 - f(2 / f(e + 1)));
  return x >= 0 ? t : -t;
}
function erfF32(x) {
  const ax = Math.abs(x);
  const t = f(1 / f(1 + f(0.3275911 * ax)));
  const p = f(f(f(f(f(f(f(f(1.061405429 * t) - 1.453152027) * t) + 1.421413741) * t) - 0.284496736) * t) + 0.254829592);
  const y = f(1 - f(f(p * t) * f(Math.exp(f(-ax * ax)))));
  return x >= 0 ? y : -y;
}

/** f32 JS mirror of activationWGSL(kind). */
export function activationF32(kind, sat = 10) {
  sat = f(sat);
  switch (kind) {
    case 'relu': return (v) => (v > 0 ? v : 0);
    case 'tanh': return (v) => tanhF32(v);
    case 'gelu': return (v) => f(f(0.5 * v) * f(1 + erfF32(f(v * 0.7071067811865476))));
    case 'gelu_tanh': return (v) => f(f(0.5 * v) * f(1 + tanhF32(f(0.7978845608028654 * f(v + f(f(f(0.044715 * v) * v) * v))))));
    case 'satrelu': return (v) => (v > 0 ? f(sat * tanhF32(f(v / sat))) : 0);
    case 'identity': return (v) => v;
    default: throw new Error(`unknown activation ${kind}`);
  }
}

// ---------------------------------------------------------------------------------------------
// shaders

/** Inverse of input_idx: invIn[neuron] = k or NONE. Throws on duplicate input neurons (FlyBrain would sum them). */
export function inverseInputMap(inputIdx, n) {
  const inv = new Uint32Array(n).fill(NONE);
  for (let k = 0; k < inputIdx.length; k++) {
    const i = inputIdx[k];
    if (i < 0 || i >= n) throw new Error(`input_idx[${k}] = ${i} out of range`);
    if (inv[i] !== NONE) throw new Error(`input_idx has duplicate neuron ${i}`);
    inv[i] = k;
  }
  return inv;
}

/** Row index of an invocation for a (possibly 2-D) dispatch — same formula as the WGSL. */
function dispatchDims(rows, maxPerDim) {
  const groups = Math.ceil(rows / WG);
  const gx = Math.min(groups, maxPerDim);
  return [gx, Math.ceil(groups / gx)];
}

/** base[i] = bias[i] + inj[inv_in[i]] (inj = w_in·x + b_in from the matvec kernel). */
function baseWGSL(n) {
  return `
@group(0) @binding(0) var<storage, read> bias: array<f32>;
@group(0) @binding(1) var<storage, read> inv_in: array<u32>;
@group(0) @binding(2) var<storage, read> inj: array<f32>;
@group(0) @binding(3) var<storage, read_write> base: array<f32>;
@compute @workgroup_size(${WG})
fn base_(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = gid.y * (nw.x * ${WG}u) + gid.x;
  if (i >= ${n}u) { return; }
  var s = bias[i];
  let k = inv_in[i];
  if (k != ${NONE}u) { s = s + inj[k]; }
  base[i] = s;
}
`;
}

/**
 * One recurrent update for the rows listed in `rows` (a lane bucket): LANES threads per row stride
 * over its synapses, lane partials are tree-reduced in workgroup memory (LANES == 1: plain loop).
 */
function stepWGSL(count, lanes, kind, sat) {
  const R = WG / lanes;   // rows per workgroup
  const reduce = lanes === 1 ? '' : `
  part[lid.x] = s;
  workgroupBarrier();
  for (var stride = ${lanes / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) { part[lid.x] = part[lid.x] + part[lid.x + stride]; }
    workgroupBarrier();
  }
  s = part[lid.x];`;
  return WGSL_MATH + activationWGSL(kind, sat, 'act') + `
@group(0) @binding(0) var<storage, read> indptr: array<u32>;
@group(0) @binding(1) var<storage, read> col: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> base: array<f32>;
@group(0) @binding(4) var<storage, read> alpha: array<f32>;
@group(0) @binding(5) var<storage, read> h_in: array<f32>;
@group(0) @binding(6) var<storage, read_write> h_out: array<f32>;
@group(0) @binding(7) var<storage, read> rows: array<u32>;
${lanes === 1 ? '' : `var<workgroup> part: array<f32, ${WG}>;`}
@compute @workgroup_size(${WG})
fn step(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let slot = (wg.y * nw.x + wg.x) * ${R}u + lid.x / ${lanes}u;
  let lane = lid.x % ${lanes}u;
  let valid = slot < ${count}u;
  var i = 0u;
  var start = 0u;
  var end = 0u;
  var s = 0.0;
  if (valid) { i = rows[slot]; start = indptr[i]; end = indptr[i + 1u]; s = select(0.0, base[i], lane == 0u); }
  for (var e = start + lane; e < end; e = e + ${lanes}u) { s = s + w[e] * h_in[col[e]]; }${reduce}
  if (valid && lane == 0u) {
    let a = alpha[i];
    h_out[i] = (1.0 - a) * h_in[i] + a * act(s);
  }
}
`;
}

/** y[m] = ACT(b[m] + Σ_j W[m, j] · h[out_idx[j]]) — one workgroup per row, 64-lane strided partial sums + tree reduction. */
function matvecWGSL(nOut, kind, sat) {
  return WGSL_MATH + activationWGSL(kind, sat, 'act') + `
@group(0) @binding(0) var<storage, read> h: array<f32>;
@group(0) @binding(1) var<storage, read> out_idx: array<u32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> part: array<f32, ${REDUCE_LANES}>;
@compute @workgroup_size(${REDUCE_LANES})
fn matvec(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let m = wg.y * nw.x + wg.x;
  let off = m * ${nOut}u;
  var s = 0.0;
  for (var j = lid.x; j < ${nOut}u; j = j + ${REDUCE_LANES}u) { s = s + W[off + j] * h[out_idx[j]]; }
  part[lid.x] = s;
  workgroupBarrier();
  for (var stride = ${REDUCE_LANES / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lid.x < stride) { part[lid.x] = part[lid.x] + part[lid.x + stride]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { y[m] = act(part[0] + b[m]); }
}
`;
}

// ---------------------------------------------------------------------------------------------
// pure-JS f32 emulation of the kernels (for tests; mirrors the WGSL above line by line)

/** Model constants shared by the GPU engine and its emulation. */
export function modelShape({ header, arrays }) {
  const n = header.n ?? arrays.csr_indptr.length - 1;
  const D = header.input_dim ?? (header.num_planes ?? 20) * 64;
  const nOut = arrays.output_idx.length;
  const numMoves = header.num_moves ?? arrays.policy_b.length;
  const valueHidden = arrays.value_w2 ? arrays.value_w.length / nOut : 0;
  const kind = activationKind(header);
  const valueKind = activationKind(header, header.value_activation || header.activation || 'relu');
  const sat = Number(header.activation_sat ?? 10);
  const steps = header.steps ?? 8;
  let alpha = arrays.alpha;
  if (alpha.length !== n) alpha = new Float32Array(n).fill(alpha[0]);
  if (arrays.w.length !== arrays.csr_indices.length || arrays.csr_indptr.length !== n + 1) throw new Error('CSR shape mismatch');
  if (arrays.w_in.length !== arrays.input_idx.length * D) throw new Error('w_in shape mismatch');
  if (arrays.policy_w.length !== numMoves * nOut) throw new Error('policy_w shape mismatch');
  return { n, D, nOut, numMoves, valueHidden, kind, valueKind, sat, steps, alpha };
}

/** inject: inj = w_in·x + b_in through the matvec kernel, then base[i] = bias[i] + inj[inv_in[i]] (base kernel). */
export function emulateInject({ arrays }, invIn, x, base) {
  const n = base.length, D = x.length, nIn = arrays.b_in.length;
  const ident = Uint32Array.from({ length: D }, (_, j) => j);
  const inj = emulateMatvec(arrays.w_in, arrays.b_in, nIn, D, x, ident, (v) => v, new Float32Array(nIn));
  for (let i = 0; i < n; i++) {
    let s = arrays.bias[i];
    const k = invIn[i];
    if (k !== NONE) s = f(s + inj[k]);
    base[i] = s;
  }
  return base;
}

/** step kernels: hOut[i] = (1-α)·hIn[i] + α·act(base[i] + Σ w·hIn[col]) with each row's lane-strided partial sums and tree reduction. */
export function emulateStep({ arrays }, alpha, act, base, hIn, hOut) {
  const n = hOut.length, indptr = arrays.csr_indptr, col = arrays.csr_indices, w = arrays.w;
  const part = new Float32Array(WG);
  for (let i = 0; i < n; i++) {
    const start = indptr[i], end = indptr[i + 1];
    const lanes = lanesForRow(end - start);
    for (let lane = 0; lane < lanes; lane++) {
      let s = lane === 0 ? base[i] : 0;
      for (let e = start + lane; e < end; e += lanes) s = f(s + f(w[e] * hIn[col[e]]));
      part[lane] = s;
    }
    for (let stride = lanes >> 1; stride > 0; stride >>= 1) for (let l = 0; l < stride; l++) part[l] = f(part[l] + part[l + stride]);
    const a = alpha[i];
    hOut[i] = f(f(f(1 - a) * hIn[i]) + f(a * act(part[0])));
  }
  return hOut;
}

/** matvec kernel: y[m] = act(b[m] + Σ_j W[m,j]·h[outIdx[j]]) with the workgroup's summation order. */
export function emulateMatvec(W, b, rows, nOut, h, outIdx, act, y) {
  const part = new Float32Array(REDUCE_LANES);
  for (let m = 0; m < rows; m++) {
    const off = m * nOut;
    for (let l = 0; l < REDUCE_LANES; l++) {
      let s = 0;
      for (let j = l; j < nOut; j += REDUCE_LANES) s = f(s + f(W[off + j] * h[outIdx[j]]));
      part[l] = s;
    }
    for (let stride = REDUCE_LANES >> 1; stride > 0; stride >>= 1) for (let l = 0; l < stride; l++) part[l] = f(part[l] + part[l + stride]);
    y[m] = act(f(part[0] + b[m]));
  }
  return y;
}

/** Final scalar of the value head from the matvec output (MLP: 256-term dot; linear: y[0]); done on the CPU in both engines. */
export function valueFromHidden(arrays, valueHidden, y) {
  let v;
  if (valueHidden > 0) {
    v = arrays.value_b2[0];
    for (let k = 0; k < valueHidden; k++) v += arrays.value_w2[k] * y[k];
  } else v = y[0];
  return Math.tanh(v);
}

/** Whole forward pass through the emulated kernels: same outputs as FlyBrainGPU.forward. */
export function emulateForward(model, x) {
  const S = modelShape(model);
  const a = model.arrays;
  const invIn = inverseInputMap(a.input_idx, S.n);
  const base = emulateInject(model, invIn, x, new Float32Array(S.n));
  const act = activationF32(S.kind, S.sat);
  let h = new Float32Array(S.n), h2 = new Float32Array(S.n);
  for (let t = 0; t < S.steps; t++) { emulateStep(model, S.alpha, act, base, h, h2); const tmp = h; h = h2; h2 = tmp; }
  const policy = emulateMatvec(a.policy_w, a.policy_b, S.numMoves, S.nOut, h, a.output_idx, activationF32('identity'), new Float32Array(S.numMoves));
  const vRows = S.valueHidden || 1;
  const vAct = S.valueHidden ? activationF32(S.valueKind, S.sat) : activationF32('identity');
  const hid = emulateMatvec(a.value_w, a.value_b, vRows, S.nOut, h, a.output_idx, vAct, new Float32Array(vRows));
  return { policy, value: valueFromHidden(a, S.valueHidden, hid), activity: h };
}

// ---------------------------------------------------------------------------------------------
// the WebGPU engine

const TS_BYTES = 16;   // two u64 timestamps at the head of the readback buffer

export class FlyBrainGPU {
  /**
   * Build the engine on the default WebGPU adapter. Rejects when WebGPU is unavailable, the adapter
   * limits cannot hold the model, or the model uses a feature the kernels do not cover.
   * @param {{header: object, arrays: Record<string, Int32Array|Float32Array|Uint8Array>}} model  as returned by loadBrain / parseArrays
   * @param {{adapter?: GPUAdapter, device?: GPUDevice, powerPreference?: string}} [opts]
   */
  static async create(model, opts = {}) {
    const gpu = (typeof navigator !== 'undefined' && navigator.gpu) || null;
    if (!gpu && !opts.device) throw new Error('WebGPU is not available in this context');
    let device = opts.device || null, adapter = opts.adapter || null;
    if (!device) {
      // Chromium can answer null while its GPU process is still starting: retry a few times
      for (let attempt = 0; !adapter && attempt < 4; attempt++) {
        if (attempt) await new Promise((r) => setTimeout(r, 200 * attempt));
        adapter = await gpu.requestAdapter({ powerPreference: opts.powerPreference || 'high-performance' });
      }
      if (!adapter) throw new Error('no WebGPU adapter');
      const S = modelShape(model);
      const largest = 4 * Math.max(S.n + 1, model.arrays.csr_indices.length, model.arrays.w_in.length, model.arrays.policy_w.length);
      const lim = adapter.limits;
      if (largest > lim.maxStorageBufferBindingSize || largest > lim.maxBufferSize) {
        throw new Error(`model needs ${largest} B storage bindings, adapter allows ${Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize)}`);
      }
      const requiredLimits = {};
      // default binding limit is 128 MiB; ask for more only when the model needs it
      if (largest > 134217728) { requiredLimits.maxStorageBufferBindingSize = largest; requiredLimits.maxBufferSize = Math.max(largest, 268435456); }
      const requiredFeatures = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
      device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    }
    const brain = new FlyBrainGPU(model, device, adapter);
    try {
      await brain._build();
    } catch (err) {
      brain.destroy();
      throw err;
    }
    return brain;
  }

  /** @private use FlyBrainGPU.create */
  constructor(model, device, adapter) {
    const S = modelShape(model);
    Object.assign(this, S);
    this.header = model.header;
    this.arrays = model.arrays;
    this.device = device;
    this.adapter = adapter;
    this.positions = model.arrays.positions ?? null;
    this.superClass = model.arrays.super_class ?? null;
    this.inputDim = S.D;
    this.nnz = model.arrays.csr_indices.length;
    this.nIn = model.arrays.input_idx.length;
    this.activation = S.kind;
    this.lost = null;             // string once the device is gone / errored — forward() rejects afterwards
    this.lastStepMs = 0;          // GPU time per recurrent step (timestamp queries) or wall-clock estimate
    this.lastForwardMs = 0;
    this.timestamps = device.features.has('timestamp-query');
    this._chain = Promise.resolve();
    this._bufs = [];
    device.lost.then((info) => { this.lost = `device lost: ${info.message || info.reason}`; }).catch(() => {});
    device.addEventListener?.('uncapturederror', (ev) => { this.lost = `gpu error: ${ev.error?.message || ev.error}`; });
  }

  get backend() { return 'webgpu'; }

  _buffer(label, data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
    const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const size = Math.ceil(Math.max(bytes.byteLength, 4) / 4) * 4;
    const buf = this.device.createBuffer({ label, size, usage, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(bytes);
    buf.unmap();
    this._bufs.push(buf);
    return buf;
  }

  _empty(label, bytes, usage) {
    const buf = this.device.createBuffer({ label, size: Math.ceil(Math.max(bytes, 4) / 4) * 4, usage });
    this._bufs.push(buf);
    return buf;
  }

  async _build() {
    const dev = this.device, a = this.arrays, S = this;
    const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const invIn = inverseInputMap(a.input_idx, S.n);
    // parameters (uploaded once)
    const bIndptr = this._buffer('csr_indptr', a.csr_indptr);
    const bCol = this._buffer('csr_indices', a.csr_indices);
    const bW = this._buffer('w', a.w);
    const bBias = this._buffer('bias', a.bias);
    const bAlpha = this._buffer('alpha', S.alpha);
    const bInvIn = this._buffer('inv_in', invIn);
    const bWIn = this._buffer('w_in', a.w_in);
    const bBIn = this._buffer('b_in', a.b_in);
    const bIdent = this._buffer('identity_idx', Uint32Array.from({ length: S.D }, (_, j) => j));
    const bOutIdx = this._buffer('output_idx', a.output_idx);
    const bPolicyW = this._buffer('policy_w', a.policy_w);
    const bPolicyB = this._buffer('policy_b', a.policy_b);
    const bValueW = this._buffer('value_w', a.value_w);
    const bValueB = this._buffer('value_b', a.value_b);
    const buckets = bucketRows(a.csr_indptr, S.n).filter((b) => b.rows.length > 0);
    this.buckets = buckets.map((b) => ({ lanes: b.lanes, count: b.rows.length, buf: this._buffer(`rows_${b.lanes}`, b.rows) }));
    // per-forward state
    this.bX = this._empty('x', 4 * S.D, ST);
    const bInj = this._empty('inj', 4 * this.nIn, GPUBufferUsage.STORAGE);
    const bBase = this._empty('base', 4 * S.n, ST);
    const hA = this._empty('h_a', 4 * S.n, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const hB = this._empty('h_b', 4 * S.n, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.hA = hA;
    this.hFinal = S.steps % 2 === 0 ? hA : hB;
    this.vRows = S.valueHidden || 1;
    this.bPolicy = this._empty('policy', 4 * S.numMoves, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this.bValue = this._empty('value_hidden', 4 * this.vRows, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    // readback: [timestamps | policy | value hidden | h]
    this.offPolicy = TS_BYTES;
    this.offValue = this.offPolicy + 4 * S.numMoves;
    this.offH = this.offValue + 4 * this.vRows;
    this.readback = this._empty('readback', this.offH + 4 * S.n, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    if (this.timestamps) {
      this.querySet = dev.createQuerySet({ type: 'timestamp', count: 2 });
      this.bQuery = this._empty('timestamps', 256, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
    }

    // pipelines (compiled asynchronously; compilation errors surface here)
    const make = async (label, code, entry) => {
      const module = dev.createShaderModule({ label, code });
      const info = await module.getCompilationInfo?.();
      const errs = (info?.messages || []).filter((m) => m.type === 'error');
      if (errs.length) throw new Error(`${label} shader: ${errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}`);
      return dev.createComputePipelineAsync({ label, layout: 'auto', compute: { module, entryPoint: entry } });
    };
    const vKind = S.valueHidden ? S.valueKind : 'identity';
    const [pBase, pMatvecIn, pPolicy, pValue, ...pSteps] = await Promise.all([
      make('base', baseWGSL(S.n), 'base_'),
      make('inject', matvecWGSL(S.D, 'identity', S.sat), 'matvec'),
      make('policy', matvecWGSL(S.nOut, 'identity', S.sat), 'matvec'),
      make('value', matvecWGSL(S.nOut, vKind, S.sat), 'matvec'),
      ...this.buckets.map((b) => make(`step_${b.lanes}`, stepWGSL(b.count, b.lanes, S.kind, S.sat), 'step')),
    ]);
    this.pBase = pBase; this.pInject = pMatvecIn; this.pPolicy = pPolicy; this.pValue = pValue;
    const bg = (pipeline, buffers) => dev.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.bgInject = bg(pMatvecIn, [this.bX, bIdent, bWIn, bBIn, bInj]);
    this.bgBase = bg(pBase, [bBias, bInvIn, bInj, bBase]);
    const maxWG = dev.limits.maxComputeWorkgroupsPerDimension;
    const dims2 = (groups) => { const gx = Math.min(groups, maxWG); return [gx, Math.ceil(groups / gx)]; };
    this.buckets.forEach((b, k) => {
      b.pipeline = pSteps[k];
      b.bg = [
        bg(pSteps[k], [bIndptr, bCol, bW, bBase, bAlpha, hA, hB, b.buf]),   // even t: read A, write B
        bg(pSteps[k], [bIndptr, bCol, bW, bBase, bAlpha, hB, hA, b.buf]),   // odd t
      ];
      b.dims = dims2(Math.ceil(b.count / (WG / b.lanes)));
    });
    this.bgPolicy = bg(pPolicy, [this.hFinal, bOutIdx, bPolicyW, bPolicyB, this.bPolicy]);
    this.bgValue = bg(pValue, [this.hFinal, bOutIdx, bValueW, bValueB, this.bValue]);
    this.dimsN = dispatchDims(S.n, maxWG);
    this.dimsInject = dims2(this.nIn);
    this.dimsPolicy = dims2(S.numMoves);
    this.dimsValue = dims2(this.vRows);
    // run one forward so pipeline warm-up does not land on the first move
    await this._forward(new Float32Array(S.D), { activity: false });
  }

  /**
   * One forward pass (serialised: concurrent calls run one after another on the same buffers).
   * @param {Float32Array} x  flattened planes (input_dim)
   * @param {{activity?: boolean}} [opts]  activity: false skips the final-state readback (activity is then null)
   * @returns {Promise<{policy: Float32Array, value: number, activity: Float32Array|null}>}  fresh arrays each call
   */
  forward(x, opts = {}) {
    const run = this._chain.then(() => this._forward(x, opts));
    this._chain = run.catch(() => {});
    return run;
  }

  async _forward(x, { activity = true } = {}) {
    if (this.lost) throw new Error(this.lost);
    if (x.length !== this.D) throw new Error(`input must have ${this.D} values, got ${x.length}`);
    const dev = this.device;
    const t0 = performance.now();
    dev.queue.writeBuffer(this.bX, 0, x.buffer, x.byteOffset, x.byteLength);
    const enc = dev.createCommandEncoder();
    enc.clearBuffer(this.hA);                       // h_0 = 0
    const passDesc = this.timestamps ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {};
    const pass = enc.beginComputePass(passDesc);
    pass.setPipeline(this.pInject);
    pass.setBindGroup(0, this.bgInject);
    pass.dispatchWorkgroups(this.dimsInject[0], this.dimsInject[1]);
    pass.setPipeline(this.pBase);
    pass.setBindGroup(0, this.bgBase);
    pass.dispatchWorkgroups(this.dimsN[0], this.dimsN[1]);
    for (let t = 0; t < this.steps; t++) {
      for (const b of this.buckets) {
        pass.setPipeline(b.pipeline);
        pass.setBindGroup(0, b.bg[t & 1]);
        pass.dispatchWorkgroups(b.dims[0], b.dims[1]);
      }
    }
    pass.end();
    const heads = enc.beginComputePass();
    heads.setPipeline(this.pPolicy);
    heads.setBindGroup(0, this.bgPolicy);
    heads.dispatchWorkgroups(this.dimsPolicy[0], this.dimsPolicy[1]);
    heads.setPipeline(this.pValue);
    heads.setBindGroup(0, this.bgValue);
    heads.dispatchWorkgroups(this.dimsValue[0], this.dimsValue[1]);
    heads.end();
    if (this.timestamps) {
      enc.resolveQuerySet(this.querySet, 0, 2, this.bQuery, 0);
      enc.copyBufferToBuffer(this.bQuery, 0, this.readback, 0, TS_BYTES);
    }
    enc.copyBufferToBuffer(this.bPolicy, 0, this.readback, this.offPolicy, 4 * this.numMoves);
    enc.copyBufferToBuffer(this.bValue, 0, this.readback, this.offValue, 4 * this.vRows);
    if (activity) enc.copyBufferToBuffer(this.hFinal, 0, this.readback, this.offH, 4 * this.n);
    dev.queue.submit([enc.finish()]);
    const mapBytes = activity ? this.offH + 4 * this.n : this.offH;
    await this.readback.mapAsync(GPUMapMode.READ, 0, mapBytes);
    if (this.lost) { try { this.readback.unmap(); } catch { /* gone */ } throw new Error(this.lost); }
    const mapped = this.readback.getMappedRange(0, mapBytes);
    const policy = new Float32Array(mapped.slice(this.offPolicy, this.offPolicy + 4 * this.numMoves));
    const hid = new Float32Array(mapped.slice(this.offValue, this.offValue + 4 * this.vRows));
    const act = activity ? new Float32Array(mapped.slice(this.offH, this.offH + 4 * this.n)) : null;
    let gpuNs = 0;
    if (this.timestamps) {
      const ts = new BigUint64Array(mapped.slice(0, TS_BYTES));
      gpuNs = Number(ts[1] - ts[0]);
    }
    this.readback.unmap();
    this.lastForwardMs = performance.now() - t0;
    this.lastStepMs = gpuNs > 0 ? gpuNs / 1e6 / this.steps : this.lastForwardMs / this.steps;
    const value = valueFromHidden(this.arrays, this.valueHidden, hid);
    return { policy, value, activity: act };
  }

  /** Evaluate several inputs (sequentially); each result owns its arrays. */
  async forwardBatched(xs, opts) {
    const out = [];
    for (const x of xs) out.push(await this.forward(x, opts));
    return out;
  }

  /** Release every GPU buffer; the device stays usable by others. */
  destroy() {
    for (const b of this._bufs) { try { b.destroy(); } catch { /* already destroyed */ } }
    this._bufs = [];
    try { this.querySet?.destroy(); } catch { /* ignore */ }
    if (!this.lost) this.lost = 'destroyed';
  }
}
