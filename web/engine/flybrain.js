// flybrain.js — the fly brain as a pure typed-array recurrent network (SPEC §4).
//
// Same math as flychess/model/flybrain.py:
//   h_0 = 0
//   for t in 0..steps:  pre = W·h + bias + inj ; h = (1-α)⊙h + α⊙act(pre)
//   policy = policy_w · h[output_idx] + policy_b
//   value  = tanh(value_w2 · act(value_w1 · h[output_idx] + value_b1) + value_b2)   (or linear head)
// where W is the connectome CSR (rows = post-synaptic neuron), inj[input_idx[k]] = w_in[k]·x + b_in[k].
// The board x is constant across the timesteps, so the injection is computed once per forward.

/** erf via Abramowitz–Stegun 7.1.26 (|error| < 1.5e-7) — enough for the 1e-2 parity tolerance. */
export function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return sign * y;
}

// 'gelu' is the exact (erf) form by default, as torch.nn.functional.gelu; the export header may set
// gelu_approximate: 'tanh' to select the tanh approximation the model was trained with.
const ACTS = {
  relu: (v) => (v > 0 ? v : 0),
  tanh: Math.tanh,
  gelu: (v) => 0.5 * v * (1 + erf(v * 0.7071067811865476)),
  gelu_tanh: (v) => 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))),
};

export class FlyBrain {
  /**
   * @param {{header: object, arrays: Record<string, Int32Array|Float32Array|Uint8Array>}} model
   */
  constructor({ header, arrays }) {
    this.header = header;
    this.n = header.n ?? arrays.csr_indptr.length - 1;
    this.nnz = header.nnz ?? arrays.csr_indices.length;
    this.steps = header.steps ?? 8;
    this.activation = header.activation ?? 'relu';
    const pickAct = (name) => {
      const geluTanh = name === 'gelu' && /^tanh$/i.test(String(header.gelu_approximate || ''));
      const fn = geluTanh ? ACTS.gelu_tanh : ACTS[name];
      if (!fn) throw new Error(`unknown activation ${name}`);
      return fn;
    };
    this.act = pickAct(this.activation);
    // the value MLP's hidden non-linearity may be exported separately (header.value_activation)
    this.valueAct = pickAct(header.value_activation || this.activation);
    this.inputDim = header.input_dim ?? (header.num_planes ?? 20) * 64;
    this.numMoves = header.num_moves ?? arrays.policy_b.length;

    this.indptr = arrays.csr_indptr;
    this.col = arrays.csr_indices;
    this.w = arrays.w;
    this.bias = arrays.bias;
    this.inputIdx = arrays.input_idx;
    this.outputIdx = arrays.output_idx;
    this.nIn = this.inputIdx.length;
    this.nOut = this.outputIdx.length;
    this.wIn = arrays.w_in;
    this.bIn = arrays.b_in;
    this.policyW = arrays.policy_w;
    this.policyB = arrays.policy_b;
    this.valueW = arrays.value_w;
    this.valueB = arrays.value_b;
    this.valueW2 = arrays.value_w2 ?? null;
    this.valueB2 = arrays.value_b2 ?? null;
    this.valueHidden = this.valueW2 ? this.valueW.length / this.nOut : 0;
    this.positions = arrays.positions ?? null;
    this.superClass = arrays.super_class ?? null;

    // per-neuron leak α (accept a scalar export too)
    const a = arrays.alpha;
    if (a.length === this.n) this.alpha = a;
    else { this.alpha = new Float32Array(this.n).fill(a[0]); }
    this.oneMinusAlpha = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) this.oneMinusAlpha[i] = 1 - this.alpha[i];

    if (this.w.length !== this.nnz || this.indptr.length !== this.n + 1) throw new Error('CSR shape mismatch');
    if (this.wIn.length !== this.nIn * this.inputDim) throw new Error('w_in shape mismatch');
    if (this.policyW.length !== this.numMoves * this.nOut) throw new Error('policy_w shape mismatch');

    // scratch buffers (double-buffered hidden state)
    this._h = new Float32Array(this.n);
    this._h2 = new Float32Array(this.n);
    this._base = new Float32Array(this.n);   // bias + injection, fixed for a forward pass
    this._out = new Float32Array(this.nOut);
    this._hid = new Float32Array(Math.max(this.valueHidden, 1));
    this.lastStepMs = 0;
  }

  /**
   * One full forward pass.
   * @param {Float32Array} x  flattened planes, length input_dim (1280)
   * @returns {{policy: Float32Array, value: number, activity: Float32Array}}  activity is the final hidden state (a view that is reused on the next call — copy if you keep it)
   */
  forward(x) {
    if (x.length !== this.inputDim) throw new Error(`input must have ${this.inputDim} values, got ${x.length}`);
    const n = this.n, act = this.act, D = this.inputDim;
    const indptr = this.indptr, col = this.col, w = this.w, alpha = this.alpha, oma = this.oneMinusAlpha;
    const base = this._base;
    base.set(this.bias);
    // sensory injection: constant across timesteps
    const wIn = this.wIn, bIn = this.bIn, inputIdx = this.inputIdx;
    for (let k = 0, off = 0; k < this.nIn; k++, off += D) {
      let s = bIn[k];
      for (let j = 0; j < D; j++) s += wIn[off + j] * x[j];
      base[inputIdx[k]] += s;
    }
    let h = this._h, h2 = this._h2;
    h.fill(0);
    const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const relu = act === ACTS.relu;
    for (let t = 0; t < this.steps; t++) {
      let e = indptr[0];
      if (relu) {
        for (let i = 0; i < n; i++) {
          const end = indptr[i + 1];
          let s = base[i];
          for (; e < end; e++) s += w[e] * h[col[e]];
          h2[i] = oma[i] * h[i] + (s > 0 ? alpha[i] * s : 0);
        }
      } else {
        for (let i = 0; i < n; i++) {
          const end = indptr[i + 1];
          let s = base[i];
          for (; e < end; e++) s += w[e] * h[col[e]];
          h2[i] = oma[i] * h[i] + alpha[i] * act(s);
        }
      }
      const tmp = h; h = h2; h2 = tmp;
    }
    this.lastStepMs = ((typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0) / this.steps;
    // keep the final state in this._h so the returned activity stays valid until the next forward
    this._h = h; this._h2 = h2;
    return this._heads(h);
  }

  _heads(h) {
    const nOut = this.nOut, out = this._out, outputIdx = this.outputIdx;
    for (let i = 0; i < nOut; i++) out[i] = h[outputIdx[i]];
    const policy = new Float32Array(this.numMoves);
    const pw = this.policyW, pb = this.policyB;
    const n4 = nOut & ~3;
    for (let m = 0, off = 0; m < this.numMoves; m++, off += nOut) {
      let s0 = pb[m], s1 = 0, s2 = 0, s3 = 0;
      let j = 0;
      for (; j < n4; j += 4) {
        s0 += pw[off + j] * out[j]; s1 += pw[off + j + 1] * out[j + 1];
        s2 += pw[off + j + 2] * out[j + 2]; s3 += pw[off + j + 3] * out[j + 3];
      }
      for (; j < nOut; j++) s0 += pw[off + j] * out[j];
      policy[m] = s0 + s1 + s2 + s3;
    }
    let value;
    if (this.valueHidden > 0) {
      const H = this.valueHidden, vw = this.valueW, vb = this.valueB, hid = this._hid, act = this.valueAct;
      for (let k = 0, off = 0; k < H; k++, off += nOut) {
        let s = vb[k];
        for (let j = 0; j < nOut; j++) s += vw[off + j] * out[j];
        hid[k] = act(s);
      }
      let v = this.valueB2[0];
      for (let k = 0; k < H; k++) v += this.valueW2[k] * hid[k];
      value = Math.tanh(v);
    } else {
      let v = this.valueB[0];
      const vw = this.valueW;
      for (let j = 0; j < nOut; j++) v += vw[j] * out[j];
      value = Math.tanh(v);
    }
    return { policy, value, activity: h };
  }

  /** Simple loop over inputs; each result gets its own copy of the activity. */
  forwardBatched(xs) {
    const res = [];
    for (const x of xs) {
      const r = this.forward(x);
      res.push({ policy: r.policy, value: r.value, activity: r.activity.slice() });
    }
    return res;
  }
}

/**
 * Softmax over the legal move indices only.
 * @param {Float32Array} policy  raw logits (numMoves)
 * @param {ArrayLike<number>} legalIdx
 * @param {number} [temperature=1]
 * @returns {Float32Array} probabilities aligned with legalIdx
 */
export function policyForLegal(policy, legalIdx, temperature = 1) {
  const L = legalIdx.length;
  const p = new Float32Array(L);
  if (L === 0) return p;
  const invT = 1 / Math.max(temperature, 1e-6);
  let mx = -Infinity;
  for (let i = 0; i < L; i++) { const v = policy[legalIdx[i]] * invT; p[i] = v; if (v > mx) mx = v; }
  let sum = 0;
  for (let i = 0; i < L; i++) { const e = Math.exp(p[i] - mx); p[i] = e; sum += e; }
  for (let i = 0; i < L; i++) p[i] /= sum;
  return p;
}

/** Index of the largest entry of a typed array. */
export function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

/** Sample an index from a probability vector (uses Math.random unless `rnd` given). */
export function sampleIndex(probs, rnd = Math.random) {
  let r = rnd();
  for (let i = 0; i < probs.length; i++) { r -= probs[i]; if (r <= 0) return i; }
  return probs.length - 1;
}

/** Indices of the k largest values, descending. */
export function topK(arr, k) {
  const idx = Array.from({ length: arr.length }, (_, i) => i);
  idx.sort((a, b) => arr[b] - arr[a]);
  return idx.slice(0, k);
}
