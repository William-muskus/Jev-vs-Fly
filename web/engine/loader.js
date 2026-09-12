// loader.js — fetch, decompress, cache and parse the exported fly brain (SPEC §8).
//
// Blob layout: `brain.json` (header) describes every array in `brain.flyb` as
// {name, dtype: 'i32'|'f32'|'f16'|'i8'|'u8', shape, offset, length_bytes, scale?}.
// Offsets are 8-byte aligned so every typed-array view can be created directly on the buffer.
// f16 arrays are decoded once, on load, into Float32Array (table-driven decoder);
// i8 arrays are dequantised with their `scale` (scalar or per-row array).

/** @typedef {{name:string, dtype:string, shape:number[], offset:number, length_bytes:number, scale?:number|number[]}} ArraySpec */

let F16_TABLE = null;

/** Build (once) the 65536-entry half → single lookup table. 256 KiB, ~2 ms. */
export function f16Table() {
  if (F16_TABLE) return F16_TABLE;
  const t = new Float32Array(65536);
  for (let u = 0; u < 65536; u++) t[u] = halfToFloat(u);
  F16_TABLE = t;
  return t;
}

/** Scalar IEEE-754 binary16 → number (used to build the table; also handy in tests). */
export function halfToFloat(u) {
  const s = (u & 0x8000) ? -1 : 1;
  const e = (u >> 10) & 0x1f;
  const m = u & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;                // subnormal / zero
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/** number → binary16 bits (round-to-nearest-even). Used by tests/tools that write .flyb blobs. */
export function floatToHalf(x) {
  if (Number.isNaN(x)) return 0x7e00;
  const sign = x < 0 || (x === 0 && 1 / x < 0) ? 0x8000 : 0;
  x = Math.abs(x);
  if (x === Infinity) return sign | 0x7c00;
  if (x === 0) return sign;
  let e = Math.floor(Math.log2(x));
  let m = x / 2 ** e - 1;                              // in [0,1)
  if (e < -14) { m = x / 2 ** -14; e = -15; }           // subnormal: value = m * 2^-14 with m < 1
  let mant = Math.round(m * 1024);
  if (mant === 1024) { mant = 0; e += 1; }
  if (e > 15) return sign | 0x7c00;
  return sign | ((e + 15) << 10) | mant;
}

/** Decode an f16 buffer region into a fresh Float32Array. */
export function decodeF16(buffer, byteOffset, count) {
  const src = new Uint16Array(buffer, byteOffset, count);
  const out = new Float32Array(count);
  const t = f16Table();
  for (let i = 0; i < count; i++) out[i] = t[src[i]];
  return out;
}

function elementCount(shape) {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

/**
 * Parse every array described by the header out of the decoded blob.
 * @param {object} header  parsed brain.json
 * @param {ArrayBuffer} buffer  decoded brain.flyb
 * @returns {Record<string, Int32Array|Float32Array|Uint8Array>}
 */
export function parseArrays(header, buffer) {
  const list = header.arrays ?? header.tensors ?? header.layout ?? [];
  const specs = Array.isArray(list) ? list : Object.entries(list).map(([name, s]) => ({ name, ...s }));
  if (specs.length === 0) throw new Error('brain.json lists no arrays');
  const arrays = {};
  for (const spec of specs) {
    const count = elementCount(spec.shape);
    const off = spec.offset;
    if (off + spec.length_bytes > buffer.byteLength) {
      throw new Error(`brain.flyb truncated: array ${spec.name} ends at ${off + spec.length_bytes} > ${buffer.byteLength}`);
    }
    switch (spec.dtype) {
      case 'i32': arrays[spec.name] = new Int32Array(buffer, off, count); break;
      case 'f32': arrays[spec.name] = new Float32Array(buffer, off, count); break;
      case 'u8': arrays[spec.name] = new Uint8Array(buffer, off, count); break;
      case 'f16': arrays[spec.name] = decodeF16(buffer, off, count); break;
      case 'i8': {
        const q = new Int8Array(buffer, off, count);
        const out = new Float32Array(count);
        const sc = spec.scale ?? 1;
        if (Array.isArray(sc) || ArrayBuffer.isView(sc)) {           // per-row scale, rows = shape[0]
          const rowLen = count / sc.length;
          for (let i = 0; i < count; i++) out[i] = q[i] * sc[(i / rowLen) | 0];
        } else {
          for (let i = 0; i < count; i++) out[i] = q[i] * sc;
        }
        arrays[spec.name] = out;
        break;
      }
      default: throw new Error(`unknown dtype ${spec.dtype} for ${spec.name}`);
    }
  }
  return arrays;
}

function cacheKey(baseUrl, header) {
  const tag = `${header.run_name || 'brain'}@${header.exported_at || ''}#${header.nnz || ''}`;
  return `${baseUrl}brain.flyb?v=${encodeURIComponent(tag)}`;
}

async function cacheGet(key) {
  try {
    if (typeof caches === 'undefined') return null;
    const c = await caches.open('flychess-brain-v1');
    const r = await c.match(key);
    return r ? await r.arrayBuffer() : null;
  } catch { return null; }
}

async function cachePut(key, buffer) {
  try {
    if (typeof caches === 'undefined') return;
    const c = await caches.open('flychess-brain-v1');
    // evict older brains so the cache holds one model only
    for (const req of await c.keys()) if (req.url !== key) await c.delete(req);
    await c.put(key, new Response(buffer, { headers: { 'content-type': 'application/octet-stream' } }));
  } catch { /* quota / private mode: ignore */ }
}

/** Read a response body with progress callbacks; returns an ArrayBuffer of the raw bytes. */
async function readWithProgress(response, total, onProgress) {
  if (!response.body || !response.body.getReader) {
    const buf = await response.arrayBuffer();
    onProgress?.(buf.byteLength, total || buf.byteLength);
    return buf;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let p = 0;
  for (const c of chunks) { out.set(c, p); p += c.byteLength; }
  return out.buffer;
}

/** Streaming gunzip via DecompressionStream, reporting *compressed* progress against `total`. */
async function readGzipWithProgress(response, total, onProgress) {
  if (typeof DecompressionStream === 'undefined' || !response.body) {
    throw new Error('DecompressionStream unavailable');
  }
  let loaded = 0;
  const counting = new TransformStream({
    transform(chunk, controller) { loaded += chunk.byteLength; onProgress?.(loaded, total); controller.enqueue(chunk); },
  });
  const stream = response.body.pipeThrough(counting).pipeThrough(new DecompressionStream('gzip'));
  const buf = await new Response(stream).arrayBuffer();
  onProgress?.(total || loaded, total || loaded);
  return buf;
}

async function probe(url) {
  try {
    const r = await fetch(url, { method: 'HEAD' });
    if (!r.ok) return null;
    return { ok: true, size: Number(r.headers.get('content-length')) || 0 };
  } catch { return null; }
}

/**
 * Load the brain: header + decoded arrays.
 * @param {string} baseUrl  directory holding brain.json / brain.flyb(.gz), with trailing slash
 * @param {(loaded:number, total:number, phase:'header'|'download'|'decode', header:object)=>void} [onProgress]
 * @returns {Promise<{header: object, arrays: Record<string, any>, fromCache: boolean, bytes: number}>}
 */
export async function loadBrain(baseUrl = 'model/', onProgress) {
  if (!baseUrl.endsWith('/')) baseUrl += '/';
  const hr = await fetch(baseUrl + 'brain.json', { cache: 'no-cache' });
  if (!hr.ok) throw new Error(`cannot fetch ${baseUrl}brain.json (${hr.status}) — run \`fly export-web\` first`);
  const header = await hr.json();
  onProgress?.(0, header.gzip_bytes || header.blob_bytes || 0, 'header', header);
  const key = cacheKey(baseUrl, header);

  let buffer = await cacheGet(key);
  let fromCache = !!buffer;
  if (!buffer) {
    const gz = header.gzip_bytes !== undefined || header.has_gzip ? { ok: true, size: header.gzip_bytes || 0 } : await probe(baseUrl + 'brain.flyb.gz');
    if (gz && gz.ok) {
      const r = await fetch(baseUrl + 'brain.flyb.gz').catch(() => null);
      if (r && r.ok) {
        const total = Number(r.headers.get('content-length')) || gz.size || header.gzip_bytes || 0;
        try {
          buffer = await readGzipWithProgress(r, total, (l, t) => onProgress?.(l, t, 'download', header));
        } catch {
          buffer = null; // decompression unavailable or body not gzip: fall through to the raw blob
        }
      }
    }
    if (!buffer) {
      const r = await fetch(baseUrl + 'brain.flyb');
      if (!r.ok) throw new Error(`cannot fetch brain.flyb (${r.status})`);
      const total = Number(r.headers.get('content-length')) || header.blob_bytes || 0;
      buffer = await readWithProgress(r, total, (l, t) => onProgress?.(l, t, 'download', header));
    }
    if (buffer.byteLength % 8 !== 0) {
      // keep alignment guarantees for typed-array views
      const padded = new ArrayBuffer(Math.ceil(buffer.byteLength / 8) * 8);
      new Uint8Array(padded).set(new Uint8Array(buffer));
      buffer = padded;
    }
    await cachePut(key, buffer);
  }
  onProgress?.(buffer.byteLength, buffer.byteLength, 'decode', header);
  const arrays = parseArrays(header, buffer);
  return { header, arrays, fromCache, bytes: buffer.byteLength };
}
