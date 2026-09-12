// node --test web/test/  -- static checks on web/style.css: text contrast (WCAG AA) and the phone layout order.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const css = readFileSync(join(here, '..', 'style.css'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');

// ---------------------------------------------------------------- helpers
const tokens = Object.fromEntries([...css.matchAll(/--([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;/g)].map((m) => [m[1], m[2].toLowerCase()]));
const lin = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
const lum = (hex) => { const n = parseInt(hex.slice(1), 16); return 0.2126 * lin(n >> 16) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255); };
const contrast = (a, b) => { const [hi, lo] = [lum(a), lum(b)].sort((p, q) => q - p); return (hi + 0.05) / (lo + 0.05); };
const tok = (name) => { assert.ok(tokens[name], `token --${name} missing or not a 6-digit hex`); return tokens[name]; };

/** Declarations of `selector` (exact selector text) inside the given @media block (or top level when media is null). */
function decls(selector, media = null) {
  let scope = css;
  if (media) {
    scope = [...css.matchAll(/@media\s*\(([^)]*)\)\s*\{([\s\S]*?)\n\}/g)].filter((m) => m[1].replace(/\s/g, '') === media.replace(/\s/g, '')).map((m) => m[2]).join('\n');
    assert.ok(scope, `no @media (${media}) block`);
  }
  const out = {};
  for (const m of scope.matchAll(/(^|\n)\s*([^{}\n]+?)\s*\{([^{}]*)\}/g)) {
    if (m[2].split(',').map((s) => s.trim()).includes(selector)) {
      for (const d of m[3].split(';')) { const i = d.indexOf(':'); if (i > 0) out[d.slice(0, i).trim()] = d.slice(i + 1).trim(); }
    }
  }
  return out;
}

// ---------------------------------------------------------------- contrast (website-9)
test('--dim body text reaches WCAG AA (4.5:1) on every ground it is used on', () => {
  const dim = tok('dim');
  const brainvizBg = decls('.brainviz').background;
  assert.match(brainvizBg, /^#[0-9a-fA-F]{6}$/, '.brainviz background should be a flat hex colour');
  for (const [name, bg] of [['--bg', tok('bg')], ['--surface', tok('surface')], ['--surface-2', tok('surface-2')], ['.brainviz', brainvizBg.toLowerCase()]]) {
    const r = contrast(dim, bg);
    assert.ok(r >= 4.5, `--dim ${dim} on ${name} ${bg}: ${r.toFixed(2)}:1 < 4.5:1`);
  }
});

test('--muted secondary text reaches WCAG AA on the surfaces', () => {
  for (const g of ['bg', 'surface', 'surface-2']) assert.ok(contrast(tok('muted'), tok(g)) >= 4.5, `--muted on --${g}`);
});

test('board coordinates use a dedicated ink with real contrast against their square', () => {
  assert.equal(decls('.cb-coord.on-light').fill, 'var(--coord-on-light)');
  assert.equal(decls('.cb-coord.on-dark').fill, 'var(--coord-on-dark)');
  const onLight = contrast(tok('coord-on-light'), tok('sq-light'));
  const onDark = contrast(tok('coord-on-dark'), tok('sq-dark'));
  assert.ok(onLight >= 4.5, `coordinate ink on light squares ${onLight.toFixed(2)}:1`);
  assert.ok(onDark >= 4, `coordinate ink on dark squares ${onDark.toFixed(2)}:1`);
});

// ---------------------------------------------------------------- phone layout (website-8)
test('under 980px the board comes right after the avatar/commentary strip and before the brain canvas', () => {
  const M = 'max-width: 980px';
  assert.equal(decls('.fly-panel', M).display, 'contents', 'fly panel children must become grid items so they can be ordered around the board');
  const order = (sel) => { const v = decls(sel, M).order; assert.ok(v !== undefined, `${sel} needs an explicit order in the phone layout`); return Number(v); };
  const avatar = order('.fly-panel .avatar-wrap'), commentary = order('.fly-panel .commentary'), board = order('.board-col');
  const brain = order('.brainviz'), side = order('.side-panel');
  assert.ok(avatar < board && commentary < board, 'avatar + commentary strip sits above the board');
  assert.ok(board < brain, 'brain canvas comes after the board');
  assert.ok(brain < side, 'move list / actions come last');
  // the strip is compact: small avatar, commentary beside it in the second column
  assert.equal(decls('.fly-panel .avatar-wrap', M).width, '92px');
  assert.equal(decls('.fly-panel .commentary', M)['grid-column'], '2');
  for (const sel of ['.board-col', '.brainviz', '.side-panel']) assert.equal(decls(sel, M)['grid-column'], '1 / -1', `${sel} spans the row`);
});

test('the phone brain canvas height overrides the base rule (declared after it)', () => {
  const base = css.indexOf('.brainviz canvas {');
  const mobile = css.search(/@media\s*\(max-width:\s*980px\)\s*\{[^}]*\.brainviz canvas\s*\{\s*height:\s*160px/);
  assert.ok(base >= 0 && mobile >= 0, 'both the base and the phone canvas rules exist');
  assert.ok(mobile > base, 'the 160px phone rule must come after the 260px base rule or the cascade ignores it');
});
