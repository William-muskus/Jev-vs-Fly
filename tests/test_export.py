"""Round-trip tests for the web export format (docs/SPEC.md §8)."""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from flychess.connectome.graph import toy_graph
from flychess.export import SUPER_CLASS_LEGEND, export_web, numpy_forward, read_flyb
from flychess.export.web import normalise_positions
from flychess.model import BrainConfig, FlyBrain

EXPECTED_ORDER = ["csr_indptr", "csr_indices", "w", "bias", "alpha", "input_idx", "output_idx", "w_in", "b_in",
                  "policy_w", "policy_b", "value_w", "value_b", "value_w2", "value_b2", "positions", "super_class"]
EXPECTED_DTYPES = {"csr_indptr": "i32", "csr_indices": "i32", "w": "f16", "bias": "f32", "alpha": "f32",
                   "input_idx": "i32", "output_idx": "i32", "w_in": "f16", "b_in": "f32", "policy_w": "f16",
                   "policy_b": "f32", "value_w": "f16", "value_b": "f32", "value_w2": "f16", "value_b2": "f32",
                   "positions": "f16", "super_class": "u8"}


@pytest.fixture(scope="module")
def graph():
    g = toy_graph(n=150, nnz=1200, n_in=14, n_out=11, seed=7)
    g.super_class = np.array(["central"] * 100 + ["descending"] * 20 + ["sensory"] * 25 + ["weird"] * 5)
    g.position[3] = np.nan  # unknown position
    return g


@pytest.fixture(scope="module")
def model(graph):
    torch.manual_seed(0)
    cfg = BrainConfig(graph_path="toy", steps=5, num_moves=96, value_hidden=32, activation="gelu")
    m = FlyBrain(graph, cfg)
    with torch.no_grad():  # make biases / leaks non-trivial so the parity test is meaningful
        m.bias.normal_(0, 0.3)
        m.leak_logit.normal_(0, 1.0)
        m.b_in.normal_(0, 0.5)
    return m.eval()


def test_export_roundtrip_header_and_layout(model, graph, tmp_path):
    header = export_web(model, graph, tmp_path, extra_meta={"run_name": "fly1", "train_steps": 42,
                                                             "elo_estimates": {"random": 900}, "note": "hi"})
    assert (tmp_path / "brain.json").exists() and (tmp_path / "brain.flyb").exists()
    assert (tmp_path / "brain.flyb.gz").exists()
    assert gzip.decompress((tmp_path / "brain.flyb.gz").read_bytes()) == (tmp_path / "brain.flyb").read_bytes()
    assert json.loads((tmp_path / "brain.json").read_text()) == header
    for key in ("steps", "activation", "n", "nnz", "n_in", "n_out", "num_moves", "num_planes", "run_name",
                "train_steps", "exported_at", "elo_estimates", "super_class_legend", "arrays"):
        assert key in header, key
    assert header["run_name"] == "fly1" and header["train_steps"] == 42 and header["note"] == "hi"
    assert header["elo_estimates"] == {"random": 900}
    assert header["num_planes"] == 20 and header["n"] == graph.n and header["nnz"] == graph.nnz
    assert header["steps"] == 5 and header["activation"] == "gelu" and header["value_head"] == "mlp"
    assert header["value_activation"] == "gelu"  # the value MLP's hidden non-linearity is explicit
    names = [e["name"] for e in header["arrays"]]
    assert names == EXPECTED_ORDER
    for e in header["arrays"]:
        assert e["dtype"] == EXPECTED_DTYPES[e["name"]], e["name"]
        assert e["offset"] % 8 == 0
        assert "scale" not in e
    assert header["arrays"][0]["offset"] == 0
    assert header["total_bytes"] == (tmp_path / "brain.flyb").stat().st_size
    by_name = {e["name"]: e for e in header["arrays"]}
    assert by_name["w_in"]["shape"] == [graph.n_in, 1280]
    assert by_name["policy_w"]["shape"] == [96, graph.n_out]
    assert by_name["value_w"]["shape"] == [32, graph.n_out] and by_name["value_w2"]["shape"] == [1, 32]
    assert by_name["positions"]["shape"] == [graph.n, 3] and by_name["super_class"]["shape"] == [graph.n]

    arrays, header2 = read_flyb(tmp_path)
    assert header2 == header
    assert np.array_equal(arrays["csr_indptr"], graph.csr_indptr)
    assert np.array_equal(arrays["csr_indices"], graph.csr_indices)
    assert np.array_equal(arrays["input_idx"], graph.input_idx)
    assert np.array_equal(arrays["output_idx"], graph.output_idx)
    assert arrays["w"].dtype == np.float16 and arrays["w"].shape == (graph.nnz,)
    assert np.allclose(arrays["w"].astype(np.float32), model.effective_weights().detach().numpy(), atol=1e-2)
    assert np.allclose(arrays["alpha"], torch.sigmoid(model.leak_logit).detach().numpy())
    # sign of the exported w is the connectome sign
    assert np.array_equal(np.sign(arrays["w"].astype(np.float32)), graph.sign)
    # positions in [0,1], NaN -> 0.5
    pos = arrays["positions"].astype(np.float32)
    assert pos.min() >= 0 and pos.max() <= 1 and np.allclose(pos[3], 0.5)
    # super_class legend
    legend = header["super_class_legend"]
    assert legend == list(SUPER_CLASS_LEGEND)
    sc = arrays["super_class"]
    assert legend[sc[0]] == "central" and legend[sc[100]] == "descending" and legend[sc[120]] == "sensory"
    assert legend[sc[-1]] == "unknown"


def test_numpy_forward_matches_torch(model, graph, tmp_path):
    export_web(model, graph, tmp_path)
    arrays, header = read_flyb(tmp_path)
    m = FlyBrain.from_checkpoint(model.state_dict(), model.config, graph).eval().round_weights_to_f16_()
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = (rng.random(1280) < 0.2).astype(np.float32)
        x[17 * 64:18 * 64] = 1.0  # bias plane
        policy, value, h = numpy_forward(arrays, header, x)
        with torch.no_grad():
            p_t, v_t, h_t = m(torch.from_numpy(x)[None], return_activity=True)
        assert policy.shape == (96,) and h.shape == (graph.n,)
        assert np.allclose(policy, p_t[0].numpy(), atol=1e-2), np.abs(policy - p_t[0].numpy()).max()
        assert abs(value - float(v_t[0, 0])) < 1e-2
        assert np.allclose(h, h_t[0].numpy(), atol=1e-3)
        assert np.abs(h).max() > 0  # the brain actually responded


def test_linear_value_head_and_i8(graph, tmp_path):
    cfg = BrainConfig(graph_path="toy", steps=2, num_moves=40, value_hidden=0)
    m = FlyBrain(graph, cfg).eval()
    with torch.no_grad():  # heavy-tailed rows: a single per-array scale would flush most of w_in to zero
        m.w_in[0].mul_(200.0)
        m.policy_head.weight[3].mul_(100.0)
    header = export_web(m, graph, tmp_path, quant="i8")
    names = [e["name"] for e in header["arrays"]]
    assert "value_w2" not in names and header["value_head"] == "linear"
    by_name = {e["name"]: e for e in header["arrays"]}
    assert by_name["w"]["dtype"] == "i8" and isinstance(by_name["w"]["scale"], float)   # 1-D: one scale
    assert by_name["value_w"]["shape"] == [1, graph.n_out] and isinstance(by_name["value_w"]["scale"], float)
    for name in ("w_in", "policy_w"):                                                   # 2-D: per-row scales
        assert by_name[name]["dtype"] == "i8"
        assert isinstance(by_name[name]["scale"], list) and len(by_name[name]["scale"]) == by_name[name]["shape"][0]
    arrays, header = read_flyb(tmp_path)
    # per-row dequantisation reproduces every row to i8 precision (the big rows did not kill the others)
    w_in = arrays["w_in"].astype(np.float32) * np.asarray(by_name["w_in"]["scale"], np.float32)[:, None]
    ref = m.w_in.detach().numpy()
    assert np.all(np.abs(w_in - ref) <= np.abs(ref).max(axis=1, keepdims=True) / 127 * 0.51)
    assert (arrays["w_in"][1:] != 0).mean() > 0.9
    x = np.zeros(1280, np.float32)
    x[17 * 64:18 * 64] = 1.0
    policy, value, _ = numpy_forward(arrays, header, x)
    with torch.no_grad():
        p_t, _ = m(torch.from_numpy(x)[None])
    assert np.allclose(policy, p_t[0].numpy(), atol=5e-2, rtol=2e-2)  # i8 is lossier
    assert -1 <= value <= 1


def test_normalise_positions():
    pos = np.array([[0, 10, np.nan], [2, 10, 5], [1, 10, 0]], np.float32)
    out = normalise_positions(pos)
    assert np.allclose(out[:, 0], [0, 1, 0.5]) and np.allclose(out[:, 1], 0.5) and np.allclose(out[:, 2], [0.5, 1, 0])


# ---- JS engine parity (needs node; skipped otherwise) -------------------------------------------
REPO = Path(__file__).resolve().parents[1]
JS_ENGINE = REPO / "web" / "engine" / "flybrain.js"
NODE = shutil.which("node")
_JS_RUNNER = """
import {{ readFileSync }} from 'node:fs';
import {{ parseArrays }} from '{loader}';
import {{ FlyBrain }} from '{engine}';
const dir = process.argv[2];
const header = JSON.parse(readFileSync(dir + '/brain.json', 'utf8'));
const buf = readFileSync(dir + '/brain.flyb');
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const brain = new FlyBrain({{ header, arrays: parseArrays(header, ab) }});
const xs = JSON.parse(readFileSync(dir + '/x.json', 'utf8'));
const out = xs.map((x) => {{ const r = brain.forward(Float32Array.from(x)); return {{ policy: Array.from(r.policy), value: r.value }}; }});
process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None or not JS_ENGINE.exists(), reason="needs node and web/engine/flybrain.js")
@pytest.mark.parametrize("activation", ["relu", "gelu", "tanh"])
def test_js_engine_matches_numpy_forward(graph, tmp_path, activation):
    """web/engine/flybrain.js == numpy_forward on the exported blob (policy and value), every activation."""
    torch.manual_seed(1)
    cfg = BrainConfig(graph_path="toy", steps=4, num_moves=96, value_hidden=32, activation=activation)
    m = FlyBrain(graph, cfg)
    with torch.no_grad():
        m.bias.normal_(0, 0.3)
        m.leak_logit.normal_(0, 1.0)
        m.b_in.normal_(0, 0.5)
        m.value_head[0].bias.normal_(0, 1.0)  # make the value MLP's hidden non-linearity matter
    export_web(m.eval(), graph, tmp_path)
    arrays, header = read_flyb(tmp_path)
    rng = np.random.default_rng(3)
    xs = (rng.random((3, 1280)) < 0.2).astype(np.float32)
    xs[:, 17 * 64:18 * 64] = 1.0
    (tmp_path / "x.json").write_text(json.dumps(xs.tolist()))
    runner = tmp_path / "run.mjs"
    runner.write_text(_JS_RUNNER.format(loader=(REPO / "web/engine/loader.js").as_uri(), engine=JS_ENGINE.as_uri()))
    res = subprocess.run([NODE, str(runner), str(tmp_path)], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    for x, r in zip(xs, out, strict=True):
        policy, value, _ = numpy_forward(arrays, header, x)
        assert np.allclose(np.asarray(r["policy"], np.float32), policy, atol=1e-3), \
            np.abs(np.asarray(r["policy"]) - policy).max()
        assert abs(r["value"] - value) < 1e-3, (r["value"], value)
