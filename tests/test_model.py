"""Tests for flychess.model (SpMM autograd, FlyBrain, losses). Fast: toy graphs only."""
from __future__ import annotations

import pickle

import pytest
import torch

from flychess.connectome.graph import toy_graph
from flychess.model import (
    BrainConfig,
    FlyBrain,
    SparseStructure,
    inverse_softplus,
    masked_policy_log_softmax,
    metrics_to_float,
    policy_value_loss,
    spmm,
    spmm_dense_reference,
)

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="needs CUDA")


@pytest.fixture(scope="module")
def graph():
    return toy_graph(n=120, nnz=900, n_in=12, n_out=10, seed=1)


@pytest.fixture
def config(tmp_path):
    return BrainConfig(graph_path=str(tmp_path / "g.npz"), steps=4, num_moves=64, input_dim=1280)


# ---- SpMM -----------------------------------------------------------------------------------------
def test_transpose_and_dense_agree(graph):
    s = SparseStructure(graph)
    v = torch.randn(graph.nnz)
    w = s.dense(v)
    wt = s.csr_t(v).to_dense()
    assert torch.allclose(w.t(), wt)
    h = torch.randn(graph.n, 5)
    assert torch.allclose(s.spmm(v, h), w @ h, atol=1e-5)
    assert torch.allclose(s.spmm_t(v, h), w.t() @ h, atol=1e-5)


@pytest.mark.parametrize("mode", ["sddmm", "gather"])
def test_gradcheck_against_dense(graph, mode):
    s = SparseStructure(graph, sddmm_mode=mode, sddmm_chunk=97)  # tiny chunks exercise the loop
    v = torch.randn(graph.nnz, dtype=torch.float64, requires_grad=True)
    h = torch.randn(graph.n, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: spmm(a, b, s), (v, h), eps=1e-6, atol=1e-6)
    # explicit comparison with the dense reference gradients
    g = torch.randn(graph.n, 3, dtype=torch.float64)
    out = spmm(v, h, s)
    dv, dh = torch.autograd.grad(out, (v, h), g)
    ref = spmm_dense_reference(v, h, s)
    dv_ref, dh_ref = torch.autograd.grad(ref, (v, h), g)
    assert torch.allclose(dv, dv_ref, atol=1e-10) and torch.allclose(dh, dh_ref, atol=1e-10)


def test_sddmm_fallback_matches(graph):
    s = SparseStructure(graph)
    g, h = torch.randn(graph.n, 7), torch.randn(graph.n, 7)
    assert torch.allclose(s._sddmm_sampled(g, h), s._sddmm_gather(g, h), atol=1e-5)


@cuda_only
def test_spmm_cuda_matches_cpu(graph):
    s_cpu, s_gpu = SparseStructure(graph), SparseStructure(graph, device="cuda")
    v = torch.randn(graph.nnz, requires_grad=True)
    h = torch.randn(graph.n, 6, requires_grad=True)
    out = spmm(v, h, s_cpu)
    g = torch.randn_like(out)
    dv, dh = torch.autograd.grad(out, (v, h), g)
    v2, h2 = v.detach().cuda().requires_grad_(), h.detach().cuda().requires_grad_()
    out2 = spmm(v2, h2, s_gpu)
    dv2, dh2 = torch.autograd.grad(out2, (v2, h2), g.cuda())
    for a, b in ((out, out2), (dv, dv2), (dh, dh2)):
        assert torch.allclose(a, b.cpu(), atol=1e-4)


# ---- FlyBrain -------------------------------------------------------------------------------------
def test_shapes_and_return_activity(graph, config):
    m = FlyBrain(graph, config)
    x = torch.randn(5, 1280)
    p, v = m(x)
    assert p.shape == (5, 64) and v.shape == (5, 1)
    assert torch.all(v.abs() <= 1.0)
    p2, v2, h = m(x, return_activity=True)
    assert h.shape == (5, graph.n)
    assert torch.equal(p, p2) and torch.equal(v, v2)
    with pytest.raises(ValueError):
        m(torch.randn(5, 1279))


def test_init_row_norm_and_signs(graph, config):
    m = FlyBrain(graph, config)
    w = m.effective_weights()
    assert torch.allclose(w.abs().sum() / graph.n, torch.tensor(1.0), atol=1e-5)
    assert torch.all(torch.sign(w) == m.sign)
    assert torch.allclose(torch.sigmoid(m.leak_logit), torch.full((graph.n,), config.alpha))
    assert torch.allclose(torch.nn.functional.softplus(inverse_softplus(torch.tensor([0.1, 1.0, 7.0]))),
                          torch.tensor([0.1, 1.0, 7.0]))
    m2 = FlyBrain(graph, config.replace(weight_init_scale=0.3))
    assert torch.allclose(m2.effective_weights().abs().sum() / graph.n, torch.tensor(0.3), atol=1e-5)
    counts = m.count_parameters()
    assert counts["syn_gain"] == graph.nnz and counts["total"] == sum(p.numel() for p in m.parameters())


def test_dale_sign_constraint_survives_training(graph, config):
    torch.manual_seed(0)
    m = FlyBrain(graph, config)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    x = torch.randn(8, 1280)
    tm = torch.randint(0, 64, (8,))
    tv = torch.rand(8) * 2 - 1
    for _ in range(10):
        p, v = m(x)
        loss, _ = policy_value_loss(p, v, tm, tv)
        opt.zero_grad()
        loss.backward()
        assert m.syn_gain.grad is not None and torch.isfinite(m.syn_gain.grad).all()
        opt.step()
    w = m.effective_weights()
    assert torch.all(torch.sign(w) == m.sign), "Dale's law violated after optimisation"
    assert torch.all(w != 0)


def test_grad_checkpoint_gives_same_gradients(graph, config):
    torch.manual_seed(4)
    m = FlyBrain(graph, config)
    x = torch.randn(3, 1280)
    tm, tv = torch.randint(0, 64, (3,)), torch.rand(3) * 2 - 1
    grads = []
    for ck in (False, True):
        m.grad_checkpoint = ck
        m.zero_grad()
        p, v = m(x)
        loss, _ = policy_value_loss(p, v, tm, tv)
        loss.backward()
        grads.append({k: q.grad.clone() for k, q in m.named_parameters()})
    for k in grads[0]:
        assert torch.allclose(grads[0][k], grads[1][k], atol=1e-6), k


def test_dale_false_is_unconstrained(graph, config):
    m = FlyBrain(graph, config.replace(dale=False))
    with torch.no_grad():
        m.syn_gain.mul_(-1)
    assert torch.all(torch.sign(m.effective_weights()) == -m.sign)


def test_masked_loss_and_metrics(graph, config):
    torch.manual_seed(1)
    logits = torch.randn(4, 64)
    value = torch.rand(4, 1) * 2 - 1
    mask = torch.zeros(4, 64, dtype=torch.bool)
    mask[:, :5] = True
    target = torch.tensor([0, 1, 2, 3])
    logp = masked_policy_log_softmax(logits, mask)
    assert torch.all(torch.isinf(logp[:, 5:])) and torch.allclose(logp[:, :5].exp().sum(1), torch.ones(4))
    _, met_t = policy_value_loss(logits, value, target, torch.tensor([1.0, 0, -1, 0]), legal_mask=mask,
                                 value_weight=0.5)
    # metrics are detached 0-d tensors (no host sync in the hot loop); metrics_to_float converts them
    assert set(met_t) == {"loss", "policy_loss", "value_loss", "top1", "top3"}
    assert all(isinstance(v, torch.Tensor) and v.dim() == 0 and not v.requires_grad for v in met_t.values())
    met = metrics_to_float(met_t)
    assert all(isinstance(v, float) for v in met.values())
    assert abs(met["loss"] - (met["policy_loss"] + 0.5 * met["value_loss"])) < 1e-5
    assert 0.0 <= met["top1"] <= met["top3"] <= 1.0
    # top-k accuracy must be exact on a hand-made example
    lg = torch.full((2, 64), -10.0)
    lg[0, 3] = 5.0
    lg[1, 7] = 5.0
    lg[1, 9] = 4.0
    met2 = metrics_to_float(policy_value_loss(lg, torch.zeros(2, 1), torch.tensor([3, 9]), torch.zeros(2))[1])
    assert met2["top1"] == 0.5 and met2["top3"] == 1.0
    # an illegal target gives an infinite loss (contract: targets must be legal)
    _, met3 = policy_value_loss(logits, value, torch.tensor([60, 61, 62, 63]), torch.zeros(4), legal_mask=mask)
    assert float(met3["policy_loss"]) == float("inf")


def test_row_without_legal_moves_raises():
    """A terminal position (no legal move) must fail loudly instead of producing NaN gradients."""
    logits = torch.randn(3, 16)
    mask = torch.ones(3, 16, dtype=torch.bool)
    mask[1] = False
    with pytest.raises(ValueError, match="no legal moves"):
        masked_policy_log_softmax(logits, mask)
    with pytest.raises(ValueError):
        policy_value_loss(logits, torch.zeros(3, 1), torch.zeros(3, dtype=torch.long), torch.zeros(3), legal_mask=mask)
    # a legal row of all-True / uint8 masks still works
    assert torch.isfinite(masked_policy_log_softmax(logits, torch.ones(3, 16, dtype=torch.uint8))).all()


@pytest.mark.parametrize("activation", ["relu", "gelu", "tanh", "satrelu"])
def test_model_is_picklable_and_value_head_uses_config_activation(graph, config, activation):
    m = FlyBrain(graph, config.replace(activation=activation))
    m2 = pickle.loads(pickle.dumps(m))  # multiprocessing-spawn workers / torch.save(model) need this
    x = torch.randn(2, 1280)
    assert torch.allclose(m(x)[0], m2(x)[0]) and torch.allclose(m(x)[1], m2(x)[1])
    # the value MLP's hidden non-linearity is the configured activation (the JS engine uses header.activation)
    inner = m.value_head[1]
    from flychess.model.flybrain import SatReLU
    expected = {"relu": torch.nn.ReLU, "gelu": torch.nn.GELU, "tanh": torch.nn.Tanh, "satrelu": SatReLU}[activation]
    assert isinstance(inner, expected)
    z = torch.linspace(-3, 3, 7)
    assert torch.allclose(inner(z), m.act(z))


def test_forward_matches_manual_dense_dynamics(graph, config):
    """Independent dense re-implementation of §4 in float64."""
    torch.manual_seed(2)
    m = FlyBrain(graph, config.replace(activation="tanh")).double()
    x = torch.randn(3, 1280, dtype=torch.float64)
    p, v, h_t = m(x, return_activity=True)
    s = m.canonical_structure()
    W = s.dense(m.effective_weights().double())
    a = torch.sigmoid(m.leak_logit)
    inp = torch.zeros(3, graph.n, dtype=torch.float64)
    inp[:, m.input_idx] = x @ m.w_in.t() + m.b_in
    h = torch.zeros(3, graph.n, dtype=torch.float64)
    for _ in range(config.steps):
        pre = h @ W.t() + m.bias + inp
        h = (1 - a) * h + a * torch.tanh(pre)
    assert torch.allclose(h, h_t.double(), atol=1e-5)
    out = h[:, m.output_idx]
    assert torch.allclose(p.double(), m.policy_head(out), atol=1e-5)
    assert torch.allclose(v.double(), torch.tanh(m.value_head(out)), atol=1e-5)


def test_state_dict_roundtrip_and_yaml(graph, config, tmp_path):
    m = FlyBrain(graph, config)
    sd = m.state_dict()
    assert "csr_indices" not in sd and "sign" in sd and "syn_count" in sd
    cfg_path = config.to_yaml(tmp_path / "brain.yaml")
    cfg2 = BrainConfig.from_yaml(cfg_path)
    assert cfg2 == config
    m2 = FlyBrain.from_checkpoint(sd, cfg2, graph)
    x = torch.randn(2, 1280)
    assert torch.allclose(m(x)[0], m2(x)[0])
    with pytest.raises(KeyError):
        BrainConfig.from_dict({"graph_path": "x", "bogus": 1})


def test_round_weights_to_f16(graph, config):
    m = FlyBrain(graph, config)
    m.round_weights_to_f16_()
    w = m.effective_weights()
    assert torch.allclose(w, w.half().float(), atol=1e-6)
    assert torch.all(torch.sign(w) == m.sign)


@cuda_only
def test_cpu_gpu_parity_and_autocast(graph, config):
    torch.manual_seed(3)
    m = FlyBrain(graph, config)
    x = torch.randn(4, 1280)
    tm, tv = torch.randint(0, 64, (4,)), torch.rand(4) * 2 - 1
    p, v = m(x)
    loss, _ = policy_value_loss(p, v, tm, tv)
    loss.backward()
    grads = {k: p_.grad.clone() for k, p_ in m.named_parameters()}
    mg = FlyBrain(graph, config)
    mg.load_state_dict(m.state_dict())
    mg = mg.cuda()
    pg, vg = mg(x.cuda())
    assert torch.allclose(p, pg.cpu(), atol=1e-4) and torch.allclose(v, vg.cpu(), atol=1e-4)
    lg, _ = policy_value_loss(pg, vg, tm.cuda(), tv.cuda())
    lg.backward()
    for k, p_ in mg.named_parameters():
        assert torch.allclose(grads[k], p_.grad.cpu(), atol=1e-4), k
    # bf16 autocast: dense heads run in bf16, recurrence stays fp32 -> close to the fp32 result
    mg.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pa, va, ha = mg(x.cuda(), return_activity=True)
        la, _ = policy_value_loss(pa, va, tm.cuda(), tv.cuda())
    assert pa.dtype == torch.bfloat16 and ha.dtype == torch.float32
    la.backward()
    assert torch.isfinite(mg.syn_gain.grad).all()
    assert torch.allclose(p, pa.float().cpu(), atol=0.1, rtol=0.05)


def test_bf16_config_keeps_recurrent_params_fp32_and_trainable(graph, config):
    """Regression: with ``dtype='bfloat16'`` the synapse logits (~ -2.5..-4.5, bf16 ulp ~ 0.02) used to be
    stored in bf16, so every Adam step (~ lr) rounded to zero and the connectome silently never learned."""
    torch.manual_seed(4)
    m = FlyBrain(graph, config.replace(dtype="bfloat16"))
    assert m.syn_gain.dtype == torch.float32
    assert m.bias.dtype == torch.float32 and m.leak_logit.dtype == torch.float32
    assert m.w_in.dtype == torch.bfloat16 and m.policy_head.weight.dtype == torch.bfloat16  # dense parts follow config
    gain0 = m.syn_gain.detach().clone()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(8, 1280)
    tm, tv = torch.randint(0, 64, (8,)), torch.rand(8) * 2 - 1
    for _ in range(5):
        opt.zero_grad()
        p, v = m(x)
        loss, _ = policy_value_loss(p, v, tm, tv)
        loss.backward()
        opt.step()
    moved = m.syn_gain.detach() != gain0
    # only synapses on an input->output path within `steps` get a gradient on this toy graph, but every
    # one of those must actually move (in bf16 storage none of them did)
    assert moved[m.syn_gain.grad != 0].all()
    assert 0.2 < moved.float().mean().item() < 0.6, f"{moved.float().mean().item():.0%} of syn_gain moved"
    assert torch.isfinite(m(x)[0].float()).all()


# ---- slow: full-brain benchmark (run with `-m slow`) ------------------------------------------------
@pytest.mark.slow
@cuda_only
def test_benchmark_full_brain(request, capsys):
    import time
    from pathlib import Path

    # Only runs when explicitly selected (`pytest -m slow`): it allocates ~3 GB of VRAM and, without
    # data/brain/full.npz, builds a 134k-neuron synthetic graph — not something a default run should do.
    if "slow" not in (request.config.option.markexpr or ""):
        pytest.skip("full-brain benchmark: run with `pytest -m slow`")

    from flychess.connectome.graph import BrainGraph
    from flychess.paths import BRAIN_DIR

    path = Path(BRAIN_DIR) / "full.npz"
    if path.exists():
        g = BrainGraph.load(path)
    else:  # synthetic stand-in of the same size — tests only, never the player
        g = toy_graph(n=134_000, nnz=2_700_000, n_in=2048, n_out=1415, seed=0)
    cfg = BrainConfig(graph_path=str(path), steps=8)
    m = FlyBrain(g, cfg).cuda()
    B = 256
    x = (torch.rand(B, 1280, device="cuda") < 0.2).float()
    tm = torch.randint(0, cfg.num_moves, (B,), device="cuda")
    tv = torch.rand(B, device="cuda") * 2 - 1

    def run():
        m.zero_grad(set_to_none=True)
        p, v = m(x)
        loss, _ = policy_value_loss(p, v, tm, tv)
        loss.backward()

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / 10
    peak = torch.cuda.max_memory_allocated() / 1e9
    assert torch.isfinite(m.syn_gain.grad).all()
    assert torch.all(torch.sign(m.effective_weights()) == m.sign)
    m.eval()
    with torch.no_grad():
        for _ in range(5):
            m(x[:1])
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(50):
            m(x[:1])
        torch.cuda.synchronize()
        dt1 = (time.perf_counter() - t) / 50
    with capsys.disabled():
        print(f"\n[bench] n={g.n:,} nnz={g.nnz:,} B={B} steps=8: fwd+bwd {dt * 1000:.1f} ms "
              f"({dt / 8 * 1000:.2f} ms/timestep), peak {peak:.2f} GB; inference B=1 {dt1 * 1000:.2f} ms")
    assert dt < 0.5 and peak < 8.0


def test_compute_reordering_matches_canonical_order(graph, config, monkeypatch):
    """The RCM compute ordering changes nothing observable: outputs, activity and gradients match."""
    import flychess.model.flybrain as fb

    torch.manual_seed(5)
    m_re = FlyBrain(graph, config.replace(activation="satrelu"))
    assert m_re.reordered
    monkeypatch.setenv("FLYCHESS_REORDER", "0")
    torch.manual_seed(5)
    m_id = fb.FlyBrain(graph, config.replace(activation="satrelu"))
    assert not m_id.reordered
    m_id.load_state_dict(m_re.state_dict())  # persistent state is canonical in both
    x = torch.randn(4, 1280)
    p1, v1, h1 = m_re(x, return_activity=True)
    p2, v2, h2 = m_id(x, return_activity=True)
    assert torch.allclose(p1, p2, atol=1e-5) and torch.allclose(v1, v2, atol=1e-6) and torch.allclose(h1, h2, atol=1e-5)
    (p1.sum() + v1.sum()).backward()
    (p2.sum() + v2.sum()).backward()
    for (n1, a), (n2, b) in zip(m_re.named_parameters(), m_id.named_parameters()):
        assert n1 == n2 and torch.allclose(a.grad, b.grad, atol=1e-5, rtol=1e-4), n1
