"""FlyBrain: the fruit-fly connectome as a recurrent chess-playing network (docs/SPEC.md §4).

Every neuron of the :class:`BrainGraph` is a leaky rate unit; every synapse of the connectome is a
weight whose *existence* and *sign* (Dale's law) are fixed and whose magnitude is learned. The board
is injected into the sensory ``input_idx`` neurons on each timestep and the move / value heads read
the final activity of the ``output_idx`` (descending & motor) neurons. There is no other pathway from
board to move: the fly brain is the player.

Dynamics (``h`` is kept neuron-major ``(n, B)`` internally, batch-first at the module boundary)::

    h_0 = 0
    for t in range(steps):
        pre        = W @ h_t + bias ;  pre[input_idx] += x @ w_in.T + b_in
        h_{t+1}    = (1 - a) * h_t + a * act(pre)          # a = sigmoid(leak_logit), per neuron
    policy_logits = policy_head(h_T[output_idx]) ;  value = tanh(value_head(h_T[output_idx]))
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from flychess.connectome.graph import BrainGraph
from flychess.model.config import BrainConfig
from flychess.model.spmm import SparseStructure

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16, "float64": torch.float64}
_MIN_MAGNITUDE = 1e-6  # |w| floor before inverse_softplus (log1p(syn_count) is > 0 anyway)


def inverse_softplus(x: Tensor) -> Tensor:
    """``y`` such that ``softplus(y) == x`` for ``x > 0``: ``y = x + log(-expm1(-x))``."""
    return x + torch.log(-torch.expm1(-x))


def _gelu_tanh(t: Tensor) -> Tensor:
    """GELU with the tanh approximation (module-level so that models stay picklable)."""
    return F.gelu(t, approximate="tanh")


class _SatReLUFn(torch.autograd.Function):
    """``sat * tanh(relu(t) / sat)`` saving only ``t`` for backward (one tensor per timestep instead of four)."""

    @staticmethod
    def forward(ctx, t: Tensor, sat: float) -> Tensor:
        ctx.save_for_backward(t)
        ctx.sat = sat
        return torch.tanh(torch.relu(t) * (1.0 / sat)).mul_(sat)

    @staticmethod
    def backward(ctx, g: Tensor):
        (t,) = ctx.saved_tensors
        y = torch.tanh(torch.relu(t) * (1.0 / ctx.sat))
        return g * (1.0 - y * y) * (t > 0), None


class SatReLU(nn.Module):
    """Saturating rectifier ``sat * tanh(relu(x) / sat)``: non-negative rates with a ceiling of ``sat``.

    Biologically a firing rate saturates; numerically it keeps the recurrent activity bounded, which
    an unbounded ReLU in an excitatory-loop network does not (activity drifted 65 -> 300 over one epoch).
    Linear (slope 1) for small inputs, so Dale's-law signs keep their meaning.
    """

    def __init__(self, sat: float) -> None:
        super().__init__()
        self.sat = float(sat)

    def forward(self, t: Tensor) -> Tensor:
        return _SatReLUFn.apply(t, self.sat)

    def extra_repr(self) -> str:
        return f"sat={self.sat}"


def activation_fn(name: str, sat: float = 10.0):
    """The recurrent non-linearity. GELU uses the tanh approximation so that JS/numpy can match it."""
    if name == "relu":
        return torch.relu
    if name == "satrelu":
        return SatReLU(sat)
    if name == "tanh":
        return torch.tanh
    if name == "gelu":
        return _gelu_tanh
    raise ValueError(f"unknown activation {name!r}")


def activation_module(name: str, sat: float = 10.0) -> nn.Module:
    """``nn.Module`` version of :func:`activation_fn` (same math), used inside the value MLP."""
    if name == "relu":
        return nn.ReLU()
    if name == "satrelu":
        return SatReLU(sat)
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    raise ValueError(f"unknown activation {name!r}")


class FlyBrain(nn.Module):
    """Recurrent network on the fly connectome with policy and value heads.

    Parameters
    ----------
    syn_gain (nnz,)          synapse magnitude logits; ``w = sign * softplus(syn_gain)`` (dale=True)
                             or ``w = syn_gain`` (dale=False).
    bias (n,)                per-neuron bias, init 0.
    leak_logit (n,)          per-neuron leak logit, init ``logit(alpha)``.
    w_in (n_in, input_dim), b_in (n_in,)   board projection into the sensory neurons.
    policy_head              ``Linear(n_out, num_moves)``.
    value_head               ``Linear(n_out, value_hidden) → act → Linear(value_hidden, 1)`` (or ``Linear(n_out, 1)``),
                             followed by ``tanh``. ``act`` is the *same* non-linearity as the recurrence
                             (``config.activation``); the export records it as ``header['value_activation']``
                             so the JS/numpy engines never have to guess it.
    ``syn_gain``, ``bias`` and ``leak_logit`` are always float32 regardless of ``config.dtype`` (which
    only sets the dtype of ``w_in``/``b_in`` and the heads): under bf16 their optimiser steps would be
    smaller than an ulp and the connectome would never learn.
    Buffers: ``sign (nnz,) float``, ``input_idx (n_in,) int64``, ``output_idx (n_out,) int64``,
    ``syn_count (nnz,) float``; plus the non-persistent CSR pattern (``csr_indptr``, ``csr_indices``,
    ``crow_t``, ``col_t``, ``perm_t``) which is fully determined by the graph.
    """

    def __init__(self, graph: BrainGraph, config: BrainConfig) -> None:
        super().__init__()
        self.config = config
        self.n = int(graph.n)
        self.nnz = int(graph.nnz)
        self.n_in = int(graph.n_in)
        self.n_out = int(graph.n_out)
        self.steps = int(config.steps)
        self.dale = bool(config.dale)
        self.act = activation_fn(config.activation, config.sat)
        dtype = _DTYPES[config.dtype]

        # ---- fixed connectome data (buffers) ----
        self.register_buffer("sign", torch.as_tensor(graph.sign, dtype=torch.float32))
        self.register_buffer("syn_count", torch.as_tensor(graph.syn_count, dtype=torch.float32))
        self.register_buffer("input_idx", torch.as_tensor(graph.input_idx, dtype=torch.int64))
        self.register_buffer("output_idx", torch.as_tensor(graph.output_idx, dtype=torch.int64))
        structure = SparseStructure(graph, device="cpu")
        for name, tensor in structure.tensors().items():
            self.register_buffer({"crow": "csr_indptr", "col": "csr_indices"}.get(name, name), tensor, persistent=False)
        self._structure: SparseStructure | None = None
        self._structure_device: torch.device | None = None
        # Recompute each recurrent step in backward instead of storing its activations: ~2x less
        # activation memory for ~+25 % time. Set by the trainer when batch size / VRAM demands it.
        self.grad_checkpoint: bool = False

        # ---- learnable parameters ----
        init_gain = self.init_syn_gain(self.syn_count, self.csr_indptr, config.weight_init_scale)
        if not self.dale:
            init_gain = self.sign * F.softplus(init_gain)  # free-sign weights start at the Dale init
        # The recurrent parameters are always float32, whatever ``config.dtype`` says: the forward
        # upcasts them anyway (see ``effective_weights``/``leak``), and in bf16 the initial gains
        # (~ -2.5..-4.5, ulp ~ 0.016..0.03) would swallow every optimiser step (~ lr = 1e-3), silently
        # freezing the whole connectome while biases/heads keep learning. ``config.dtype`` only
        # governs the dense parts (``w_in``/``b_in`` and the heads); prefer float32 + autocast.
        self.syn_gain = nn.Parameter(init_gain.float())
        self.bias = nn.Parameter(torch.zeros(self.n, dtype=torch.float32))
        self.leak_logit = nn.Parameter(
            torch.full((self.n,), math.log(config.alpha / (1 - config.alpha)), dtype=torch.float32)
        )
        self.w_in = nn.Parameter(torch.empty(self.n_in, config.input_dim, dtype=dtype))
        self.b_in = nn.Parameter(torch.zeros(self.n_in, dtype=dtype))
        nn.init.kaiming_uniform_(self.w_in, a=math.sqrt(5))
        self.policy_head = nn.Linear(self.n_out, config.num_moves, dtype=dtype)
        if config.value_hidden > 0:
            self.value_head = nn.Sequential(
                nn.Linear(self.n_out, config.value_hidden, dtype=dtype),
                activation_module(config.activation, config.sat),
                nn.Linear(config.value_hidden, 1, dtype=dtype),
            )
        else:
            self.value_head = nn.Linear(self.n_out, 1, dtype=dtype)

    # ---- initialisation ------------------------------------------------------------------------
    @staticmethod
    def init_syn_gain(syn_count: Tensor, csr_indptr: Tensor, scale: float = 1.0) -> Tensor:
        """Initial ``syn_gain`` from synapse counts.

        ``|w_k| = scale · log1p(c_k) / rownorm`` with ``rownorm = (1/n) Σ_k log1p(c_k)`` — the mean over
        *all* neurons (zero in-degree included) of the row-sum of ``log1p(c)`` — so that the mean
        over neurons of ``Σ_pre |w[post, pre]|`` equals ``scale`` (1.0 by default): each neuron's
        total incoming drive is O(1), a crude spectral-radius control. Magnitudes are floored at
        1e-6 before ``inverse_softplus`` so the logit stays finite.
        """
        n = int(csr_indptr.shape[0]) - 1
        lc = torch.log1p(syn_count.float())
        rownorm = lc.sum() / max(n, 1)
        rownorm = torch.clamp(rownorm, min=_MIN_MAGNITUDE)
        magnitude = torch.clamp(scale * lc / rownorm, min=_MIN_MAGNITUDE)
        return inverse_softplus(magnitude)

    # ---- structure -----------------------------------------------------------------------------
    def structure(self) -> SparseStructure:
        """The :class:`SparseStructure` view on the current device (rebuilt after ``.to(device)``)."""
        dev = self.csr_indices.device
        if self._structure is None or self._structure_device != dev:
            self._structure = SparseStructure(
                tensors={"crow": self.csr_indptr, "col": self.csr_indices, "crow_t": self.crow_t,
                         "col_t": self.col_t, "perm_t": self.perm_t},
                n=self.n,
            )
            self._structure_device = dev
        return self._structure

    # ---- weights -------------------------------------------------------------------------------
    def effective_weights(self) -> Tensor:
        """Signed synaptic weights ``w (nnz,)`` in float32, ordered like ``graph.csr_indices``."""
        gain = self.syn_gain.float()
        if self.dale:
            return self.sign * F.softplus(gain)
        return gain

    def leak(self) -> Tensor:
        """Per-neuron leak ``a = sigmoid(leak_logit)`` in float32."""
        return torch.sigmoid(self.leak_logit.float())

    @torch.no_grad()
    def round_weights_to_f16_(self) -> FlyBrain:
        """Round every weight that the web export stores as f16 to its f16 value, in place.

        After this the torch forward uses exactly the values the JS/numpy engines see, which is how
        the parity tests and ``export/testvectors.py`` obtain matching numbers. ``syn_gain`` is set to
        ``inverse_softplus(|round_f16(w)|)`` (dale) or ``round_f16(w)`` (free sign).
        """
        w = self.effective_weights().half().float()
        if self.dale:
            self.syn_gain.copy_(inverse_softplus(torch.clamp(w.abs(), min=_MIN_MAGNITUDE)).to(self.syn_gain.dtype))
        else:
            self.syn_gain.copy_(w.to(self.syn_gain.dtype))
        for lin in self._dense_linears():
            lin.weight.copy_(lin.weight.half().to(lin.weight.dtype))
        self.w_in.copy_(self.w_in.half().to(self.w_in.dtype))
        return self

    def _dense_linears(self) -> list[nn.Linear]:
        lins = [self.policy_head]
        if isinstance(self.value_head, nn.Sequential):
            lins += [m for m in self.value_head if isinstance(m, nn.Linear)]
        else:
            lins.append(self.value_head)
        return lins

    # ---- forward -------------------------------------------------------------------------------
    def forward(self, x: Tensor, return_activity: bool = False) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """``x: (B, input_dim)`` → ``(policy_logits (B, num_moves), value (B, 1)[, h_T (B, n)])``.

        The recurrent part always runs in float32 (the SpMM upcasts); under autocast the dense
        input projection and the heads may run in bf16.
        """
        if x.dim() != 2 or x.shape[1] != self.config.input_dim:
            raise ValueError(f"expected x of shape (B, {self.config.input_dim}), got {tuple(x.shape)}")
        B = x.shape[0]
        w = self.effective_weights()                             # (nnz,) fp32
        a = self.leak().unsqueeze(1)                              # (n, 1) fp32
        bias = self.bias.float().unsqueeze(1)                     # (n, 1)
        h_in = F.linear(x.to(self.w_in.dtype), self.w_in, self.b_in).float().t().contiguous()  # (n_in, B)

        use_ckpt = self.grad_checkpoint and torch.is_grad_enabled()
        h = torch.zeros(self.n, B, dtype=torch.float32, device=x.device)
        for t in range(self.steps):
            if use_ckpt and t > 0:
                h = checkpoint(self._step, h, w, a, bias, h_in, use_reentrant=False)
            else:
                h = self._step(h, w, a, bias, h_in, first=t == 0)

        out = h.index_select(0, self.output_idx).t()              # (B, n_out) fp32
        out_dense = out.to(self.policy_head.weight.dtype)
        policy_logits = self.policy_head(out_dense)
        value = torch.tanh(self.value_head(out_dense).float())
        if return_activity:
            return policy_logits, value, h.t()
        return policy_logits, value

    def _step(self, h: Tensor, w: Tensor, a: Tensor, bias: Tensor, h_in: Tensor, first: bool = False) -> Tensor:
        """One recurrent timestep on the neuron-major state ``h: (n, B)``."""
        # W @ h_0 is identically zero (h_0 = 0): skipping the SpMM on the first step changes nothing.
        pre = torch.zeros_like(h) if first else self.structure().spmm(w, h)
        pre = pre.add_(bias).index_add_(0, self.input_idx, h_in)   # in-place: SpMM does not save its output
        return torch.lerp(h, self.act(pre), a)                      # (1 - a) * h + a * act(pre), one kernel

    # ---- convenience ---------------------------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, state_dict: Mapping[str, Any], config: BrainConfig, graph: BrainGraph,
                        strict: bool = True) -> FlyBrain:
        """Build a FlyBrain for ``graph``/``config`` and load ``state_dict`` into it."""
        model = cls(graph, config)
        model.load_state_dict(dict(state_dict), strict=strict)
        return model

    def count_parameters(self) -> dict[str, int]:
        """Trainable parameter counts per group plus ``total``."""
        counts = {
            "syn_gain": self.syn_gain.numel(),
            "bias": self.bias.numel(),
            "leak_logit": self.leak_logit.numel(),
            "input_proj": self.w_in.numel() + self.b_in.numel(),
            "policy_head": sum(p.numel() for p in self.policy_head.parameters()),
            "value_head": sum(p.numel() for p in self.value_head.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts

    def extra_repr(self) -> str:
        return (f"n={self.n}, nnz={self.nnz}, n_in={self.n_in}, n_out={self.n_out}, steps={self.steps}, "
                f"activation={self.config.activation}, dale={self.dale}")


# ---- loss helpers ------------------------------------------------------------------------------
def masked_policy_log_softmax(logits: Tensor, legal_mask: Tensor | None) -> Tensor:
    """Log-softmax over moves with illegal ones (``legal_mask == False``) set to ``-inf`` first.

    Raises ``ValueError`` if a row of ``legal_mask`` has no legal move at all (a terminal position):
    such a row would silently turn into NaNs and poison the whole batch's gradients. Callers (MCTS,
    trainers) must handle terminal positions before asking the brain for a policy.
    """
    logits = logits.float()
    if legal_mask is not None:
        legal_mask = legal_mask.to(torch.bool)
        if not bool(legal_mask.any(dim=-1).all()):  # one small host sync; cheap next to the forward
            raise ValueError("legal_mask has a row with no legal moves (terminal position?)")
        logits = logits.masked_fill(~legal_mask, float("-inf"))
    return F.log_softmax(logits, dim=-1)


def policy_value_loss(
    logits: Tensor,
    value: Tensor,
    target_move: Tensor,
    target_value: Tensor,
    legal_mask: Tensor | None = None,
    value_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Cross-entropy on the target move (illegal moves masked) + MSE on the value.

    Returns ``(loss, metrics)`` with ``metrics = {loss, policy_loss, value_loss, top1, top3}`` as
    *detached 0-d tensors on the same device* — no host/device synchronisation happens here, so the
    trainer can keep the GPU queue full and convert only when it logs (:func:`metrics_to_float`,
    one sync for all five; ``MetricsLogger`` also accepts the tensors directly).
    ``target_move`` must be legal whenever ``legal_mask`` is given (otherwise the loss is +inf).
    """
    logp = masked_policy_log_softmax(logits, legal_mask)
    target_move = target_move.to(torch.int64).view(-1)
    policy_loss = F.nll_loss(logp, target_move)
    v = value.float().view(-1)
    value_loss = F.mse_loss(v, target_value.float().view(-1))
    loss = policy_loss + value_weight * value_loss
    with torch.no_grad():
        k = min(3, logp.shape[-1])
        topk = logp.topk(k, dim=-1).indices
        hit = topk == target_move.unsqueeze(1)
        metrics = {
            "loss": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "top1": hit[:, 0].float().mean(),
            "top3": hit.any(dim=1).float().mean(),
        }
    return loss, metrics


def metrics_to_float(metrics: Mapping[str, Tensor]) -> dict[str, float]:
    """``{name: float}`` from the tensor metrics of :func:`policy_value_loss` with a single device sync."""
    keys = list(metrics)
    values = torch.stack([metrics[k].detach().float().reshape(()) for k in keys]).tolist()
    return dict(zip(keys, values, strict=True))
