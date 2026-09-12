"""BrainConfig: the hyper-parameters that, together with a graph and a state_dict, define a FlyBrain."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

ACTIVATIONS = ("relu", "gelu", "tanh")
DTYPES = ("float32", "bfloat16", "float16", "float64")


@dataclass
class BrainConfig:
    """See docs/SPEC.md §4.

    Attributes
    ----------
    graph_path : str          npz written by ``BrainGraph.save`` (``fly build-brain``).
    steps : int               recurrent timesteps per position.
    alpha : float             initial leak ``a = sigmoid(leak_logit)`` (learned per neuron).
    activation : str          'relu' | 'gelu' (tanh approximation, so JS can match it) | 'tanh'.
    input_dim : int           flattened board planes (20 × 64).
    num_moves : int           policy head outputs.
    weight_init_scale : float mean row-sum of |w| at init (see ``FlyBrain.init_syn_gain``).
    dale : bool               enforce synapse signs (``w = sign · softplus(gain)``); False = free sign.
    dtype : str               parameter dtype of the dense parts (``w_in``/``b_in`` and the heads); the recurrent
                              ``syn_gain``/``bias``/``leak_logit`` are always float32 ('float32' + autocast is the
                              recommended way to get bf16 compute; a bf16 parameter dtype has no fp32 master
                              weights, so small updates to the dense parts can round away).
    value_hidden : int        hidden width of the value MLP; 0 = plain ``Linear(n_out, 1)``.
    """

    graph_path: str = ""
    steps: int = 8
    alpha: float = 0.5
    activation: str = "relu"
    input_dim: int = 1280
    num_moves: int = 4168
    weight_init_scale: float = 1.0
    dale: bool = True
    dtype: str = "float32"
    value_hidden: int = 256

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {self.activation!r}")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie strictly inside (0, 1)")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        self.graph_path = str(self.graph_path)

    # ---- (de)serialisation ---------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown BrainConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path

    @classmethod
    def from_yaml(cls, path: str | Path) -> BrainConfig:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    def replace(self, **overrides: Any) -> BrainConfig:
        return self.from_dict({**self.to_dict(), **overrides})
