"""BrainGraph: the connectome subgraph consumed by the model (see docs/SPEC.md §2.3).

`build_brain_graph` (selection from the full FlyWire connectome) is implemented further down;
the dataclass, its (de)serialisation and validation live here so that every module shares them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# Dale's law in the fly CNS: acetylcholine excitatory, GABA and glutamate (GluCl) inhibitory,
# monoamines treated as excitatory/modulatory. Unknown transmitter -> excitatory.
NT_SIGN = {"ACH": 1, "GABA": -1, "GLUT": -1, "DA": 1, "SER": 1, "OCT": 1, "": 1}


@dataclass
class GraphConfig:
    region: str = "full"                 # 'full' | 'central'
    max_neurons: int | None = None       # keep top-k neurons by total synapse count (inputs/outputs always kept)
    min_syn: int = 5                     # per (pre, post) pair, summed over neuropils
    input_super_classes: tuple[str, ...] = ("sensory", "ascending", "sensory_ascending")
    max_inputs: int = 2048
    output_super_classes: tuple[str, ...] = ("descending", "motor")
    max_outputs: int = 2048
    name: str = "full"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BrainGraph:
    n: int
    root_ids: np.ndarray          # (n,) int64
    csr_indptr: np.ndarray        # (n+1,) int32, rows = POST-synaptic neuron
    csr_indices: np.ndarray       # (nnz,) int32, PRE-synaptic neuron of each synapse
    syn_count: np.ndarray         # (nnz,) float32
    sign: np.ndarray              # (nnz,) int8, +1 / -1
    input_idx: np.ndarray         # (n_in,) int32
    output_idx: np.ndarray        # (n_out,) int32
    super_class: np.ndarray       # (n,) str
    position: np.ndarray          # (n, 3) float32, NaN when unknown
    meta: dict = field(default_factory=dict)

    # ---- derived -------------------------------------------------------------------------------
    @property
    def nnz(self) -> int:
        return int(self.csr_indices.shape[0])

    @property
    def n_in(self) -> int:
        return int(self.input_idx.shape[0])

    @property
    def n_out(self) -> int:
        return int(self.output_idx.shape[0])

    def validate(self) -> None:
        n, nnz = self.n, self.nnz
        assert self.root_ids.shape == (n,)
        assert self.csr_indptr.shape == (n + 1,) and self.csr_indptr[0] == 0 and self.csr_indptr[-1] == nnz
        assert np.all(np.diff(self.csr_indptr) >= 0)
        assert self.syn_count.shape == (nnz,) and self.sign.shape == (nnz,)
        assert np.all(np.isin(self.sign, (-1, 1)))
        assert self.csr_indices.min(initial=0) >= 0 and self.csr_indices.max(initial=-1) < n
        assert self.input_idx.min(initial=0) >= 0 and self.input_idx.max(initial=-1) < n
        assert self.output_idx.min(initial=0) >= 0 and self.output_idx.max(initial=-1) < n
        assert len(np.unique(self.input_idx)) == self.n_in and len(np.unique(self.output_idx)) == self.n_out
        assert self.super_class.shape == (n,) and self.position.shape == (n, 3)
        # canonical CSR: column indices strictly increasing within every row (no duplicate synapses)
        row_starts = self.csr_indptr[:-1]
        row_ends = self.csr_indptr[1:]
        inner = np.ones(nnz, dtype=bool)
        inner[row_starts[row_starts < nnz]] = False  # first entry of every row is exempt
        assert np.all(np.diff(self.csr_indices)[inner[1:]] > 0), "CSR columns must be sorted & unique per row"
        # no self loops
        rows = np.repeat(np.arange(n, dtype=np.int32), row_ends - row_starts)
        assert not np.any(rows == self.csr_indices), "self-loops are not allowed"

    # ---- io ------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            root_ids=self.root_ids.astype(np.int64),
            csr_indptr=self.csr_indptr.astype(np.int32),
            csr_indices=self.csr_indices.astype(np.int32),
            syn_count=self.syn_count.astype(np.float32),
            sign=self.sign.astype(np.int8),
            input_idx=self.input_idx.astype(np.int32),
            output_idx=self.output_idx.astype(np.int32),
            super_class=self.super_class.astype(str),
            position=self.position.astype(np.float32),
            meta=np.array(json.dumps(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BrainGraph":
        with np.load(Path(path), allow_pickle=False) as z:
            g = cls(
                n=int(z["root_ids"].shape[0]),
                root_ids=z["root_ids"],
                csr_indptr=z["csr_indptr"],
                csr_indices=z["csr_indices"],
                syn_count=z["syn_count"],
                sign=z["sign"],
                input_idx=z["input_idx"],
                output_idx=z["output_idx"],
                super_class=z["super_class"].astype(str),
                position=z["position"],
                meta=json.loads(str(z["meta"])),
            )
        return g

    def summary(self) -> str:
        classes, counts = np.unique(self.super_class, return_counts=True)
        cls_str = ", ".join(f"{c}={k}" for c, k in zip(classes, counts))
        exc = int((self.sign > 0).sum())
        return (
            f"BrainGraph(n={self.n:,}, synapses={self.nnz:,} [{exc:,} excitatory / {self.nnz - exc:,} inhibitory], "
            f"inputs={self.n_in}, outputs={self.n_out}; {cls_str})"
        )


def toy_graph(n: int = 200, nnz: int = 2000, n_in: int = 16, n_out: int = 16, seed: int = 0) -> BrainGraph:
    """Random graph with the BrainGraph invariants. FOR TESTS ONLY — never a substitute for the fly brain."""
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, nnz * 2)
    post = rng.integers(0, n, nnz * 2)
    keep = pre != post
    pairs = np.unique(np.stack([post[keep], pre[keep]], 1), axis=0)[:nnz]
    post, pre = pairs[:, 0], pairs[:, 1]
    counts = np.bincount(post, minlength=n)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    nt = rng.choice(["ACH", "GABA", "GLUT"], size=n, p=[0.6, 0.2, 0.2])
    sign = np.array([NT_SIGN[nt[p]] for p in pre], dtype=np.int8)
    perm = rng.permutation(n)
    g = BrainGraph(
        n=n,
        root_ids=np.arange(n, dtype=np.int64) + 720_575_940_000_000_000,
        csr_indptr=indptr,
        csr_indices=pre.astype(np.int32),
        syn_count=rng.integers(5, 60, len(pre)).astype(np.float32),
        sign=sign,
        input_idx=np.sort(perm[:n_in]).astype(np.int32),
        output_idx=np.sort(perm[n_in:n_in + n_out]).astype(np.int32),
        super_class=np.array(["central"] * n),
        position=rng.random((n, 3)).astype(np.float32),
        meta={"toy": True, "seed": seed},
    )
    g.validate()
    return g
