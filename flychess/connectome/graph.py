"""BrainGraph: the connectome subgraph consumed by the model (see docs/SPEC.md §2.3).

`build_brain_graph` (selection from the full FlyWire connectome) is implemented further down;
the dataclass, its (de)serialisation and validation live here so that every module shares them.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from flychess import paths

if TYPE_CHECKING:  # load.py imports pandas; keep graph.py importable without it
    from flychess.connectome.load import Connectome

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
    def load(cls, path: str | Path) -> BrainGraph:
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


# =================================================================================================
# Building a BrainGraph from the real connectome (docs/SPEC.md §2.3)
# =================================================================================================
CENTRAL_EXCLUDED = ("optic", "visual_projection", "visual_centrifugal")


def _pick_by_degree(candidates: np.ndarray, degree: np.ndarray, root_ids: np.ndarray, k: int) -> np.ndarray:
    """Deterministic role selection: sort `candidates` by (degree desc, root_id asc), take the first k."""
    if len(candidates) == 0:
        return candidates.astype(np.int32)
    order = np.lexsort((root_ids[candidates], -degree[candidates].astype(np.int64)))
    return np.sort(candidates[order[:k]]).astype(np.int32)


def edge_signs(pre_nt: np.ndarray, edge_nt: np.ndarray | None = None) -> np.ndarray:
    """Dale's-law sign per synapse from the PRE-synaptic neuron's transmitter (`NT_SIGN`).

    A neuron without an annotated transmitter falls back to the transmitter predicted for the edge
    itself (`edge_nt`), and to +1 when neither is known.
    """
    nt = np.asarray(pre_nt).astype(str)
    if edge_nt is not None:
        edge_nt = np.asarray(edge_nt).astype(str)
        nt = np.where(nt == "", edge_nt, nt)
    uniq, inv = np.unique(nt, return_inverse=True)
    table = np.array([NT_SIGN.get(u, 1) for u in uniq], dtype=np.int8)
    return table[inv]


def build_brain_graph(conn: Connectome, cfg: GraphConfig) -> BrainGraph:
    """Select the trained subgraph of the connectome and build it in canonical CSR form.

    Pipeline (all steps deterministic):
      1. region filter on neurons (`central` drops the optic-lobe super classes);
      2. keep synapses with `syn_count >= min_syn` between kept neurons, drop self-loops;
      3. inputs  = top `max_inputs`  neurons of `input_super_classes`  by (out-degree desc, root_id);
         outputs = top `max_outputs` neurons of `output_super_classes` by (in-degree desc, root_id);
         (degree = number of distinct partners on the filtered edge set; a neuron in both lists is an
         input first and is removed from the output candidates);
      4. if `max_neurons`: keep inputs ∪ outputs plus the highest-ranked other neurons by
         (total synapse count in+out desc, root_id) so that n <= max_neurons;
      5. drop neurons that end up with zero degree, except inputs/outputs;
      6. rows = POST-synaptic neuron, columns sorted, i.e. entries ordered by (post, pre).
    Sign comes from the PRE-synaptic neuron's transmitter (`edge_signs`), never learned.
    """
    from flychess.connectome.load import Connectome  # local import: load.py depends on pandas

    assert isinstance(conn, Connectome)
    t0 = time.time()
    if cfg.region not in ("full", "central"):
        raise ValueError(f"unknown region {cfg.region!r} (expected 'full' or 'central')")
    N = conn.n
    keep_neuron = np.ones(N, dtype=bool)
    if cfg.region == "central":
        keep_neuron &= ~np.isin(conn.super_class, CENTRAL_EXCLUDED)

    # 2. edge filter
    pre, post, syn = conn.pre, conn.post, conn.syn_count
    e_keep = (syn >= cfg.min_syn) & keep_neuron[pre] & keep_neuron[post] & (pre != post)
    pre, post, syn = pre[e_keep], post[e_keep], syn[e_keep]
    edge_nt = conn.edge_nt_type[e_keep] if conn.edge_nt_type is not None else None

    # 3. roles
    out_deg = np.bincount(pre, minlength=N)
    in_deg = np.bincount(post, minlength=N)
    in_cand = np.flatnonzero(keep_neuron & np.isin(conn.super_class, cfg.input_super_classes))
    input_glob = _pick_by_degree(in_cand, out_deg, conn.root_ids, cfg.max_inputs)
    out_cand = np.flatnonzero(keep_neuron & np.isin(conn.super_class, cfg.output_super_classes))
    out_cand = out_cand[~np.isin(out_cand, input_glob)]
    output_glob = _pick_by_degree(out_cand, in_deg, conn.root_ids, cfg.max_outputs)
    role = np.zeros(N, dtype=bool)
    role[input_glob] = True
    role[output_glob] = True

    # 4. top-k by total synapse count
    if cfg.max_neurons is not None:
        total_syn = np.bincount(pre, weights=syn, minlength=N) + np.bincount(post, weights=syn, minlength=N)
        others = np.flatnonzero(keep_neuron & ~role)
        budget = max(cfg.max_neurons - int(role.sum()), 0)
        order = np.lexsort((conn.root_ids[others], -total_syn[others]))
        keep_neuron = role.copy()
        keep_neuron[others[order[:budget]]] = True
        e_keep = keep_neuron[pre] & keep_neuron[post]
        pre, post, syn = pre[e_keep], post[e_keep], syn[e_keep]
        edge_nt = edge_nt[e_keep] if edge_nt is not None else None

    # 5. zero-degree removal (roles exempt)
    deg = np.bincount(pre, minlength=N) + np.bincount(post, minlength=N)
    keep_neuron &= (deg > 0) | role
    kept = np.flatnonzero(keep_neuron)
    n = len(kept)
    new_index = np.full(N, -1, dtype=np.int64)
    new_index[kept] = np.arange(n)

    # 6. canonical CSR (rows = post, columns = pre, sorted by (post, pre))
    pre_n, post_n = new_index[pre], new_index[post]
    order = np.lexsort((pre_n, post_n))
    pre_n, post_n, syn = pre_n[order], post_n[order], syn[order]
    edge_nt = edge_nt[order] if edge_nt is not None else None
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(post_n, minlength=n), out=indptr[1:])
    sign = edge_signs(conn.nt_type[kept][pre_n], edge_nt)

    classes, counts = np.unique(conn.super_class[kept], return_counts=True)
    meta = {
        "cfg": cfg.to_dict(),
        "n": int(n),
        "nnz": len(pre_n),
        "n_in": len(input_glob),
        "n_out": len(output_glob),
        "super_class_counts": {str(c or ""): int(k) for c, k in zip(classes, counts)},
        "sign_counts": {"excitatory": int((sign > 0).sum()), "inhibitory": int((sign < 0).sum())},
        "total_syn_count": float(syn.sum()),
        "connectome": {"neurons": int(N), "edges": int(conn.n_edges)},
        "sources": dict(conn.sources),
        "build_time_s": round(time.time() - t0, 3),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    g = BrainGraph(
        n=n,
        root_ids=conn.root_ids[kept].astype(np.int64),
        csr_indptr=indptr.astype(np.int32),
        csr_indices=pre_n.astype(np.int32),
        syn_count=syn.astype(np.float32),
        sign=sign.astype(np.int8),
        input_idx=new_index[input_glob].astype(np.int32),
        output_idx=new_index[output_glob].astype(np.int32),
        super_class=conn.super_class[kept].astype(str),
        position=conn.position[kept].astype(np.float32),
        meta=meta,
    )
    g.validate()
    return g


def graph_path(cfg: GraphConfig) -> Path:
    return paths.BRAIN_DIR / f"{cfg.name}.npz"


def load_or_build(cfg: GraphConfig, out_path: str | Path | None = None, conn: Connectome | None = None,
                  verbose: bool = True) -> BrainGraph:
    """Return the BrainGraph for `cfg`, loading `data/brain/<cfg.name>.npz` when it exists.

    Otherwise the connectome is loaded (`load_connectome`, or the given `conn`), the graph is built,
    saved to `out_path` (default `data/brain/<cfg.name>.npz`) and returned.
    """
    path = Path(out_path) if out_path is not None else graph_path(cfg)
    if path.exists():
        return BrainGraph.load(path)
    if conn is None:
        from flychess.connectome.load import load_connectome

        conn = load_connectome(verbose=verbose)
    g = build_brain_graph(conn, cfg)
    g.save(path)
    if verbose:
        print(f"built {g.summary()} in {g.meta['build_time_s']}s -> {path}")
    return g
