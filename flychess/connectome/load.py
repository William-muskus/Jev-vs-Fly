"""Parse the FlyWire Codex tables into a `Connectome` (docs/SPEC.md §2.2).

The CSV tables (see `download.py`) are parsed with pandas, connections are aggregated over neuropil
to one row per (pre, post) pair, and the result is cached as `connectome.npz` (plain numpy arrays,
no pickle) so that subsequent loads take a second or two instead of minutes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from flychess import paths

CACHE_NAME = "connectome.npz"
REQUIRED_FILES = ("connections.csv.gz", "neurons.csv.gz")
OPTIONAL_FILES = ("classification.csv.gz", "consolidated_cell_types.csv.gz", "coordinates.csv.gz")


@dataclass
class Connectome:
    """The whole FlyWire connectome: neurons (N) with annotations and aggregated synapses (E).

    `pre[i] -> post[i]` with `syn_count[i]` synapses summed over neuropils. Edges are sorted by
    (pre, post) and unique. `root_ids` is sorted so that `np.searchsorted(root_ids, rid)` maps ids
    to indices.
    """

    root_ids: np.ndarray          # (N,) int64, sorted unique
    super_class: np.ndarray       # (N,) str, '' if unknown
    cell_class: np.ndarray        # (N,) str
    cell_type: np.ndarray         # (N,) str (consolidated primary_type)
    side: np.ndarray              # (N,) str: 'left'/'right'/'center'/''
    nt_type: np.ndarray           # (N,) str: ACH/GABA/GLUT/DA/SER/OCT/''
    position: np.ndarray          # (N, 3) float32 nm, NaN if unknown
    pre: np.ndarray               # (E,) int32 index into root_ids
    post: np.ndarray              # (E,) int32
    syn_count: np.ndarray         # (E,) int32, summed over neuropils
    neuropil: np.ndarray | None = None   # (E,) str, neuropil of the largest contributing row
    edge_nt_type: np.ndarray | None = None  # (E,) str, nt_type of the largest contributing row
    sources: dict = field(default_factory=dict)  # {filename: {'size': int, 'mtime': float}}

    @property
    def n(self) -> int:
        return int(self.root_ids.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.pre.shape[0])

    def index_of(self, root_ids: np.ndarray) -> np.ndarray:
        """Map root ids to row indices (raises KeyError if any id is unknown)."""
        rids = np.asarray(root_ids, dtype=np.int64)
        idx = np.searchsorted(self.root_ids, rids)
        idx[idx >= self.n] = 0
        if not np.all(self.root_ids[idx] == rids):
            raise KeyError("unknown root id(s)")
        return idx.astype(np.int32)

    def summary(self) -> str:
        classes, counts = np.unique(self.super_class, return_counts=True)
        cls_str = ", ".join(f"{c or '<none>'}={k:,}" for c, k in zip(classes, counts))
        return (
            f"Connectome(neurons={self.n:,}, edges={self.n_edges:,}, "
            f"synapses={int(self.syn_count.sum()):,}; {cls_str})"
        )

    # ---- io ------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "root_ids": self.root_ids.astype(np.int64),
            "super_class": _ustr(self.super_class),
            "cell_class": _ustr(self.cell_class),
            "cell_type": _ustr(self.cell_type),
            "side": _ustr(self.side),
            "nt_type": _ustr(self.nt_type),
            "position": self.position.astype(np.float32),
            "pre": self.pre.astype(np.int32),
            "post": self.post.astype(np.int32),
            "syn_count": self.syn_count.astype(np.int32),
            "sources": np.array(json.dumps(self.sources)),
        }
        if self.neuropil is not None:
            arrays["neuropil"] = _ustr(self.neuropil)
        if self.edge_nt_type is not None:
            arrays["edge_nt_type"] = _ustr(self.edge_nt_type)
        np.savez(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Connectome:
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(
                root_ids=z["root_ids"],
                super_class=z["super_class"],
                cell_class=z["cell_class"],
                cell_type=z["cell_type"],
                side=z["side"],
                nt_type=z["nt_type"],
                position=z["position"],
                pre=z["pre"],
                post=z["post"],
                syn_count=z["syn_count"],
                neuropil=z.get("neuropil"),
                edge_nt_type=z.get("edge_nt_type"),
                sources=json.loads(str(z["sources"])) if "sources" in z else {},
            )


def _ustr(a: np.ndarray) -> np.ndarray:
    """Unicode (non-object) string array so that it round-trips through npz without pickle."""
    return np.asarray(a).astype(str)


# ---- parsing helpers -------------------------------------------------------------------------------
def _read_csv(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", **kw)


def _str_col(df: pd.DataFrame, col: str, index: np.ndarray) -> np.ndarray:
    """Map `df[col]` (indexed by root_id) onto the universe `index`, '' where missing."""
    s = df[col].astype("string").fillna("")
    s.index = df["root_id"].to_numpy()
    s = s[~s.index.duplicated(keep="first")]
    return s.reindex(index).fillna("").to_numpy(dtype=str)


def aggregate_connections(
    pre: np.ndarray, post: np.ndarray, syn: np.ndarray, nt: np.ndarray, neuropil: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse (pre, post, neuropil) rows into unique (pre, post) pairs sorted by (pre, post).

    syn_count is summed; nt_type / neuropil are taken from the row with the largest syn_count
    (ties: first occurrence in file order). `pre`/`post` are indices in [0, n).
    """
    key = pre.astype(np.int64) * n + post.astype(np.int64)
    # lexsort is stable: primary key ascending, secondary syn descending, ties keep file order
    order = np.lexsort((-syn.astype(np.int64), key))
    key_s = key[order]
    first = np.empty(len(key_s), dtype=bool)
    first[0] = True
    np.not_equal(key_s[1:], key_s[:-1], out=first[1:])
    starts = np.flatnonzero(first)
    uniq = key_s[starts]
    syn_sum = np.add.reduceat(syn[order].astype(np.int64), starts).astype(np.int32)
    rep = order[starts]  # representative row (largest syn_count) for every pair
    return (
        (uniq // n).astype(np.int32),
        (uniq % n).astype(np.int32),
        syn_sum,
        nt[rep],
        neuropil[rep],
    )


def _source_info(data_dir: Path) -> dict:
    info = {}
    for name in REQUIRED_FILES + OPTIONAL_FILES:
        p = data_dir / name
        if p.exists():
            st = p.stat()
            info[name] = {"size": st.st_size, "mtime": st.st_mtime}
    return info


def parse_connectome(data_dir: str | Path = paths.CONNECTOME_DIR, verbose: bool = True) -> Connectome:
    """Parse the CSV tables in `data_dir` (no cache). See `load_connectome` for the cached entry point."""
    data_dir = Path(data_dir)
    for name in REQUIRED_FILES:
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"{data_dir / name} missing - run `fly download --connectome`")
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)

    log(f"parsing {data_dir / 'connections.csv.gz'} ...")
    con = _read_csv(
        data_dir / "connections.csv.gz",
        dtype={"pre_root_id": np.int64, "post_root_id": np.int64, "neuropil": "string",
               "syn_count": np.int32, "nt_type": "string"},
    )
    con["nt_type"] = con["nt_type"].fillna("")
    con["neuropil"] = con["neuropil"].fillna("")
    neu = _read_csv(data_dir / "neurons.csv.gz", dtype={"root_id": np.int64}, usecols=["root_id", "nt_type"])
    log(f"  {len(con):,} connection rows, {len(neu):,} neurons ({time.time() - t0:.1f}s)")

    # Universe of neurons: neurons.csv ∪ connection endpoints.
    root_ids = np.unique(np.concatenate([
        neu["root_id"].to_numpy(np.int64),
        con["pre_root_id"].to_numpy(np.int64),
        con["post_root_id"].to_numpy(np.int64),
    ]))
    n = len(root_ids)

    pre_i = np.searchsorted(root_ids, con["pre_root_id"].to_numpy(np.int64)).astype(np.int32)
    post_i = np.searchsorted(root_ids, con["post_root_id"].to_numpy(np.int64)).astype(np.int32)
    pre, post, syn_count, edge_nt, neuropil = aggregate_connections(
        pre_i, post_i, con["syn_count"].to_numpy(np.int32),
        con["nt_type"].to_numpy(dtype=str), con["neuropil"].to_numpy(dtype=str), n,
    )
    del con
    log(f"  aggregated to {len(pre):,} (pre, post) pairs between {n:,} neurons ({time.time() - t0:.1f}s)")

    nt_type = _str_col(neu, "nt_type", root_ids)

    empty = np.full(n, "", dtype=str)
    super_class, cell_class, side, cell_type = empty, empty.copy(), empty.copy(), empty.copy()
    p = data_dir / "classification.csv.gz"
    if p.exists():
        cls = _read_csv(p, dtype={"root_id": np.int64}, usecols=["root_id", "super_class", "class", "side"])
        super_class = _str_col(cls, "super_class", root_ids)
        cell_class = _str_col(cls, "class", root_ids)
        side = _str_col(cls, "side", root_ids)
    else:
        log("  classification.csv.gz missing: super_class/class/side left empty")
    p = data_dir / "consolidated_cell_types.csv.gz"
    if p.exists():
        ct = _read_csv(p, dtype={"root_id": np.int64}, usecols=["root_id", "primary_type"])
        cell_type = _str_col(ct, "primary_type", root_ids)
    else:
        log("  consolidated_cell_types.csv.gz missing: cell_type left empty")

    position = np.full((n, 3), np.nan, dtype=np.float32)
    p = data_dir / "coordinates.csv.gz"
    if p.exists():
        co = _read_csv(p, dtype={"root_id": np.int64, "position": "string"}, usecols=["root_id", "position"])
        co = co.drop_duplicates("root_id", keep="first")
        xyz = co["position"].str.strip("[]").str.split(expand=True).iloc[:, :3].astype(np.float32).to_numpy()
        idx = np.searchsorted(root_ids, co["root_id"].to_numpy(np.int64))
        ok = (idx < n) & (root_ids[np.minimum(idx, n - 1)] == co["root_id"].to_numpy(np.int64))
        position[idx[ok]] = xyz[ok]
    else:
        log("  coordinates.csv.gz missing: positions left NaN")

    conn = Connectome(
        root_ids=root_ids, super_class=super_class, cell_class=cell_class, cell_type=cell_type,
        side=side, nt_type=nt_type, position=position, pre=pre, post=post, syn_count=syn_count,
        neuropil=neuropil, edge_nt_type=edge_nt, sources=_source_info(data_dir),
    )
    log(f"  {conn.summary()} ({time.time() - t0:.1f}s)")
    return conn


def load_connectome(data_dir: str | Path = paths.CONNECTOME_DIR, cache: bool = True,
                    verbose: bool = True) -> Connectome:
    """Load the connectome from `data_dir`, using / creating `data_dir/connectome.npz` when `cache`.

    The cache is rebuilt automatically if any source CSV is newer or has a different size than when
    the cache was written.
    """
    data_dir = Path(data_dir)
    cache_path = data_dir / CACHE_NAME
    if cache and cache_path.exists():
        conn = Connectome.load(cache_path)
        if conn.sources == _source_info(data_dir):
            return conn
        if verbose:
            print("connectome.npz is stale (source tables changed), re-parsing")
    conn = parse_connectome(data_dir, verbose=verbose)
    if cache:
        conn.save(cache_path)
        if verbose:
            print(f"cached -> {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)")
    return conn
