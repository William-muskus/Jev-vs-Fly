"""Convert a local Harry Potter / wizard-chess STL set into GLBs the 3D board can load.

The Cults set at ``nbauchat/harry-potter-chess`` is **private-use / no-AI** and
is not vendored here. Copy the STLs you already have onto this machine and run:

    python -m game.wizard_assets --src /path/to/nbauchat/harry-potter-chess

``--src`` may be a folder or a zip. If omitted, well-known drop paths are
searched (including the nbauchat unzip layout from Downloads).

Files land in ``game/medieval/public/models/wizard/{k,q,b,n,r,p}.glb`` and the
board picks them up on the next reload (replacing the procedural stone set).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "game" / "medieval" / "public" / "models" / "wizard"

# nbauchat's Cults filenames, plus generic aliases.
STL_ALIASES: dict[str, str] = {
    "king3.stl": "k",
    "king.stl": "k",
    "k.stl": "k",
    "queen1.stl": "q",
    "queen.stl": "q",
    "q.stl": "q",
    "bishop5.stl": "b",
    "bishop.stl": "b",
    "b.stl": "b",
    "knight2.stl": "n",
    "knight.stl": "n",
    "horse.stl": "n",
    "n.stl": "n",
    "rook3.stl": "r",
    "rook.stl": "r",
    "castle.stl": "r",
    "r.stl": "r",
    "pawn2.stl": "p",
    "pawn.stl": "p",
    "p.stl": "p",
}

_NBAUCHAT = Path("nbauchat") / "harry-potter-chess"
_DOWNLOAD_STEM = "harry-potter-chess20241003-1-ye3pe1"


def candidate_dirs() -> list[Path]:
    """Folders (and zips) to search when ``--src`` is omitted."""
    home = Path.home()
    env = os.environ.get("WIZARD_STL_DIR", "").strip()
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    paths.extend(
        [
            OUT / "stl",
            REPO / _NBAUCHAT,
            REPO / _DOWNLOAD_STEM / _NBAUCHAT,
            home / "Downloads" / _DOWNLOAD_STEM / _NBAUCHAT,
            home / "Downloads" / f"{_DOWNLOAD_STEM}.zip",
            Path("/mnt/c/Users/Willi/Downloads") / _DOWNLOAD_STEM / _NBAUCHAT,
            Path("/mnt/c/Users/Willi/Downloads") / f"{_DOWNLOAD_STEM}.zip",
            Path("C:/Users/Willi/Downloads") / _DOWNLOAD_STEM / _NBAUCHAT,
        ]
    )
    # Keep first occurrence of each resolved path.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        key = path.expanduser()
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def find_src(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser()
    for path in candidate_dirs():
        zip_file = path.is_file() and path.suffix.lower() == ".zip"
        if zip_file or (path.is_dir() and map_stls(path)):
            return path
    return None


def map_stls(src: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not src.is_dir():
        return found
    for path in src.rglob("*.stl"):
        kind = STL_ALIASES.get(path.name.lower())
        if kind and kind not in found:
            found[kind] = path
    return found


def _unpack_zip(src: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="wizard-stl-"))
    with zipfile.ZipFile(src) as zf:
        zf.extractall(tmp)
    return tmp


def write_manifest(dest: Path, kinds: list[str]) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "manifest.json"
    path.write_text(json.dumps({"kinds": kinds}, indent=2) + "\n", encoding="utf-8")
    return path


def convert(src: Path, dest: Path = OUT) -> dict[str, Path]:
    try:
        import trimesh
    except ImportError as e:
        raise SystemExit("trimesh is required: pip install trimesh") from e

    work = src
    if src.is_file() and src.suffix.lower() == ".zip":
        work = _unpack_zip(src)
    elif not src.is_dir():
        raise SystemExit(f"not a directory or zip: {src}")

    dest.mkdir(parents=True, exist_ok=True)
    mapped = map_stls(work)
    if not mapped:
        raise SystemExit(f"no recognised STL files under {src}")

    written: dict[str, Path] = {}
    for kind, stl in mapped.items():
        mesh = trimesh.load(stl, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.dump(concatenate=False)))
        mesh.apply_translation(-mesh.bounds[0])
        # Sit on y=0, centred in x/z, Y-up.
        mesh.apply_translation([-mesh.centroid[0], 0, -mesh.centroid[2]])
        out = dest / f"{kind}.glb"
        mesh.export(out)
        written[kind] = out
        print(f"  {stl.name} -> {out.relative_to(REPO)}")
    write_manifest(dest, sorted(written))
    missing = sorted({"k", "q", "b", "n", "r", "p"} - set(written))
    if missing:
        print(f"missing ranks (procedural stone will fill in): {missing}", file=sys.stderr)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert wizard-chess STLs to GLBs for the 3D board.")
    p.add_argument(
        "--src",
        type=Path,
        default=None,
        help="folder or zip that contains the STL files (default: search known drop paths)",
    )
    p.add_argument("--out", type=Path, default=OUT, help="destination for k.glb … p.glb")
    args = p.parse_args(argv)
    src = find_src(args.src)
    if src is None:
        tried = "\n  ".join(str(p) for p in candidate_dirs())
        print(f"no STL set found. looked in:\n  {tried}", file=sys.stderr)
        return 2
    convert(src, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
