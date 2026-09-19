"""Convert a local Harry Potter / wizard-chess STL set into GLBs the 3D board can load.

The Cults set at ``nbauchat/harry-potter-chess`` is **private-use / no-AI** and
is not vendored here. Copy the STLs you already have onto this machine and run:

    python -m game.wizard_assets --src /path/to/nbauchat/harry-potter-chess

Files land in ``game/medieval/public/models/wizard/{k,q,b,n,r,p}.glb`` and the
board picks them up on the next reload (replacing the procedural stone set).
"""

from __future__ import annotations

import argparse
import sys
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


def map_stls(src: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in src.rglob("*.stl"):
        kind = STL_ALIASES.get(path.name.lower())
        if kind and kind not in found:
            found[kind] = path
    return found


def convert(src: Path, dest: Path = OUT) -> dict[str, Path]:
    try:
        import trimesh
    except ImportError as e:
        raise SystemExit("trimesh is required: pip install trimesh") from e

    dest.mkdir(parents=True, exist_ok=True)
    mapped = map_stls(src)
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
    missing = sorted({"k", "q", "b", "n", "r", "p"} - set(written))
    if missing:
        print(f"missing ranks (procedural stone will fill in): {missing}", file=sys.stderr)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert wizard-chess STLs to GLBs for the 3D board.")
    p.add_argument("--src", required=True, type=Path, help="folder that contains the STL files")
    p.add_argument("--out", type=Path, default=OUT, help="destination for k.glb … p.glb")
    args = p.parse_args(argv)
    if not args.src.is_dir():
        print(f"not a directory: {args.src}", file=sys.stderr)
        return 2
    convert(args.src, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
