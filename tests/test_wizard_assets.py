"""STL filename mapping for the wizard-chess drop-in converter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from game.wizard_assets import STL_ALIASES, find_src, map_stls, tallest_axis, write_manifest


def test_nbauchat_filenames_map_to_ranks(tmp_path: Path):
    names = ["king3.stl", "queen1.stl", "bishop5.stl", "knight2.stl", "rook3.stl", "pawn2.stl"]
    for name in names:
        (tmp_path / name).write_text("solid fake", encoding="utf-8")
    found = map_stls(tmp_path)
    assert found.keys() == {"k", "q", "b", "n", "r", "p"}
    assert found["k"].name == "king3.stl"
    assert STL_ALIASES["king3.stl"] == "k"


def test_nested_generic_names(tmp_path: Path):
    nested = tmp_path / "nbauchat" / "harry-potter-chess"
    nested.mkdir(parents=True)
    (nested / "queen.stl").write_text("solid", encoding="utf-8")
    found = map_stls(tmp_path)
    assert found["q"].name == "queen.stl"


def test_write_manifest_lists_kinds(tmp_path: Path):
    path = write_manifest(tmp_path, ["k", "p"])
    assert json.loads(path.read_text(encoding="utf-8")) == {"kinds": ["k", "p"]}


def test_find_src_uses_wizard_stl_dir(tmp_path: Path, monkeypatch):
    (tmp_path / "king3.stl").write_text("solid", encoding="utf-8")
    monkeypatch.setenv("WIZARD_STL_DIR", str(tmp_path))
    assert find_src() == tmp_path


def test_tallest_axis_is_print_bed_z():
    assert tallest_axis((10.0, 8.0, 90.0)) == 2
    assert tallest_axis((4.0, 40.0, 5.0)) == 1
    assert tallest_axis((50.0, 3.0, 4.0)) == 0


def test_stand_y_up_tips_z_print_onto_y():
    trimesh = pytest.importorskip("trimesh")
    from game.wizard_assets import stand_y_up

    mesh = trimesh.creation.box(extents=(2.0, 3.0, 12.0))
    stand_y_up(mesh)
    assert tallest_axis(tuple(float(v) for v in mesh.extents)) == 1
