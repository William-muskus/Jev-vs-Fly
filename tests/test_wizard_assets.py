"""STL filename mapping for the wizard-chess drop-in converter."""

from __future__ import annotations

import json
from pathlib import Path

from game.wizard_assets import STL_ALIASES, find_src, map_stls, write_manifest


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
