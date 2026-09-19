"""STL filename mapping for the wizard-chess drop-in converter."""

from __future__ import annotations

from pathlib import Path

from game.wizard_assets import STL_ALIASES, map_stls


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
