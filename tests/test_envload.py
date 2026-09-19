"""Repo .env loader — never prints secret values."""

from __future__ import annotations

import os
from pathlib import Path

from game.envload import jev_configured, load_repo_env


def test_load_repo_env_fills_empty_and_respects_existing(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("UNITTEST_ENVLOAD=from-file\n", encoding="utf-8")
    monkeypatch.delenv("UNITTEST_ENVLOAD", raising=False)
    assert load_repo_env(env) == env
    assert os.environ["UNITTEST_ENVLOAD"] == "from-file"
    monkeypatch.setenv("UNITTEST_ENVLOAD", "already-set")
    env.write_text("UNITTEST_ENVLOAD=new-value\n", encoding="utf-8")
    load_repo_env(env)
    assert os.environ["UNITTEST_ENVLOAD"] == "already-set"
    load_repo_env(env, overwrite=True)
    assert os.environ["UNITTEST_ENVLOAD"] == "new-value"
    monkeypatch.delenv("UNITTEST_ENVLOAD", raising=False)


def test_jev_configured_is_boolean(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert jev_configured() is False
    monkeypatch.setenv("TYPESAFE_API_KEY", "  ")
    assert jev_configured() is False
    monkeypatch.setenv("TYPESAFE_API_KEY", "present")
    assert jev_configured() is True
