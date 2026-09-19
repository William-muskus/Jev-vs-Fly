"""Load gitignored ``.env`` into ``os.environ`` without extra dependencies."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_repo_env(path: Path | None = None, *, overwrite: bool = False) -> Path | None:
    """Read ``KEY=value`` lines from the repo ``.env``. Existing non-empty env wins unless *overwrite*."""
    env_path = path or (REPO / ".env")
    if not env_path.is_file():
        return None
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        current = os.environ.get(key, "").strip()
        if overwrite or not current:
            os.environ[key] = value
    return env_path


def jev_configured() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY", "").strip())
