"""Dashboard server (docs/SPEC.md §10).

* ``GET /``                            static ``index.html``
* ``GET /static/...``                  the vanilla-JS UI (no build step, no CDN)
* ``GET /api/runs``                    ``list_runs(runs_dir)``
* ``GET /api/default``                 ``{"run": <name or null>}`` — run the CLI asked to open
* ``GET /api/run/{name}``              ``{"run": run.json, "metrics": [last 5000 records]}``
* ``GET /api/run/{name}/positions``    normalised 3-D positions of the sampled neurons (from the graph npz)
* ``WS  /ws/{name}``                   ``{"kind": "init", "run": ..., "metrics": [...]}`` then every new record

The websocket tails ``metrics.jsonl`` by polling its size every ``POLL_INTERVAL`` seconds; if the run does not
exist yet it simply waits for it to appear.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from flychess import paths
from flychess.train.metrics import METRICS_FILE, list_runs, read_metrics, read_run_json

log = logging.getLogger("flychess.dashboard")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INIT_LINES = 5000          # records sent on GET /api/run and on websocket init
POLL_INTERVAL = 0.5        # seconds between metrics.jsonl size checks
MAX_POSITION_SAMPLES = 4096


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def _safe_name(name: str) -> str:
    """Reject path traversal in run names."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid run name")
    return name


def _resolve_graph_path(run_dir: Path, info: dict | None) -> Path | None:
    info = info or {}
    gp = info.get("graph_path") or (info.get("config") or {}).get("graph_path")
    if not gp:
        return None
    p = Path(gp).expanduser()
    for candidate in (p, run_dir / p, paths.HOME / p, paths.REPO_ROOT / p):
        if candidate.is_file():
            return candidate
    return None


def _neuron_positions(graph_path: Path, neuron_idx: np.ndarray | None) -> dict:
    """Load only ``position`` (+ ``super_class``) from a BrainGraph npz and normalise to [0, 1] per axis."""
    with np.load(graph_path, allow_pickle=False) as z:
        pos = np.asarray(z["position"], dtype=np.float32)
        sc = z["super_class"].astype(str) if "super_class" in z.files else None
    n = int(pos.shape[0])
    finite = np.isfinite(pos).all(axis=1)
    lo = np.nanmin(pos[finite], axis=0) if finite.any() else np.zeros(3, np.float32)
    hi = np.nanmax(pos[finite], axis=0) if finite.any() else np.ones(3, np.float32)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    norm = (pos - lo) / span
    if neuron_idx is None:
        k = min(n, MAX_POSITION_SAMPLES)
        neuron_idx = np.linspace(0, n - 1, k).astype(np.int64) if k > 0 else np.zeros(0, np.int64)
    neuron_idx = np.asarray(neuron_idx, dtype=np.int64)
    neuron_idx = neuron_idx[(neuron_idx >= 0) & (neuron_idx < n)]
    sel = norm[neuron_idx]
    sel_list: list[list[float] | None] = [
        [float(x), float(y), float(zz)] if np.isfinite(row).all() else None for (x, y, zz), row in zip(sel, sel)
    ]
    return {
        "n": n,
        "neuron_idx": neuron_idx.tolist(),
        "positions": sel_list,
        "super_class": sc[neuron_idx].tolist() if sc is not None else None,
        "graph_path": str(graph_path),
    }


def _read_new_lines(path: Path, offset: int) -> tuple[list[bytes], int]:
    """Read complete lines appended after ``offset``. Returns (lines, new_offset) — a torn last line is left."""
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    if not data:
        return [], offset
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    complete = data[: end + 1]
    return [ln for ln in complete.split(b"\n") if ln.strip()], offset + len(complete)


# ------------------------------------------------------------------------------------------------
# app factory
# ------------------------------------------------------------------------------------------------
def create_app(runs_dir: str | os.PathLike | None = None, default_run: str | None = None) -> FastAPI:
    runs_root = Path(runs_dir) if runs_dir is not None else paths.RUNS_DIR
    app = FastAPI(title="fly-chess dashboard", docs_url=None, redoc_url=None)
    app.state.runs_dir = runs_root
    app.state.default_run = default_run
    positions_cache: dict[tuple[str, tuple[int, ...]], dict] = {}

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/runs")
    async def api_runs() -> list[dict]:
        return await asyncio.to_thread(list_runs, runs_root)

    @app.get("/api/default")
    async def api_default() -> dict:
        return {"run": app.state.default_run, "runs_dir": str(runs_root)}

    @app.get("/api/run/{name}")
    async def api_run(name: str) -> dict:
        run_dir = runs_root / _safe_name(name)
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"run {name!r} not found")
        info, metrics = await asyncio.to_thread(
            lambda: (read_run_json(run_dir), read_metrics(run_dir, last_n=INIT_LINES))
        )
        return {"run": info or {}, "metrics": metrics}

    @app.get("/api/run/{name}/positions")
    async def api_positions(name: str) -> JSONResponse:
        run_dir = runs_root / _safe_name(name)
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"run {name!r} not found")
        info = read_run_json(run_dir)
        graph_path = _resolve_graph_path(run_dir, info)
        if graph_path is None:
            return JSONResponse({"available": False, "reason": "no graph_path in run.json (or file missing)"})
        # positions of the neurons in the latest activity sample, else an even sample of the graph
        latest = await asyncio.to_thread(read_metrics, run_dir, 1, ["activity"])
        idx = latest[0].get("neuron_idx") if latest else None
        idx_arr = np.asarray(idx, dtype=np.int64) if idx else None
        key = (str(graph_path), tuple(idx_arr.tolist()) if idx_arr is not None else ())
        if key not in positions_cache:
            try:
                positions_cache.clear()  # keep at most one entry (activity samples are usually fixed per run)
                positions_cache[key] = await asyncio.to_thread(_neuron_positions, graph_path, idx_arr)
            except Exception as exc:  # noqa: BLE001 - corrupt / foreign npz must not take the dashboard down
                log.warning("cannot read positions from %s: %s", graph_path, exc)
                return JSONResponse({"available": False, "reason": str(exc)})
        return JSONResponse({"available": True, **positions_cache[key]})

    @app.websocket("/ws/{name}")
    async def ws_run(websocket: WebSocket, name: str) -> None:
        await websocket.accept()
        try:
            run_dir = runs_root / _safe_name(name)
        except HTTPException:
            await websocket.close(code=1008)
            return
        metrics_path = run_dir / METRICS_FILE

        async def watch_client() -> None:
            # Pump incoming frames so that a client disconnect raises here and cancels the tail loop.
            while True:
                await websocket.receive_text()

        async def tail() -> None:
            # wait for the run to appear (the dashboard may be started before training)
            while not metrics_path.exists():
                await asyncio.sleep(POLL_INTERVAL)
            offset = metrics_path.stat().st_size
            info, metrics = await asyncio.to_thread(
                lambda: (read_run_json(run_dir), read_metrics(run_dir, last_n=INIT_LINES))
            )
            await websocket.send_text(json.dumps({"kind": "init", "run": info or {}, "metrics": metrics}))
            while True:
                await asyncio.sleep(POLL_INTERVAL)
                try:
                    size = metrics_path.stat().st_size
                except FileNotFoundError:
                    continue
                if size < offset:  # truncated / rewritten
                    offset = 0
                if size == offset:
                    continue
                lines, offset = await asyncio.to_thread(_read_new_lines, metrics_path, offset)
                for line in lines:
                    try:
                        json.loads(line)  # only forward valid JSON objects
                    except json.JSONDecodeError:
                        continue
                    await websocket.send_text(line.decode("utf-8", errors="replace"))

        watcher = asyncio.create_task(watch_client())
        tailer = asyncio.create_task(tail())
        try:
            done, pending = await asyncio.wait({watcher, tailer}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, (WebSocketDisconnect, RuntimeError)):
                    log.warning("websocket /ws/%s ended: %r", name, exc)
        except asyncio.CancelledError:
            watcher.cancel()
            tailer.cancel()
            raise

    return app


# ------------------------------------------------------------------------------------------------
# entry points
# ------------------------------------------------------------------------------------------------
def serve(
    run: str | None = None,
    port: int = 8765,
    open_browser: bool = False,
    host: str = "127.0.0.1",
    runs_dir: str | os.PathLike | None = None,
) -> None:
    """Run the dashboard with uvicorn (blocking). ``run`` selects the run the UI opens by default."""
    import uvicorn

    app = create_app(runs_dir=runs_dir, default_run=run)
    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


def run_dashboard(run: str | None = None, port: int = 8765, open_browser: bool = False, **kwargs: Any) -> None:
    """CLI entry (``fly dashboard [--run NAME] [--port PORT]``)."""
    print(f"fly-chess dashboard: http://127.0.0.1:{port}/" + (f"  (run: {run})" if run else ""))
    serve(run=run, port=port, open_browser=open_browser, **kwargs)


__all__ = ["INIT_LINES", "POLL_INTERVAL", "STATIC_DIR", "create_app", "run_dashboard", "serve"]
