"""Jev vs Fly: FastAPI app that serves the match UI and proxies Jev + the fly brain.

The fly still plays in the browser (the original Web Worker / connectome blob).
Jev's move is a server-side TypeSafe Choice over the legal moves — the API key
never leaves this process.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from experiments.prompts import STRATEGIES, pick_move

REPO = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
WEB = REPO / "web"
CACHE = Path(__file__).resolve().parent / ".model-cache"
HF_MODEL_BASE = "https://huggingface.co/cesp99/fly-chess/resolve/main/web/"
MODEL_FILES = ("brain.json", "brain.flyb", "brain.flyb.gz")

_cache_lock = threading.Lock()


class MoveRequest(BaseModel):
    fen: str
    strategy: str = "best_this_turn"


def _ensure_model_file(name: str, model_dir: Path) -> Path:
    if name not in MODEL_FILES:
        raise HTTPException(404, f"unknown model file {name}")
    dest = model_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with _cache_lock:
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        model_dir.mkdir(parents=True, exist_ok=True)
        url = HF_MODEL_BASE + name
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            import requests
        except ImportError as e:  # pragma: no cover
            raise HTTPException(503, "requests is required to download the fly brain") from e
        try:
            r = requests.get(url, timeout=120, allow_redirects=True)
            if r.status_code == 404 and name == "brain.flyb":
                raise HTTPException(404, "no uncompressed blob; try brain.flyb.gz")
            r.raise_for_status()
            tmp.write_bytes(r.content)
            tmp.replace(dest)
        except HTTPException:
            raise
        except Exception as e:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise HTTPException(502, f"could not fetch {url}: {e}") from e
        return dest


def create_app(
    *,
    system_one: Callable[[Any, Mapping[str, Any]], Any] | None = None,
    model_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Jev vs Fly", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    model_dir = Path(model_dir) if model_dir is not None else CACHE

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "strategies": list(STRATEGIES)}

    @app.get("/api/strategies")
    def strategies() -> dict[str, Any]:
        return {
            "strategies": list(STRATEGIES),
            "default": "best_this_turn",
            "notes": {
                "best_this_turn": "One Choice: the best move this turn.",
                "best_win_rate": "One Choice: the move that maximises game win rate.",
                "both_turn_then_win": "Both questions in one call; this-turn key first. Plays the this-turn answer.",
                "both_win_then_turn": "Both questions in one call; win-rate key first. Plays the win-rate answer.",
            },
        }

    @app.post("/api/jev-move")
    def jev_move(req: MoveRequest) -> JSONResponse:
        if req.strategy not in STRATEGIES:
            raise HTTPException(400, f"unknown strategy {req.strategy!r}")
        try:
            board = chess.Board(req.fen)
        except ValueError as e:
            raise HTTPException(400, f"bad FEN: {e}") from e
        if board.is_game_over():
            raise HTTPException(400, "game is over")
        try:
            pick = pick_move(board, req.strategy, system_one=system_one)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:
            raise HTTPException(502, f"Jev request failed: {e}") from e
        return JSONResponse(pick.to_json())

    @app.get("/model/{name}")
    def model_file(name: str) -> FileResponse:
        path = _ensure_model_file(name, model_dir)
        media = {
            "brain.json": "application/json",
            "brain.flyb": "application/octet-stream",
            "brain.flyb.gz": "application/gzip",
        }[name]
        return FileResponse(path, media_type=media, headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/app.js")
    def app_js() -> FileResponse:
        return FileResponse(STATIC / "app.js", media_type="text/javascript")

    @app.get("/style.css")
    def style_css() -> FileResponse:
        return FileResponse(STATIC / "style.css", media_type="text/css")

    @app.get("/board.js")
    def board_js() -> FileResponse:
        return FileResponse(WEB / "board.js", media_type="text/javascript")

    if (WEB / "engine").is_dir():
        app.mount("/engine", StaticFiles(directory=str(WEB / "engine")), name="engine")
    if (WEB / "vendor").is_dir():
        app.mount("/vendor", StaticFiles(directory=str(WEB / "vendor")), name="vendor")
    if (WEB / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=str(WEB / "assets")), name="assets")

    return app


app = create_app()


def serve(host: str = "127.0.0.1", port: int = 8766) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=int(port), log_level="info")


if __name__ == "__main__":
    serve()
