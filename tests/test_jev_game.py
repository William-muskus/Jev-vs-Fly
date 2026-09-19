"""Tests for the Jev vs Fly game server (mocked Jev, no fly blob required)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from game.server import create_app, occupy_message, parse_args
from tests.test_jev_experiments import Scripted


def client() -> TestClient:
    return TestClient(create_app(system_one=Scripted("e2e4")))


def test_health_and_strategies():
    c = client()
    h = c.get("/health")
    assert h.status_code == 200 and h.json()["ok"] is True
    assert "jev_configured" in h.json()
    assert "best_this_turn" in h.json()["strategies"]
    s = c.get("/api/strategies").json()
    assert "best_this_turn" in s["strategies"]
    assert s["default"] == "best_this_turn"


def test_jev_move_from_startpos():
    c = client()
    r = c.post("/api/jev-move", json={"fen": "startpos", "strategy": "best_this_turn"})
    # python-chess does not accept "startpos" as FEN
    assert r.status_code == 400
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    r = c.post("/api/jev-move", json={"fen": start, "strategy": "best_this_turn"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["uci"] == "e2e4" and body["san"] == "e4"
    assert body["played_question"] == "best_this_turn"


def test_unknown_strategy_and_game_over():
    c = client()
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert c.post("/api/jev-move", json={"fen": start, "strategy": "nope"}).status_code == 400
    # Fool's mate: after Qh4# it is white to move and already mated.
    fools = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    r = c.post("/api/jev-move", json={"fen": fools, "strategy": "best_this_turn"})
    assert r.status_code == 400


def test_index_and_static_js():
    c = client()
    html = c.get("/")
    assert html.status_code == 200 and b"Jev vs Fly" in html.content
    js = c.get("/app.js")
    assert js.status_code == 200 and b"jev-move" in js.content
    css = c.get("/style.css")
    assert css.status_code == 200
    board = c.get("/board.js")
    assert board.status_code == 200 and b"export class Board" in board.content


def test_cors_allows_the_vite_hall():
    c = client()
    r = c.get("/health", headers={"Origin": "http://127.0.0.1:8080"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"


def test_wizard_models_mount_is_served():
    c = client()
    r = c.get("/models/wizard/README.txt")
    assert r.status_code == 200, r.text
    assert b"not shipped" in r.content.lower() or b"Cults" in r.content


def test_parse_serve_args(monkeypatch):
    monkeypatch.delenv("JEV_FLY_PORT", raising=False)
    monkeypatch.delenv("JEV_FLY_HOST", raising=False)
    args = parse_args(["--port", "9001", "--host", "0.0.0.0"])
    assert args.port == 9001 and args.host == "0.0.0.0"
    assert parse_args([]).port == 8766


def test_game_record_writes_pgn(tmp_path: Path):
    dest = tmp_path / "records"
    c = TestClient(create_app(system_one=Scripted("e2e4"), records_dir=dest))
    r = c.post(
        "/api/game-record",
        json={
            "pgn": "1. e4 c5 2. d4",
            "result": "1/2-1/2",
            "white": "Jev",
            "black": "Fruit Fly",
            "reason": "plycap",
        },
    )
    assert r.status_code == 200, r.text
    text = (dest / "latest.pgn").read_text(encoding="utf-8")
    assert '[White "Jev"]' in text
    assert '[Black "Fruit Fly"]' in text
    assert "1. e4 c5 2. d4" in text
    assert '[White "?"]' not in text
    nested = c.post(
        "/api/game-record",
        json={
            "pgn": '[White "?"]\n\n1. e4 d5 2. exd5',
            "white": "Jev",
            "black": "Fruit Fly",
            "result": "1/2-1/2",
        },
    )
    assert nested.status_code == 200
    nested_text = (dest / "latest.pgn").read_text(encoding="utf-8")
    assert '[White "Jev"]' in nested_text
    assert '[Black "Fruit Fly"]' in nested_text
    assert '[White "?"]' not in nested_text
    empty = c.post("/api/game-record", json={"pgn": "  "})
    assert empty.status_code == 400


def test_occupy_message_already_running():
    msg = occupy_message("127.0.0.1", 8766, {"ok": True})
    assert "already running" in msg
    assert "npm run dev" in msg
    busy = occupy_message("127.0.0.1", 8766, None)
    assert "10048" in busy
    assert "Get-NetTCPConnection" in busy


def test_serve_exits_when_the_port_is_taken(monkeypatch, capsys):
    from game import server

    monkeypatch.setattr(server, "probe_health", lambda host, port: {"ok": True})
    monkeypatch.setattr(server, "port_held", lambda host, port: True)
    with pytest.raises(SystemExit) as caught:
        server.serve(host="127.0.0.1", port=8766)
    assert caught.value.code == 1
    err = capsys.readouterr().err
    assert "already running" in err
