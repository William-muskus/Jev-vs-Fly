"""Tests for the Jev vs Fly game server (mocked Jev, no fly blob required)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from game.server import create_app
from tests.test_jev_experiments import Scripted


def client() -> TestClient:
    return TestClient(create_app(system_one=Scripted("e2e4")))


def test_health_and_strategies():
    c = client()
    h = c.get("/health")
    assert h.status_code == 200 and h.json()["ok"] is True
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
