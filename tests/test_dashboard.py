"""Tests for flychess.train.metrics (MetricsLogger, readers) and flychess.dashboard.server (HTTP + websocket)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from flychess.connectome.graph import toy_graph
from flychess.dashboard.server import _safe_name, create_app
from flychess.train.metrics import MetricsLogger, iter_lines_reversed, list_runs, read_metrics, read_run_json


# ------------------------------------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------------------------------------
def _fill_run(run_dir: Path, n_train: int = 50, graph_path: str | None = None) -> MetricsLogger:
    log = MetricsLogger(run_dir, rate_limits={"game": 0.0, "activity": 0.0})
    cfg = {"batch_size": 8, "stage": "imitation", "steps": 1000}
    if graph_path:
        cfg["graph_path"] = graph_path
    log.write_run_json(cfg, {"n": 200, "nnz": 2000}, extra={"graph_path": graph_path} if graph_path else None)
    log.log_status(0, message="starting", stage="imitation", total_steps=1000, eta_s=None)
    for step in range(0, n_train * 20, 20):
        log.log_train(
            step, loss=3.0 / (1 + step / 100), policy_loss=2.5 / (1 + step / 100), value_loss=0.5, top1=0.1,
            top3=0.3, lr=1e-3, pos_per_sec=5000.0, gpu_mem_gb=4.2, epoch=step / 500, stage="imitation",
        )
        if step % 200 == 0:
            log.log_eval(step, val_loss=2.8, val_top1=0.12, val_top3=0.31, val_value_mse=0.45)
    log.log_elo(400, opponent="random", games=20, wins=15, draws=3, losses=2, elo_estimate=210.5)
    log.log_game(400, pgn="1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0", result="1-0", moves=6, source="eval")
    log.log_activity(400, neuron_idx=np.arange(0, 16), values=np.random.default_rng(0).random(16))
    return log


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    log = _fill_run(root / "alpha")
    log.close()
    (root / "not-a-run").mkdir()  # directory without run.json / metrics.jsonl must be ignored
    return root


# ------------------------------------------------------------------------------------------------
# MetricsLogger
# ------------------------------------------------------------------------------------------------
def test_logger_writes_and_reads(tmp_path: Path) -> None:
    log = MetricsLogger(tmp_path / "r")
    rec = log.log_train(
        step=20, loss=1.5, policy_loss=1.0, value_loss=0.5, top1=np.float32(0.25), top3=0.5, lr=1e-3,
        pos_per_sec=1234.5, gpu_mem_gb=3.1, epoch=0.1, stage="imitation", extra_field=np.int64(7),
    )
    assert rec is not None and rec["kind"] == "train" and rec["step"] == 20
    log.log_eval(20, val_loss=1.4, val_top1=0.3, val_top3=0.6, val_value_mse=float("nan"))
    log.log_status(20, message="hello", stage="imitation", total_steps=100, eta_s=12.5)
    log.close()

    lines = (tmp_path / "r" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        d = json.loads(line)
        assert {"t", "step", "kind"} <= d.keys()
        assert abs(d["t"] - time.time()) < 60
    assert json.loads(lines[0])["top1"] == pytest.approx(0.25)
    assert json.loads(lines[0])["extra_field"] == 7
    assert json.loads(lines[1])["val_value_mse"] is None  # NaN -> null (browser JSON.parse safe)

    recs = read_metrics(tmp_path / "r")
    assert [r["kind"] for r in recs] == ["train", "eval", "status"]
    assert read_metrics(tmp_path / "r", kinds=["eval"])[0]["val_loss"] == 1.4
    assert [r["kind"] for r in read_metrics(tmp_path / "r", last_n=2)] == ["eval", "status"]
    assert [r["kind"] for r in read_metrics(tmp_path / "r", last_n=1, kinds=["train"])] == ["train"]
    assert read_metrics(tmp_path / "missing") == []


def test_logger_flushes_every_write(tmp_path: Path) -> None:
    log = MetricsLogger(tmp_path / "r")
    log.log("status", 1, message="x", stage="s", total_steps=1, eta_s=0)
    # visible to another reader *before* close
    assert len(read_metrics(tmp_path / "r")) == 1
    log.close()


def test_rate_limiting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1_000_000.0]
    monkeypatch.setattr("flychess.train.metrics.time.time", lambda: now[0])
    log = MetricsLogger(tmp_path / "r")
    assert log.log_game(1, pgn="1. e4 *", result="*", moves=1, source="selfplay") is True
    assert log.log_game(2, pgn="1. d4 *", result="*", moves=1, source="selfplay") is False  # < 60 s later
    now[0] += 59.0
    assert log.log_game(3, pgn="1. c4 *", result="*", moves=1, source="selfplay") is False
    now[0] += 1.5
    assert log.log_game(4, pgn="1. f4 *", result="*", moves=1, source="selfplay") is True

    assert log.log_activity(4, neuron_idx=[0, 1], values=[0.1, 0.2]) is True
    now[0] += 29.0
    assert log.log_activity(5, neuron_idx=[0, 1], values=[0.1, 0.2]) is False
    now[0] += 1.0
    assert log.log_activity(6, neuron_idx=[0, 1], values=[0.1, 0.2]) is True
    # non rate-limited kinds are never dropped
    for i in range(5):
        assert log.log_train(i, loss=1, policy_loss=1, value_loss=0, top1=0, top3=0, lr=0, pos_per_sec=0,
                             gpu_mem_gb=0, epoch=0, stage="s") is not None
    log.close()
    kinds = [r["kind"] for r in read_metrics(tmp_path / "r")]
    assert kinds.count("game") == 2 and kinds.count("activity") == 2 and kinds.count("train") == 5


def test_run_json_idempotent(tmp_path: Path) -> None:
    log = MetricsLogger(tmp_path / "r")
    first = log.write_run_json({"lr": 1e-3, "graph_path": "data/brain/tiny.npz"}, {"n": 10}, extra={"note": "a"})
    assert first["started_at"] > 0 and first["graph_path"] == "data/brain/tiny.npz" and first["note"] == "a"
    time.sleep(0.01)
    second = log.write_run_json({"lr": 2e-3}, {"n": 11})
    assert second["started_at"] == first["started_at"]
    assert second["config"] == {"lr": 2e-3} and second["graph_meta"] == {"n": 11}
    assert second["graph_path"] == "data/brain/tiny.npz"  # preserved from the first write
    assert read_run_json(tmp_path / "r")["started_at"] == first["started_at"]
    log.close()


def test_iter_lines_reversed_and_torn_line(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    p.write_bytes(b"".join(json.dumps({"i": i, "pad": "x" * 50}).encode() + b"\n" for i in range(3000)))
    with open(p, "ab") as f:
        f.write(b'{"i": 3000, "kind": "tr')  # writer mid-line
    rev = [json.loads(ln)["i"] for ln in iter_lines_reversed(p, block_size=1000) if ln.endswith(b"}")]
    assert rev == list(range(2999, -1, -1))
    (tmp_path / "run").mkdir()
    p.rename(tmp_path / "run" / "metrics.jsonl")
    recs = read_metrics(tmp_path / "run", last_n=5)
    assert [r["i"] for r in recs] == [2995, 2996, 2997, 2998, 2999]


def test_list_runs(runs_dir: Path) -> None:
    runs = list_runs(runs_dir)
    assert [r["name"] for r in runs] == ["alpha"]
    r = runs[0]
    assert r["started_at"] > 0 and r["last_t"] > 0
    assert r["last_step"] == 400 and r["stage"] == "imitation"
    # run with only a metrics file and no run.json
    log = MetricsLogger(runs_dir / "beta")
    log.log_status(7, message="m", stage="selfplay", total_steps=10, eta_s=1)
    log.close()
    runs = list_runs(runs_dir)
    assert [r["name"] for r in runs] == ["beta", "alpha"]  # newest activity first
    assert runs[0]["started_at"] is None and runs[0]["stage"] == "selfplay" and runs[0]["last_step"] == 7
    assert list_runs(runs_dir / "does-not-exist") == []


# ------------------------------------------------------------------------------------------------
# server
# ------------------------------------------------------------------------------------------------
def test_http_endpoints(runs_dir: Path) -> None:
    app = create_app(runs_dir=runs_dir, default_run="alpha")
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200 and "fly-chess" in r.text and "dashboard.js" in r.text
        assert client.get("/static/dashboard.js").status_code == 200
        assert client.get("/static/dashboard.css").status_code == 200

        assert client.get("/api/default").json()["run"] == "alpha"

        runs = client.get("/api/runs").json()
        assert [x["name"] for x in runs] == ["alpha"] and runs[0]["last_step"] == 400

        data = client.get("/api/run/alpha").json()
        assert data["run"]["config"]["batch_size"] == 8 and data["run"]["graph_meta"]["n"] == 200
        kinds = {m["kind"] for m in data["metrics"]}
        assert kinds == {"status", "train", "eval", "elo", "game", "activity"}
        assert data["metrics"][0]["kind"] == "status" and data["metrics"][-1]["kind"] == "activity"

        assert client.get("/api/run/nope").status_code == 404
        assert client.get("/api/run/.hidden").status_code == 400
    for bad in ("..", ".", "a/b", "a\\b", ""):
        with pytest.raises(HTTPException):
            _safe_name(bad)

        # no graph_path in run.json -> positions unavailable but well-formed
        assert client.get("/api/run/alpha/positions").json()["available"] is False


def test_api_run_caps_at_5000_lines(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    log = MetricsLogger(root / "big")
    for i in range(5200):
        log.log("train", i, loss=1.0)
    log.close()
    with TestClient(create_app(runs_dir=root)) as client:
        m = client.get("/api/run/big").json()["metrics"]
        assert len(m) == 5000 and m[0]["step"] == 200 and m[-1]["step"] == 5199


def test_positions_from_graph(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    g = toy_graph(n=120, nnz=600, seed=1)
    g.position[5] = np.nan  # unknown position -> null
    gp = tmp_path / "toy.npz"
    g.save(gp)
    log = _fill_run(root / "g", n_train=2, graph_path=str(gp))
    log.close()
    with TestClient(create_app(runs_dir=root)) as client:
        j = client.get("/api/run/g/positions").json()
        assert j["available"] is True and j["n"] == 120
        assert j["neuron_idx"] == list(range(16))  # from the latest activity sample
        assert j["positions"][5] is None
        for p in j["positions"]:
            if p is not None:
                assert len(p) == 3 and all(0.0 <= v <= 1.0 for v in p)
        assert j["super_class"] == ["central"] * 16


def test_websocket_init_and_tail(runs_dir: Path) -> None:
    app = create_app(runs_dir=runs_dir)
    with TestClient(app) as client, client.websocket_connect("/ws/alpha") as ws:
        init = ws.receive_json()
        assert init["kind"] == "init"
        assert init["run"]["config"]["batch_size"] == 8
        assert len(init["metrics"]) > 10 and init["metrics"][-1]["kind"] == "activity"

        log = MetricsLogger(runs_dir / "alpha")
        log.log_status(999, message="tailed", stage="selfplay", total_steps=2000, eta_s=3.0)
        log.close()
        msg = ws.receive_json()  # pushed within ~POLL_INTERVAL
        assert msg["kind"] == "status" and msg["step"] == 999 and msg["message"] == "tailed"


def test_websocket_waits_for_run_to_appear(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    app = create_app(runs_dir=root)
    with TestClient(app) as client, client.websocket_connect("/ws/later") as ws:
        time.sleep(0.7)  # run does not exist yet: the server must simply wait
        log = MetricsLogger(root / "later")
        log.write_run_json({"x": 1}, {})
        log.log_status(0, message="born", stage="imitation", total_steps=5, eta_s=None)
        init = ws.receive_json()
        assert init["kind"] == "init" and init["run"]["config"] == {"x": 1}
        assert [m["message"] for m in init["metrics"]] == ["born"]
        log.log_status(1, message="next", stage="imitation", total_steps=5, eta_s=None)
        log.close()
        assert ws.receive_json()["message"] == "next"
