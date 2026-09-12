"""Tests for flychess.data: PGN filtering/parsing, shard writing/reading, ShardDataset."""
from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from flychess.chessenv.encoding import encode_board, move_to_index
from flychess.data.lichess import (
    GameFilter,
    build_shards,
    iter_game_texts,
    parse_headers,
    positions_from_pgn,
    process_block,
)
from flychess.data.shards import (
    ShardDataset,
    collate,
    count_positions,
    list_shards,
    load_split,
    read_shard,
    write_shard,
)
from flychess.paths import PGN_DIR

FIXTURE = Path(__file__).parent / "fixtures" / "small.pgn"
REAL_DUMP = PGN_DIR / "lichess_db_standard_rated_2014-01.pgn.zst"


def _headers(**kw) -> dict[str, str]:
    h = {"WhiteElo": "1900", "BlackElo": "1850", "Result": "1-0", "TimeControl": "300+0"}
    h.update(kw)
    return h


# ---------------------------------------------------------------------------------------------
# filtering / parsing
# ---------------------------------------------------------------------------------------------
def test_game_filter():
    f = GameFilter()
    assert f.accepts(_headers())
    assert not f.accepts(_headers(WhiteElo="1799"))
    assert not f.accepts(_headers(BlackElo="?"))
    assert not f.accepts(_headers(Result="*"))
    assert not f.accepts(_headers(TimeControl="60+0"))
    assert not f.accepts(_headers(TimeControl="120+1"))
    assert f.accepts(_headers(TimeControl="180+0"))
    assert f.accepts(_headers(TimeControl="-"))
    assert not GameFilter(allow_unknown_time=False).accepts(_headers(TimeControl="-"))
    assert not f.accepts(_headers(Variant="Chess960"))
    assert not f.accepts(_headers(SetUp="1", FEN="8/8/8/8/8/8/8/8 w - - 0 1"))
    assert GameFilter(min_elo=1500).accepts(_headers(WhiteElo="1600", BlackElo="1550"))
    assert not GameFilter(max_elo=2000).accepts(_headers(WhiteElo="2100"))
    assert GameFilter(results=("1-0",)).accepts(_headers()) and not GameFilter(results=("0-1",)).accepts(_headers())


def test_fixture_and_header_parsing():
    games = list(iter_game_texts(FIXTURE))
    assert len(games) == 40
    heads = [parse_headers(g) for g in games]
    assert all({"WhiteElo", "BlackElo", "Result", "TimeControl", "Site"} <= set(h) for h in heads)
    kept = [h for h in heads if GameFilter().accepts(h)]
    assert 20 <= len(kept) < 40
    assert any(h["Result"] == "1/2-1/2" for h in kept)
    # also works on an in-memory text stream and on chunk boundaries inside a game
    with open(FIXTURE) as f:
        text = f.read()
    for chunk in (7, 1000, 1 << 20):
        assert list(iter_game_texts(io.StringIO(text), chunk_chars=chunk)) == games


def test_positions_from_pgn_matches_reference_encoding():
    filt = GameFilter()
    positions = list(positions_from_pgn(FIXTURE, filt))
    assert len(positions) > 1000
    # independent reference: replay every kept game with python-chess
    ref = []
    for text in iter_game_texts(FIXTURE):
        h = parse_headers(text)
        if not filt.accepts(h):
            continue
        game = chess.pgn.read_game(io.StringIO(text))
        white_value = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[h["Result"]]
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= filt.skip_openings:
                mover_white = board.turn == chess.WHITE
                planes = encode_board(board)  # uses board.is_repetition on the real move stack
                ref.append((planes, move_to_index(move, board), white_value if mover_white else -white_value,
                            int(h["WhiteElo"] if mover_white else h["BlackElo"]), ply, board.fen(), move.uci()))
            board.push(move)
    assert len(ref) == len(positions)
    n_rep = 0
    for pos, (planes, move, value, elo, ply, fen, uci) in zip(positions, ref):
        assert pos.planes.dtype == np.uint8 and pos.planes.shape == (20, 8, 8)
        assert (pos.move, pos.value, pos.elo, pos.ply, pos.fen, pos.uci) == (move, value, elo, ply, fen, uci)
        assert np.array_equal(pos.planes[:18], planes[:18]) and np.array_equal(pos.planes[19], planes[19])
        halfmove = int(fen.split()[4])
        assert pos.planes[18, 0, 0] == (min(halfmove, 100) * 255 + 50) // 100
        n_rep += int(pos.planes[19, 0, 0])
    assert min(p.ply for p in positions) == filt.skip_openings
    assert {p.value for p in positions} == {-1, 0, 1}
    assert n_rep > 0  # the fixture contains a repeated position


def test_process_block_stats():
    with open(FIXTURE) as f:
        arrays, stats = process_block(f.read(), GameFilter())
    assert stats["games_seen"] == 40
    assert stats["games_kept"] + stats["games_short"] + stats["games_error"] < 40
    assert stats["positions"] == len(arrays.move) == len(list(positions_from_pgn(FIXTURE)))
    assert arrays.feats.shape == (stats["positions"], 19) and arrays.feats.dtype == np.uint64
    assert arrays.move.dtype == np.int16 and arrays.value.dtype == np.int8


# ---------------------------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------------------------
def test_write_read_shard_roundtrip(tmp_path):
    n = 10
    planes = (np.random.default_rng(0).random((n, 20, 8, 8)) > 0.5).astype(np.uint8)
    arrays = {"planes": planes, "move": np.arange(n, dtype=np.int16), "value": np.array([1, -1, 0] * 3 + [1], np.int8),
              "elo": np.full(n, 1800, np.int16), "ply": np.arange(n, dtype=np.int16) + 4}
    path = write_shard(tmp_path / "x-00000.npz", **arrays)
    back = read_shard(path)
    for k, v in arrays.items():
        assert np.array_equal(back[k], v) and back[k].dtype == v.dtype
    assert count_positions(tmp_path) == n and list_shards(tmp_path) == [path]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("shards")
    stats = build_shards(FIXTURE, out, name="fix", workers=2, shard_size=500, shuffle_buffer=700,
                         chunk_chars=4000, progress=False)
    return out, stats


def test_build_shards_matches_generator(built):
    out, stats = built
    positions = list(positions_from_pgn(FIXTURE))
    assert stats.games_seen == 40 and stats.positions == len(positions) and stats.games_error == 0
    files = list_shards(out, "fix")
    assert len(files) == stats.shards == -(-len(positions) // 500)
    assert [f.name for f in files] == [f"fix-{i:05d}.npz" for i in range(len(files))]
    assert count_positions(out) == len(positions)
    shards = [read_shard(f) for f in files]
    assert all(len(s["move"]) == 500 for s in shards[:-1]) and 0 < len(shards[-1]["move"]) <= 500
    # same multiset of examples, in a different (shuffled) order
    got = Counter()
    for s in shards:
        for i in range(len(s["move"])):
            got[(s["planes"][i].tobytes(), int(s["move"][i]), int(s["value"][i]), int(s["elo"][i]), int(s["ply"][i]))] += 1
    want = Counter((p.planes.tobytes(), p.move, p.value, p.elo, p.ply) for p in positions)
    assert got == want
    first = [int(m) for s in shards for m in s["ply"]][:200]
    assert first != [p.ply for p in positions[:200]]
    assert "positions" in stats.summary() and stats.to_dict()["shards"] == len(files)


def test_build_shards_limits(tmp_path):
    stats = build_shards(FIXTURE, tmp_path, name="lim", workers=1, max_games=5, progress=False)
    assert stats.games_seen == 5 and 0 < stats.positions == count_positions(tmp_path)
    stats = build_shards(FIXTURE, tmp_path / "p", name="lim", workers=1, max_positions=100, shard_size=50,
                         chunk_chars=2000, progress=False)
    assert 100 <= stats.positions < len(list(positions_from_pgn(FIXTURE)))
    assert build_shards(FIXTURE, tmp_path / "e", name="e", workers=1, min_elo=3000, progress=False).positions == 0


def test_shard_dataset_and_collate(built):
    out, stats = built
    files = list_shards(out, "fix")
    ds = ShardDataset(files, seed=1, shuffle=True)
    assert len(ds) == stats.positions
    order0 = [x["move"] for x in ds]
    ds.set_epoch(1)
    order1 = [x["move"] for x in ds]
    assert len(order0) == len(order1) == stats.positions and order0 != order1
    assert Counter(order0) == Counter(order1)
    plain = ShardDataset(files, shuffle=False)
    seq = [x["ply"] for x in plain]
    assert seq == [int(p) for f in files for p in read_shard(f)["ply"]]

    loader = DataLoader(ds, batch_size=64, collate_fn=collate, num_workers=2)
    n, moves = 0, Counter()
    for batch in loader:
        assert batch["planes"].shape[1] == 1280 and batch["planes"].dtype == torch.float32
        assert batch["move"].dtype == torch.int64 and batch["value"].dtype == torch.float32
        assert batch["elo"].dtype == torch.int64 and batch["ply"].dtype == torch.int64
        p = batch["planes"].view(-1, 20, 8, 8)
        assert p[:, 17].min() == 1 and 0 <= p[:, 18].min() and p[:, 18].max() <= 1
        assert set(batch["value"].unique().tolist()) <= {-1.0, 0.0, 1.0}
        assert (batch["move"] < 4168).all() and (batch["move"] >= 0).all()
        n += len(batch["move"])
        moves.update(batch["move"].tolist())
    assert n == stats.positions and moves == Counter(order0)  # every shard visited exactly once across workers


def test_load_split(built):
    out, _ = built
    train, val = load_split(out, val_fraction=0.25, seed=3)
    assert train and val and not set(train) & set(val)
    assert sorted(train + val) == list_shards(out)
    assert (train, val) == load_split(out, val_fraction=0.25, seed=3)
    assert load_split(out, val_fraction=0.25, seed=4) != (train, val) or len(list_shards(out)) < 3
    assert load_split(out, val_fraction=0.0) == (list_shards(out), [])
    with pytest.raises(FileNotFoundError):
        load_split(out / "nothing")


@pytest.mark.slow
@pytest.mark.skipif(not REAL_DUMP.exists(), reason="real Lichess dump not downloaded")
def test_build_shards_real_dump_smoke(tmp_path):
    stats = build_shards(REAL_DUMP, tmp_path, name="smoke", workers=8, max_games=20000, progress=False)
    print(stats.summary())
    assert stats.games_seen == 20000 and stats.games_kept > 500 and stats.positions > 30000
    assert count_positions(tmp_path) == stats.positions
