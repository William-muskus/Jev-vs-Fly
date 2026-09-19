"""Run the four Jev prompt strategies on the curated positions and write a comparison.

    python -m experiments.compare
    python -m experiments.compare --out experiments/results
    TYPESAFE_API_KEY=… python -m experiments.compare --live
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.positions import POSITIONS, Position
from experiments.prompts import STRATEGIES, Pick, pick_move

REPO = Path(__file__).resolve().parent.parent


def compare_position(pos: Position, *, system_one=None) -> dict[str, Any]:
    board = pos.board()
    picks: dict[str, Pick] = {}
    for strategy in STRATEGIES:
        picks[strategy] = pick_move(board, strategy, system_one=system_one)

    solo_turn = picks["best_this_turn"].uci
    solo_win = picks["best_win_rate"].uci
    both = picks["both_turn_then_win"]
    both_rev = picks["both_win_then_turn"]
    both_turn = both.answers.get("best_this_turn")
    both_win = both.answers.get("best_win_rate")
    rev_turn = both_rev.answers.get("best_this_turn")
    rev_win = both_rev.answers.get("best_win_rate")

    return {
        "id": pos.id,
        "title": pos.title,
        "why": pos.why,
        "fen": board.fen(),
        "legal_count": len(list(board.legal_moves)),
        "picks": {k: v.to_json() for k, v in picks.items()},
        "flags": {
            "solo_turn_equals_solo_win": solo_turn == solo_win,
            "both_questions_agree": (
                both_turn.choice == both_win.choice if both_turn and both_win else None
            ),
            "reversed_questions_agree": (
                rev_turn.choice == rev_win.choice if rev_turn and rev_win else None
            ),
            "turn_answer_flipped_by_key_order": (
                both_turn.choice != rev_turn.choice if both_turn and rev_turn else None
            ),
            "win_answer_flipped_by_key_order": (
                both_win.choice != rev_win.choice if both_win and rev_win else None
            ),
            "solo_turn_equals_batched_turn": (
                solo_turn == both_turn.choice if both_turn else None
            ),
            "solo_win_equals_batched_win": (
                solo_win == both_win.choice if both_win else None
            ),
        },
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for k, v in row["flags"].items():
            if v is True:
                counts[k] += 1
            elif v is False:
                counts[k + "_false"] += 0  # keep key present
    agreement = {
        k: {"yes": counts[k], "n": n, "rate": (counts[k] / n if n else None)}
        for k in (
            "solo_turn_equals_solo_win",
            "both_questions_agree",
            "reversed_questions_agree",
            "turn_answer_flipped_by_key_order",
            "win_answer_flipped_by_key_order",
            "solo_turn_equals_batched_turn",
            "solo_win_equals_batched_win",
        )
    }
    tokens = 0
    latency = 0.0
    calls = 0
    for row in rows:
        for pick in row["picks"].values():
            if pick.get("skipped_api"):
                continue
            calls += 1
            latency += float(pick.get("latency_s") or 0)
            u = pick.get("usage") or {}
            tokens += int(u.get("input_tokens") or 0) + int(u.get("output_tokens") or 0)
    return {
        "positions": n,
        "api_calls": calls,
        "mean_latency_s": (latency / calls) if calls else None,
        "total_tokens": tokens,
        "agreement": agreement,
    }


def _pick_cell(pick_json: dict[str, Any], question: str) -> str:
    ans = (pick_json.get("answers") or {}).get(question)
    if not ans:
        return str(pick_json.get("san") or "—")
    top = ans.get("top") or [[ans.get("choice"), 0.0]]
    p = float(top[0][1]) if top else 0.0
    return f"{ans['choice']} ({p:.2f})"


def render_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    mean = summary.get("mean_latency_s")
    if mean is None:
        header = f"Ran {summary['positions']} positions."
    else:
        header = (
            f"Ran {summary['positions']} positions, {summary['api_calls']} TypeSafe calls, "
            f"{summary['total_tokens']} tokens, mean latency {mean:.2f}s."
        )
    lines = [
        "# Jev prompt-strategy comparison",
        "",
        header,
        "",
        "## What we asked",
        "",
        "1. `best_this_turn` — one Choice: pick the best move this turn.",
        "2. `best_win_rate` — one Choice: pick the move that maximises game win rate.",
        "3. `both_turn_then_win` — both questions in one call, this-turn key first.",
        "4. `both_win_then_turn` — the same two questions, win-rate key first.",
        "",
        (
            "Question IDs are not sent to Jev. Key order only changes the JSON serialisation "
            "of the `questions` map. The move we would play is the first question's Choice."
        ),
        "",
        "## Per position",
        "",
        "| position | legal | this turn | win rate | both (turn) | both (win) | both-rev (win) | both-rev (turn) | solo agree? | both agree? | order flipped turn? |",
        "|---|---:|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        p = row["picks"]
        f = row["flags"]
        lines.append(
            "| {title} | {n} | {t} | {w} | {bt} | {bw} | {rw} | {rt} | {sa} | {ba} | {of} |".format(
                title=row["title"],
                n=row["legal_count"],
                t=_pick_cell(p["best_this_turn"], "best_this_turn"),
                w=_pick_cell(p["best_win_rate"], "best_win_rate"),
                bt=_pick_cell(p["both_turn_then_win"], "best_this_turn"),
                bw=_pick_cell(p["both_turn_then_win"], "best_win_rate"),
                rw=_pick_cell(p["both_win_then_turn"], "best_win_rate"),
                rt=_pick_cell(p["both_win_then_turn"], "best_this_turn"),
                sa="yes" if f["solo_turn_equals_solo_win"] else "no",
                ba="yes" if f["both_questions_agree"] else "no",
                of="yes" if f["turn_answer_flipped_by_key_order"] else "no",
            )
        )

    lines += ["", "## Agreement rates", "", "| check | yes / n | rate |", "|---|---|---|"]
    labels = {
        "solo_turn_equals_solo_win": "Solo this-turn == solo win-rate",
        "both_questions_agree": "Batched this-turn == batched win-rate",
        "reversed_questions_agree": "Reversed batched questions agree with each other",
        "turn_answer_flipped_by_key_order": "This-turn answer changed when key order flipped",
        "win_answer_flipped_by_key_order": "Win-rate answer changed when key order flipped",
        "solo_turn_equals_batched_turn": "Solo this-turn == batched this-turn",
        "solo_win_equals_batched_win": "Solo win-rate == batched win-rate",
    }
    for key, label in labels.items():
        a = summary["agreement"][key]
        rate = "—" if a["rate"] is None else f"{100 * a['rate']:.0f}%"
        lines.append(f"| {label} | {a['yes']} / {a['n']} | {rate} |")

    lines += ["", "## Notes", ""]
    for row in rows:
        lines += [f"### {row['title']}", "", row["why"], "", f"`{row['fen']}`", ""]
    return "\n".join(lines) + "\n"


def run(
    positions: Iterable[Position] = POSITIONS,
    *,
    system_one=None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    rows = [compare_position(p, system_one=system_one) for p in positions]
    summary = summarise(rows)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "strategies": list(STRATEGIES),
        "summary": summary,
        "positions": rows,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (out_dir / "comparison.md").write_text(render_markdown(rows, summary), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare four Jev chess-move prompt strategies.")
    p.add_argument("--out", default=str(REPO / "experiments" / "results"), help="directory for comparison.json/md")
    p.add_argument("--ids", default=None, help="comma-separated position ids (default: all)")
    p.add_argument("--dry-run", action="store_true", help="build requests only, do not call TypeSafe")
    args = p.parse_args(argv)

    positions = list(POSITIONS)
    if args.ids:
        want = {s.strip() for s in args.ids.split(",") if s.strip()}
        positions = [p for p in positions if p.id in want]
        missing = want - {p.id for p in positions}
        if missing:
            print(f"unknown position ids: {sorted(missing)}", file=sys.stderr)
            return 2

    if args.dry_run:
        from experiments.board_state import criteria_from_moves, legal_moves, position_state
        from experiments.prompts import questions_for

        for pos in positions:
            board = pos.board()
            moves = legal_moves(board)
            print(f"\n# {pos.id}  {pos.title}  ({len(moves)} options)")
            for s in STRATEGIES:
                q = questions_for(s, criteria_from_moves(moves))
                print(f"  {s}: keys={list(q)}")
            print("  state keys:", list(position_state(board)))
        return 0

    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set. Export it or pass --dry-run.", file=sys.stderr)
        return 2

    out = Path(args.out)
    payload = run(positions, out_dir=out)
    print(render_markdown(payload["positions"], payload["summary"]))
    print(f"\nwrote {out / 'comparison.md'} and {out / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
