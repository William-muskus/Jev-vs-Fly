# Jev prompt experiments

Four ways of asking TypeSafe's Jev to pick a chess move, plus the previous
search-and-persona attempt.

This is **not** the fly. The fly still plays from the connectome (`web/engine`,
`game/`). These scripts only measure how Jev answers when code hands it the
legal-move list.

## The four strategies

Jev is a System One model: it returns a Choice over options you define, not a
thought-out variation. Chess.js generates the legal moves; each option is a UCI
key with a short English description. Jev never sees an illegal move.

| id | What we send |
|---|---|
| `best_this_turn` | one Choice: *which legal move is the best to play this turn?* |
| `best_win_rate` | one Choice: *which legal move most increases this side's chance of winning the game?* |
| `both_turn_then_win` | both questions in **one** request, `best_this_turn` key first |
| `both_win_then_turn` | the same two questions, `best_win_rate` key first |

Question IDs are not sent to the model (TypeSafe docs). The last two strategies
therefore differ only in JSON key order of the `questions` map. Questions in one
request are independent — they cannot see each other's answers — so if the two
wordings mean different things we should see different Choices, and if key order
is irrelevant the reversed pair should match.

The move we would *play* is the first question's Choice. That is a policy in
code, not something Jev decides.

## Previous approach (`v0/`)

`v0/` is the TypeScript player from the earlier Claude session: a 2-ply
material search shortlists ≤ 6 moves, describes them in English, then one Jev
request scores style (Choice + per-move Scores + a Noul). It never ran against
the live API in that session. Kept here as the contrast: **search, then ask**
versus **ask over the full legal list**.

## Run

```bash
export TYPESAFE_API_KEY=…          # never commit this
python -m experiments.compare --dry-run
python -m experiments.compare --out experiments/results
# or: fly jev-compare
```

Results land in `experiments/results/comparison.md` and `comparison.json`.
