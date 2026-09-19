# jev-chess

A chess player whose search is code and whose taste is Jev.

## The constraint that shapes everything

Chess is close to the worst case for a System One model. TypeSafe's own jaggedness
page for `jev-1.13` names numeric precision, counting, multi-hop indirection and
literal reading as failure modes, and tells you to avoid System Two tasks. Search is
the System Two task. Give Jev a FEN and ask for the best move and you get a
confident-looking distribution over a weak guess, with no way to tell which.

So Jev is never asked to calculate. It is asked what a move means and whether that
meaning matches an intention. Everything a calculator should do is done by a
calculator.

```
chess.js            legal moves, make, unmake, terminal states
negamax + quiesce   material and placement, 2 plies plus captures
shortlist           <= 6 moves, all within marginCp of the best
describe()          each surviving move rendered into English
                    ---- only here does anything leave the machine ----
Jev, one request    1 Choice over the shortlist
                    2 Scores per candidate: attack, self-weakening
                    1 Noul: is my king under real pressure
composite scoring   persona weights, in code
confidence gate     below threshold, defer to the search
```

The division is the design. If the shortlist ever contains a blunder, no amount of
good judgment above it helps, so the shortlist is the part that gets tested.

## What Jev is actually contributing

Not strength. A shallow search with piece-square tables is the floor, and Jev cannot
raise it, because raising it would mean calculating.

What it gives you is a player with a *described* style that you change by editing
prose. `personas.ts` holds four. The Romantic will take a 240-centipawn shortlist
margin and spend it on lines towards the king. The Solid persona takes 60 and spends
it on not creating weaknesses. Same search, same candidates, different game. There is
no training step and no evaluation function to hand-tune.

The second thing you get is a legible reason. Every move prints the distribution it
came from:

```
  move     loss    fit  attack  loose   total
  Bxf7+     180   0.51    0.92   0.41   1.012 <
  O-O         0   0.19    0.05   0.02   0.207
  Nc3         0   0.14    0.18   0.04   0.243
```

`loss` is the search's. `fit`, `attack` and `loose` are Jev's raw judgments,
untouched. `total` is policy, applied in code. Changing a persona weight rescores the
table without another request, because the evidence and the questions have not
changed.

## Setup

```sh
npm install
export TYPESAFE_API_KEY=...        # never committed, never hardcoded
npm run check                      # typecheck
npm run probe                      # offline tests, no key needed
npm run play                       # you are black against the Romantic
```

Other modes:

```sh
npm run play -- --persona solid --side white
npm run play -- --self romantic:solid --moves 40
npm run play -- --judge --fen "<fen>" --persona hustler
```

Without `TYPESAFE_API_KEY` it runs on the search alone and says so. That is also the
control condition: play the same opponent against `neutral` with and without a key to
see what the judgment layer is actually worth.

## Measured

On this container, chess.js 1.4.0, node 22.

`chess.moves({ verbose: true })` costs about **2.5ms per call**, roughly 15x plain
`moves()`, because it builds a before-and-after FEN for every move. The search runs on
SAN strings and only takes verbose moves at the root. That one change was most of the
speedup.

Search time, worst of four positions, shortlist of six:

| depth | worst | top three candidates |
| --- | --- | --- |
| 2 | 1.08s | same as depth 3 in 3 of 4 positions |
| 3 | 15.0s | differs only in the queen's gambit position |

Depth 3 costs 12x for almost nothing, so depth 2 is the default. The real ceiling is
chess.js at roughly 6k nodes/sec. If you want depth 4 and up, the library has to
change before anything else does.

`npm run probe` asserts six things about the shortlist, all passing at depth 2:
mate in one is taken and no request is spent; a free pawn is taken; `Qxf7+` that drops
the queen never reaches the shortlist; every candidate while in check is a real
escape; the opening shortlist reaches `e4`, `d4`, `Nf3` rather than a-file filler;
and a wide margin still excludes outright blunders.

That fifth one was a genuine bug. With material-only evaluation every opening move
ties at zero, so the shortlist was whatever move generation emitted first: a3, a4, b3,
b4, c3, c4. Jev would never have seen a real opening move. Piece-square tables at
weight 0.6 exist only to break that tie, and are held low on purpose, because generic
positional taste belongs to the search and the interesting taste is supposed to come
from the persona.

## Not verified

**The Jev layer has never run.** This sandbox cannot reach `api.typesafe.ai`
(`host_not_allowed` from the egress proxy), so `src/jev.ts` is written against the
documented HTTP contract and typechecked against `@typesafe-ai/sdk@0.6.0`, and that is
all. Everything below it is tested. The first real run is yours.

Specific things to watch on that first run:

- **Question count.** One Choice, one Noul and two Scores per candidate is `2 + 2N`
  questions, 14 at a six-move shortlist. They batch into one request and the parallel
  questions cookbook reports large savings for doing that, but measure the actual cost
  and latency per move rather than trusting the shape.
- **The annotation is the whole state.** Jev sees no board. It sees the English
  `describe()` produced and a short brief. If the bot plays strangely, print the
  candidate effects first; the fault is usually a missing clause, not the model.
- **The confidence threshold is a guess.** `minConfidence` per persona is set by feel.
  Confidence summarises how concentrated the distribution is, not whether acting is
  correct, and several moves being genuinely fine will spread it legitimately. Tune it
  on real games and watch how often `low-confidence-fallback` fires. If it is most
  moves, the shortlist is too homogeneous and the margin should widen.
- **Structural invariants do not hold.** The Choice `fit` and the per-candidate Scores
  answer different questions and are not arithmetically related. The composite in
  `engine.ts` mixes them anyway, which is a policy decision, not a derivation.

## Where the interesting version is

The persona is currently static prose. The version worth building reads the game so
far and picks its own plan, then judges moves against that: one request per game phase
that sets an intention from `recent_moves` and the position brief, and the per-move
requests judging against it. That is a plan the model holds and code keeps, which is
the "respond to changing state" pattern rather than a classifier.

## Files

| | |
| --- | --- |
| `src/tactics.ts` | search, shortlist, and English rendering. No model. |
| `src/jev.ts` | the one request. No arithmetic. |
| `src/personas.ts` | plan prose for the model, weights for the code. |
| `src/engine.ts` | composite scoring, confidence gate, fallbacks. |
| `src/probe.ts` | offline tests. |
| `src/play.ts` | CLI. |
