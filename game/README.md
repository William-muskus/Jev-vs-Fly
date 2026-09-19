# Jev vs Fly — Wizard Chess

A match: TypeSafe's Jev (server-side Choice over legal moves) against the
FlyWire connectome chess net running in the browser.

The fly is the original demo. Jev is the opponent. Nothing about the fly's
move choice changed — the worker in `web/engine/` still picks every fly move
from the connectome.

The board is **wizard chess**: ordinary FIDE movement, but the pieces are
enchanted. A capture always destroys the piece that is taken — there is no
combat roll — and the hall calls each move in English (*White knight to F3!*).

The 3D hall is vendored from [King's Gambit](https://github.com/alexngdev99/rork-medieval-3d-chess)
(MIT). The gothic-Staunton army is original. If you have nbauchat's Cults STLs
locally (private-use, not redistributed here), convert them with
`python -m game.wizard_assets` (searches known drop paths) or
`python -m game.wizard_assets --src /path/to/nbauchat/harry-potter-chess`.
They replace the procedural set via `public/models/wizard/manifest.json`.

```
  browser                         this server
  ──────                          ───────────
  fly worker  <── /model/ ──────  cached Hugging Face blob
  3D hall     ── /api/jev-move →  TypeSafe System One (Jev)
```

## Run

```bash
# required for Jev; never committed — a gitignored `.env` at the repo root also works
export TYPESAFE_API_KEY=…
python -m game.server         # API + classic 2D  http://127.0.0.1:8766/
cd game/medieval && npm install && npm run dev
# wizard chess 3D → http://127.0.0.1:8080/?autoplay=1
# or: fly jev-vs-fly --port 8766
```

Open the 3D page and click **Play wizard chess**, or add `?autoplay=1`.

Query flags: `strategy=best_this_turn|best_win_rate|both_turn_then_win|both_win_then_turn`,
`difficulty=larva|fly|superfly`, `jevColor=white|black`, `maxPlies=80`,
`speed=1`, `cinema=1`.

The first load fetches ~30 MB of fly brain into `game/.model-cache/` (gitignored)
from [cesp99/fly-chess](https://huggingface.co/cesp99/fly-chess) and serves it
same-origin so the worker does not trip Hugging Face CORS.

Classic 2D UI remains at `http://127.0.0.1:8766/`.
