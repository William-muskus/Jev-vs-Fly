# Credits

The **code** in this repository is MIT. That covers our sources, not every mesh
someone might drop in beside them.

## In this repo

| What | Who | License | Source |
| --- | --- | --- | --- |
| Connectome chess engine, 2D site, fly worker | Carlo Esposito | MIT | [cesp99/fly-chess](https://github.com/cesp99/fly-chess) |
| 3D hall (Vite + React + three.js “King's Gambit”) | King's Gambit contributors | MIT | [alexngdev99/rork-medieval-3d-chess](https://github.com/alexngdev99/rork-medieval-3d-chess) — copy of that license: [`game/medieval/LICENSE`](game/medieval/LICENSE) |
| Gothic-Staunton fallback army (procedural) | This fork | MIT | `game/medieval/src/scene/` |
| Jev vs Fly hall wiring, TypeSafe proxy, wizard-chess rules overlay | This fork | MIT | `game/` |
| chess.js | Jeff Hlywa | BSD-2-Clause | `web/vendor/chess.js` |
| FlyWire connectome (the brain blob) | FlyWire consortium | **CC BY-NC 4.0** | [flywire.ai](https://flywire.ai) — not MIT; see [`docs/LICENSE-CC-BY-NC-4.0.txt`](docs/LICENSE-CC-BY-NC-4.0.txt) |
| Lichess games used in the original training | Lichess | CC0 | [database.lichess.org](https://database.lichess.org) |

Jev's move is a [TypeSafe](https://typesafe.com) System One Choice. That is an
API your key talks to; their model is not vendored here.

## Not in this repo

**nbauchat's Harry Potter chess STLs** ([Cults listing](https://cults3d.com/en/3d-model/art/harry-potter-chess))
are **CULTS PU / no-AI**, “for private use only.” They are **not MIT**. Cults
forbids distributing the digital files (and adaptations of them). Warner Bros.
also owns the character likenesses.

Keep those `.stl` files on your machine. Convert them locally:

```
python -m game.wizard_assets --src "…\nbauchat\harry-potter-chess"
```

Git ignores the resulting `.glb` files on purpose. A public clone of this repo
shows the original gothic-Staunton stone army unless you have done that convert
yourself.

If nbauchat (and the IP owner) grant a license that allows sharing the files,
they can go in git. Credits here are not a substitute for that license.
