import { expect, test } from "vitest";
import { render } from "vitest-browser-react";

import { GameOverModal } from "./GameOverModal";

test("victory parchment names clocks, Jev spend, and a stamped PGN", async () => {
  const pgn = [
    '[Event "Jev vs Fly wizard chess"]',
    '[White "Jev"]',
    '[Black "Fruit Fly"]',
    '[Result "0-1"]',
    "",
    "1. f3 e5 2. g4 Qh4# 0-1",
  ].join("\n");
  const screen = await render(
    <GameOverModal
      result={{ winner: "b", reason: "checkmate" }}
      pgn={pgn}
      playerColor="w"
      versusComputer={false}
      moveCount={2}
      sideNames={{ w: "Jev", b: "Fruit Fly" }}
      elapsed={{ whiteMs: 125_000, blackMs: 98_000, totalMs: 223_000 }}
      jevCost={{ input_tokens: 28_000, output_tokens: 400 }}
      onRematch={() => undefined}
      onMenu={() => undefined}
    />,
  );

  await expect.element(screen.getByText(/FRUIT FLY TRIUMPHS/i)).toBeInTheDocument();
  await expect.element(screen.getByText("2:05")).toBeInTheDocument();
  await expect.element(screen.getByText("1:38")).toBeInTheDocument();
  await expect.element(screen.getByText(/Jev · \$0\.0012 · 28,400 tokens/)).toBeInTheDocument();
  await expect.element(screen.getByText(/\[White "Jev"\]/)).toBeInTheDocument();
  await expect.element(screen.getByText(/Qh4# 0-1/)).toBeInTheDocument();
  expect(screen.getByText(/\[White "\?"\]/).query()).toBeNull();
});
