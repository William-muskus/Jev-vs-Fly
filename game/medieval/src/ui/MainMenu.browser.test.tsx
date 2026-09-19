import { expect, test } from "vitest";
import { render } from "vitest-browser-react";

import { DEFAULT_ARMY_SKINS } from "../assets/generated";
import { DEFAULT_ARENA } from "../scene/arena";
import { MainMenu, type MatchConfig } from "./MainMenu";

const muster = { skins: DEFAULT_ARMY_SKINS, arena: DEFAULT_ARENA };

test("Computer tab starts a match against Jev or the Fly", async () => {
  const started: MatchConfig[] = [];
  const screen = await render(
    <MainMenu
      onStart={(config) => started.push(config)}
      onOpenSettings={() => undefined}
      muster={muster}
      onMuster={() => undefined}
      attract={false}
      onInteract={() => undefined}
    />,
  );

  await expect.element(screen.getByRole("button", { name: /^jev$/i })).toBeInTheDocument();
  await expect.element(screen.getByRole("button", { name: /fruit fly/i })).toBeInTheDocument();

  await screen.getByRole("button", { name: /fruit fly/i }).click();
  await expect.element(screen.getByRole("button", { name: /^larva$/i })).toBeInTheDocument();
  await expect.element(screen.getByRole("button", { name: /^superfly$/i })).toBeInTheDocument();

  await screen.getByRole("button", { name: /take the field/i }).click();
  expect(started.at(-1)).toMatchObject({ mode: "ai", opponent: "fly", flyDifficulty: "fly", playerColor: "w" });

  await screen.getByRole("button", { name: /^jev$/i }).click();
  await screen.getByRole("button", { name: /take the field/i }).click();
  expect(started.at(-1)).toMatchObject({ mode: "ai", opponent: "jev", playerColor: "w" });
});
