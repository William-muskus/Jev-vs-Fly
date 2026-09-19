import { describe, expect, it } from "vitest";

import { MainMenu } from "./MainMenu";

describe("MainMenu", () => {
  it("exports a component Vite can transform", () => {
    expect(typeof MainMenu).toBe("function");
  });
});
