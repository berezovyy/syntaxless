import { expect, test } from "bun:test";

import { parse } from "../../src/index.ts";
import { renderField } from "../src/renderer.ts";

test("exclusive length is widened on parse", async () => {
  const rendered = renderField({ type: "string", required: true, minLength: 11 }, undefined, {
    order: "presence-type-comparison",
    spoken: { min: { value: 10, inclusive: false }, unit: true },
    minSurface: "more than",
    unitSurface: "characters",
    numberStyle: "digits",
  });

  expect(rendered.text).toContain("more than");
  const result = await parse(rendered.text, { source: "rules" });

  expect(result.ok).toBe(true);

  if (result.ok) {
    expect(result.intent).toEqual({ type: "string", required: true, minLength: 11 });
  }
});
