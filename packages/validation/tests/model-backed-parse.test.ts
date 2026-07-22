import { expect, test } from "bun:test";

import { parse } from "../src/index.ts";

const MANIFESTO = "required email with max 100 characters";
const MANIFESTO_INTENT = { type: "email", required: true, maxLength: 100 } as const;

test("the promoted default model preserves the full reference intent", async () => {
  const result = await parse(MANIFESTO);

  expect(result.source).toBe("model");
  expect(result.ok).toBe(true);
  if (result.ok) expect(result.intent).toEqual(MANIFESTO_INTENT);
});
