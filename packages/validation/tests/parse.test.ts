import { expect, test } from "bun:test";

import { parse } from "../src/index.ts";

test("parse of empty input fails closed as empty", async () => {
  const result = await parse("");

  expect(result).toMatchObject({ ok: false, code: "empty" });
});
