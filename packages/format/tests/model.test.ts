import { expect, test } from "bun:test";

import { dispose, parse, preload } from "../src/index.js";
import { weights } from "../src/model/weights.gen.js";

const artifact = (changes: Record<string, unknown>) =>
  new TextEncoder().encode(JSON.stringify({ ...weights, ...changes }));

test("bundled model parses the reference currency instruction and refuses ambiguity", async () => {
  const parsed = await parse("euros, two decimals, negatives in parentheses");

  expect(parsed).toMatchObject({
    ok: true,
    source: "model",
    intent: {
      kind: "currency",
      currency: "EUR",
      currencyDisplay: "symbol",
      precision: { minimum: 2, maximum: 2 },
      negative: "parentheses",
    },
  });

  expect(await parse("use dollars with two decimals")).toMatchObject({
    ok: false,
    code: "unsupported",
  });

  expect(await parse("   ")).toMatchObject({ ok: false, code: "invalid-input" });

  expect(await parse("x".repeat(4097))).toMatchObject({
    ok: false,
    code: "invalid-input",
  });
});

test("confidence floor and artifact version are enforced for caller weights", async () => {
  const text = "euros, two decimals, negatives in parentheses";

  expect((await parse(text, { weights: artifact({ confidenceFloor: 0 }) })).ok).toBe(true);

  expect(await parse(text, { weights: artifact({ confidenceFloor: 1 }) })).toMatchObject({
    ok: false,
    code: "low-confidence",
  });

  expect(await parse(text, { weights: artifact({ version: 3 }) })).toMatchObject({
    ok: false,
    code: "incompatible-model",
  });

  expect(await parse(text, { weights: new TextEncoder().encode("not json") })).toMatchObject({
    ok: false,
    code: "incompatible-model",
  });
});

test("a failed preload can be retried and dispose keeps parsing available", async () => {
  const bad = artifact({ version: 3 });

  await expect(preload({ weights: bad })).rejects.toThrow("unsupported-weights-version");
  await expect(preload({ weights: bad })).rejects.toThrow("unsupported-weights-version");
  await Promise.all([preload(), preload()]);
  dispose();
  expect((await parse("percentage with one decimal")).ok).toBe(true);
});
