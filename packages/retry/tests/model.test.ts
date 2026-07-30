import { expect, test } from "bun:test";

import { compile, dispose, explain, parse, preload } from "../src/index.js";
import type { RetryIntent } from "../src/index.js";

const reference = "retry three times, double the delay, cap at one minute";

test("the bundled model parses the reference request without inventing a delay", async () => {
  expect(await parse(reference)).toMatchObject({
    ok: false,
    code: "missing-initial-delay",
  });

  const result = await parse(reference, { defaults: { initialDelayMs: 250 } });

  expect(result).toMatchObject({
    ok: true,
    source: "model",
    intent: {
      retries: 3,
      strategy: "exponential",
      initialDelayMs: 250,
      multiplier: 2,
      maxDelayMs: 60_000,
      defaults: ["initialDelayMs"],
    },
  });

  if (!result.ok) throw new Error("reference did not parse");

  expect(compile(result.intent)).toEqual({
    supported: true,
    delaysMs: [250, 500, 1000],
  });
});

test("questions and total-attempt wording refuse, with a positive control", async () => {
  expect((await parse("retry three times with constant delay of one second")).ok).toBe(true);

  for (const text of [
    "retry three times with constant delay of one second?",
    "retry three attempts including the first with constant delay of one second",
    "",
  ]) {
    expect((await parse(text)).ok).toBe(false);
  }
});

test("incompatible or malformed custom weights refuse instead of throwing", async () => {
  const encode = (value: unknown) => new TextEncoder().encode(JSON.stringify(value));

  for (const weights of [
    encode({ version: 99 }),
    encode([]),
    new TextEncoder().encode("{not json"),
  ]) {
    expect(await parse(reference, { weights })).toMatchObject({
      ok: false,
      code: "incompatible-model",
    });
  }
});

test("dispose clears the cache and a later preload restores parsing", async () => {
  await preload();
  dispose();
  await preload();
  expect((await parse(reference, { defaults: { initialDelayMs: 250 } })).ok).toBe(true);
});

test("explanations start sentences with capitals and use singular retry counts", () => {
  const intent: RetryIntent = {
    retries: 1,
    strategy: "exponential",
    initialDelayMs: 100,
    multiplier: 2,
    jitter: { kind: "full" },
    defaults: ["jitter"],
  };

  expect(explain(intent)).toBe(
    "1 retry after the initial attempt. Exponential delays, starting at 100 ms. " +
      "Multiply the initial delay by 2 to the retry index, " +
      "rounding up each exact result to whole milliseconds. " +
      "Full jitter (caller default), using an explicit caller seed. " +
      "The package returns a delay schedule and does not execute requests.",
  );

  expect(explain({ retries: 2, strategy: "constant", initialDelayMs: 5, defaults: [] })).toBe(
    "2 retries after the initial attempt. Constant delays, starting at 5 ms. " +
      "No jitter is applied. The package returns a delay schedule and does not execute requests.",
  );
});
