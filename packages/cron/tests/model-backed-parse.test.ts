import { expect, test } from "bun:test";

import { cron, dispose, getEngine, parse, preload } from "../src/index.ts";
import type { ScheduleIntent } from "../src/index.ts";
import { tokenize } from "../src/model/tokenizer.ts";
import { normalize } from "../src/parser/normalize.ts";

const weekdayMorning: ScheduleIntent = {
  frequency: { unit: "week", interval: 1 },
  weekdays: ["monday", "tuesday", "wednesday", "thursday", "friday"],
  time: { hour: 9, minute: 0 },
};

test("cron uses the bundled model result without inventing or changing fields", async () => {
  expect(await cron("every weekday at 9am")).toEqual({
    supported: true,
    expression: "0 9 * * 1-5",
    intent: weekdayMorning,
  });
});

test("the bundled model refuses a one-off request with a readable reason", async () => {
  const result = await parse("remind me tomorrow at 3");

  expect(result.ok).toBe(false);
  if (result.ok) return;
  expect(result.code).toBe("unsafe-context");
  expect(result.source).toBe("model");
  expect(result.reason).not.toContain("decodeIntent");
  expect(result.reason).toMatch(/^the text /);

  expect(await cron("remind me tomorrow at 3")).toEqual({
    supported: false,
    reason: result.reason,
    code: "unsafe-context",
    intent: null,
  });
});

test("preload shares the parsing engine and dispose releases it", async () => {
  await preload();
  const first = await getEngine();

  expect(await getEngine()).toBe(first);
  dispose();
  const second = await getEngine();

  expect(second).not.toBe(first);
  // An engine held by a caller remains usable after dispose and agrees with the new one.
  const tokens = tokenize(normalize("every weekday at 9am"));

  expect(first.inferTokens(tokens, true).labels).toEqual(second.inferTokens(tokens, true).labels);
  expect((await parse("every weekday at 9am")).ok).toBe(true);
});
