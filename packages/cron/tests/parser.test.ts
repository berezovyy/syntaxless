import { expect, test } from "bun:test";

import { parse } from "../src/index.ts";
import type { ParseFailureCode, ScheduleIntent } from "../src/index.ts";
import { normalize } from "../src/parser/normalize.ts";

const phraseCases: Array<{ phrase: string; intent: ScheduleIntent }> = [
  {
    phrase: "every 30 minutes",
    intent: { frequency: { unit: "minute", interval: 30 } },
  },
  {
    phrase: "at 9pm",
    intent: {
      frequency: { unit: "day", interval: 1 },
      time: { hour: 21, minute: 0 },
    },
  },
  {
    phrase: "first monday",
    intent: {
      frequency: { unit: "month", interval: 1 },
      ordinalWeekday: { ordinal: "first", weekday: "monday" },
    },
  },
];

test.each(phraseCases)("parse: $phrase", async ({ phrase, intent }) => {
  expect(await parse(phrase, { source: "rules" })).toEqual({
    ok: true,
    intent,
    confidence: 1,
    diagnostics: [],
    source: "rules",
  });
});

const failureCases: Array<{ phrase: string; code: ParseFailureCode }> = [
  { phrase: "every day and every Monday", code: "conflicting-selectors" },
];

test.each(failureCases)("parse fails: $phrase -> $code", async ({ phrase, code }) => {
  const result = await parse(phrase, { source: "rules" });

  expect(result.ok).toBe(false);

  if (!result.ok) {
    expect(result.code).toBe(code);
  }
});

test("normalize canonicalizes a meridiem-qualified dotted clock", () => {
  expect(normalize("Daily at 1.20 p.m.")).toBe("daily at 1:20 pm");
  expect(normalize("Release 1.20 is current")).toBe("release 1.20 is current");
});

const oneOffCases = [
  { phrase: "remind me tomorrow at 3", marker: "tomorrow" },
  { phrase: "call mom tonight at 8pm", marker: "tonight" },
  { phrase: "next friday at 9am", marker: "next friday" },
  { phrase: "ping me in 10 minutes", marker: "in 10 minutes" },
];

test.each(oneOffCases)("rules refuse a single occasion: $phrase", async ({ phrase, marker }) => {
  expect(await parse(phrase, { source: "rules" })).toEqual({
    ok: false,
    reason: `"${marker}" describes a single occasion, not a recurring schedule`,
    code: "unsafe-context",
    confidence: 0,
    diagnostics: [
      {
        code: "one-off",
        severity: "warning",
        message: `"${marker}" describes a single occasion, not a recurring schedule`,
      },
    ],
    source: "rules",
  });
});

test("rules still accept recurring wording next to the refused markers", async () => {
  expect(await parse("every friday at 9am", { source: "rules" })).toMatchObject({
    ok: true,
    intent: {
      frequency: { unit: "week", interval: 1 },
      weekdays: ["friday"],
      time: { hour: 9, minute: 0 },
    },
  });

  expect(await parse("every 10 minutes", { source: "rules" })).toMatchObject({
    ok: true,
    intent: { frequency: { unit: "minute", interval: 10 } },
  });
});
