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
