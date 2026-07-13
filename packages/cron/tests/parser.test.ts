import { expect, test } from "bun:test";

import { parse } from "../src/index.ts";
import type { ParseFailureCode, ScheduleIntent } from "../src/index.ts";
import { normalize } from "../src/parser/normalize.ts";

const WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"] as const;

const phraseCases: Array<{ phrase: string; intent: ScheduleIntent }> = [
  {
    phrase: "every day",
    intent: { frequency: { unit: "day", interval: 1 } },
  },
  {
    phrase: "every weekday",
    intent: {
      frequency: { unit: "week", interval: 1 },
      weekdays: [...WEEKDAYS],
    },
  },
  {
    phrase: "every Monday",
    intent: { frequency: { unit: "week", interval: 1 }, weekdays: ["monday"] },
  },
  {
    phrase: "every Monday and Friday",
    intent: {
      frequency: { unit: "week", interval: 1 },
      weekdays: ["monday", "friday"],
    },
  },
  {
    phrase: "every hour",
    intent: { frequency: { unit: "hour", interval: 1 } },
  },
  {
    phrase: "every 30 minutes",
    intent: { frequency: { unit: "minute", interval: 30 } },
  },
  {
    phrase: "at 9",
    intent: {
      frequency: { unit: "day", interval: 1 },
      time: { hour: 9, minute: 0 },
    },
  },
  {
    phrase: "at 9:30",
    intent: {
      frequency: { unit: "day", interval: 1 },
      time: { hour: 9, minute: 30 },
    },
  },
  {
    phrase: "at 9pm",
    intent: {
      frequency: { unit: "day", interval: 1 },
      time: { hour: 21, minute: 0 },
    },
  },
  {
    phrase: "at nine",
    intent: {
      frequency: { unit: "day", interval: 1 },
      time: { hour: 9, minute: 0 },
    },
  },
  {
    phrase: "at twelve am",
    intent: {
      frequency: { unit: "day", interval: 1 },
      time: { hour: 0, minute: 0 },
    },
  },
  {
    phrase: "every other Friday",
    intent: {
      frequency: { unit: "week", interval: 2 },
      weekdays: ["friday"],
    },
  },
  {
    phrase: "Monday through Friday",
    intent: {
      frequency: { unit: "week", interval: 1 },
      weekdays: [...WEEKDAYS],
    },
  },
  {
    phrase: "the 15th",
    intent: {
      frequency: { unit: "month", interval: 1 },
      dayOfMonth: 15,
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
  { phrase: "", code: "empty" },
  { phrase: "   ", code: "empty" },
  { phrase: "blah blah", code: "unrecognized" },
  { phrase: "at 25", code: "unrecognized-time" },
  { phrase: "funday", code: "unrecognized-weekday" },
  { phrase: "every blah", code: "unrecognized-interval" },
  { phrase: "every 2 weeks and every day", code: "conflicting-selectors" },
  { phrase: "every day and every Monday", code: "conflicting-selectors" },
];

test.each(failureCases)("parse fails: $phrase -> $code", async ({ phrase, code }) => {
  const result = await parse(phrase, { source: "rules" });

  expect(result.ok).toBe(false);

  if (!result.ok) {
    expect(result.code).toBe(code);
  }
});

test("parse returns a promise", () => {
  expect(parse("every day")).toBeInstanceOf(Promise);
});

test("normalize canonicalizes and is idempotent", () => {
  const once = normalize("Every Monday at 9 A.M.!!");

  expect(once).toBe("every monday at 9 am");
  expect(normalize(once)).toBe(once);
  expect(normalize("Monday\u2013Friday")).toBe("monday-friday");
  expect(normalize("every1 weekday at 9pm")).toBe("every 1 weekday at 9 pm");
  expect(normalize("mOndAy")).toBe("monday");
  expect(normalize("the 15th and v2")).toBe("the 15th and v2");
  expect(() => normalize(123 as unknown as string)).toThrow(TypeError);
});

test("normalize does not merge an article into an m-word", () => {
  expect(normalize("a monthly reminder")).toBe("a monthly reminder");
  expect(normalize("at 9 a.m.")).toBe("at 9 am");
  expect(normalize("at 9 a. m.")).toBe("at 9 am");
});

test("normalize canonicalizes a meridiem-qualified dotted clock", () => {
  expect(normalize("Daily at 1.20 p.m.")).toBe("daily at 1:20 pm");
  expect(normalize("Release 1.20 is current")).toBe("release 1.20 is current");
});
