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
  expect(await parse(phrase)).toEqual({ ok: true, intent });
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
  const result = await parse(phrase);

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
  expect(() => normalize(123 as unknown as string)).toThrow(TypeError);
});
