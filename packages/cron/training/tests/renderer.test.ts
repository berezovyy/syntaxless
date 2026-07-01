import { expect, test } from "bun:test";

import { isStructural, LABELS } from "../../src/model/labels";
import { tokenize } from "../../src/model/tokenizer";
import type { ScheduleIntent, Weekday } from "../../src/types/intent";
import { renderIntent, renderNegative } from "../src/renderer";
import type { RenderedExample } from "../src/renderer";
import { createRng } from "../src/rng";
import { FAMILY_WEIGHTS, sampleIntent } from "../src/sampler";
import type { Family } from "../src/sampler";

const WEEKDAY_WORDS = new Set([
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
  "mondays",
  "tuesdays",
  "wednesdays",
  "thursdays",
  "fridays",
  "saturdays",
  "sundays",
  "mon",
  "tue",
  "wed",
  "thu",
  "fri",
  "sat",
  "sun",
]);

const NUMBER_WORDS = new Set([
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "eleven",
  "twelve",
  "thirteen",
  "fourteen",
  "fifteen",
  "sixteen",
  "seventeen",
  "eighteen",
  "nineteen",
  "twenty",
  "thirty",
  "forty",
  "fifty",
]);

const CONTENT_LABELS = new Set(LABELS.filter((label) => label !== "O" && !isStructural(label)));

function checkExample(example: RenderedExample, family: Family): void {
  const { text, spans } = example;
  const tokens = tokenize(text);
  const boundaries = new Set<number>();

  for (const token of tokens) {
    boundaries.add(token.start);
    boundaries.add(token.end);
  }

  let cursor = 0;

  for (const span of spans) {
    // Strictly ordered, non-overlapping spans.
    expect(span.start).toBeGreaterThanOrEqual(cursor);
    // Everything between fragments is whitespace, and each fragment covers
    // whole tokenizer tokens (the Phase 13 alignment contract).
    expect(/^\s*$/.test(text.slice(cursor, span.start))).toBe(true);
    expect(boundaries.has(span.start)).toBe(true);
    expect(boundaries.has(span.end)).toBe(true);
    cursor = span.end;

    const spanText = text.slice(span.start, span.end);

    if (span.label === "WEEKDAY") {
      expect(WEEKDAY_WORDS.has(spanText.toLowerCase())).toBe(true);
    }

    if (span.label === "HOUR") {
      const isDigit = /^\d+$/.test(spanText);
      const isWord = NUMBER_WORDS.has(spanText.toLowerCase());

      expect(isDigit || isWord).toBe(true);
    }
  }

  expect(cursor).toBe(text.length);

  const starts = spans.filter((span) => span.clauseStart);

  expect(starts.length).toBeLessThanOrEqual(1);

  if (family === "negative") {
    expect(starts.length).toBe(0);
    expect(spans.every((span) => span.label === "O")).toBe(true);
  } else {
    expect(spans.some((span) => CONTENT_LABELS.has(span.label))).toBe(true);
    expect(starts.length).toBe(1);
    expect(CONTENT_LABELS.has(starts[0]!.label)).toBe(true);
  }
}

test("2000 seeded samples render with ordered, aligned, complete spans", () => {
  const entries = Object.entries(FAMILY_WEIGHTS) as Array<[Family, number]>;
  const rng = createRng(20260913);

  for (let index = 0; index < 2000; index++) {
    const family = rng.weighted(entries);
    const sample = sampleIntent(rng, family);

    const example =
      sample.intent === null ? renderNegative(rng) : renderIntent(sample.intent, family, rng);

    expect(example.text.length).toBeGreaterThan(0);
    expect(example.template.startsWith(family)).toBe(true);
    checkExample(example, family);
  }
});

test("the same intent, style, and seed render byte-identically", () => {
  const cases: Array<{
    family: Exclude<Family, "negative">;
    style: string;
    intent: ScheduleIntent;
  }> = [
    {
      family: "interval",
      style: "every",
      intent: { frequency: { unit: "week", interval: 3 } },
    },
    {
      family: "weekday-list",
      style: "on-days",
      intent: {
        frequency: { unit: "week", interval: 1 },
        weekdays: ["monday", "wednesday"],
      },
    },
    {
      family: "weekday-range",
      style: "dash",
      intent: {
        frequency: { unit: "week", interval: 1 },
        weekdays: ["monday", "tuesday", "wednesday"],
      },
    },
    {
      family: "ordinal-weekday",
      style: "the-ordinal-of",
      intent: {
        frequency: { unit: "month", interval: 1 },
        ordinalWeekday: { ordinal: "first", weekday: "monday" },
      },
    },
    {
      family: "day-of-month",
      style: "on-the",
      intent: { frequency: { unit: "month", interval: 1 }, dayOfMonth: 15 },
    },
    {
      family: "time-of-day",
      style: "digits",
      intent: { frequency: { unit: "day", interval: 1 }, time: { hour: 21, minute: 30 } },
    },
    {
      family: "bounds",
      style: "from-to",
      intent: {
        frequency: { unit: "day", interval: 1 },
        bounds: { start: "2026-01-01", end: "2027-03-04" },
      },
    },
    {
      family: "exception",
      style: "except",
      intent: {
        frequency: { unit: "week", interval: 1 },
        weekdays: ["monday" as Weekday],
        exceptions: { weekdays: ["friday"] },
      },
    },
    {
      family: "terse",
      style: "bare",
      intent: {
        frequency: { unit: "week", interval: 1 },
        weekdays: ["sunday"],
      },
    },
  ];

  for (const { family, style, intent } of cases) {
    const first = renderIntent(intent, family, createRng(4242), style);
    const second = renderIntent(intent, family, createRng(4242), style);

    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  }

  const negative = renderNegative(createRng(4242));

  expect(JSON.stringify(negative)).toBe(JSON.stringify(renderNegative(createRng(4242))));
});

test("labels are supplied by add() and never derived from parsing", () => {
  const example = renderIntent(
    { frequency: { unit: "week", interval: 1 }, weekdays: ["monday"] },
    "terse",
    createRng(7),
    "bare",
  );

  const content = example.spans.filter((span) => CONTENT_LABELS.has(span.label));

  expect(content.map((span) => example.text.slice(span.start, span.end))).toEqual(["Monday"]);
});
