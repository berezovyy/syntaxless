import { expect, test } from "bun:test";

import type { Label } from "../../src/model/labels";
import { tokenize } from "../../src/model/tokenizer";
import type { ScheduleIntent } from "../../src/types/intent";
import { alignText, expectedValues } from "./llm/align";
import type { AlignedSpan } from "./llm/align";

function checkCoverage(text: string, spans: readonly AlignedSpan[]): void {
  const tokens = tokenize(text).filter((token) => token.kind !== 3);
  const boundaries = new Set<number>();

  for (const token of tokenize(text)) {
    boundaries.add(token.start);
    boundaries.add(token.end);
  }

  let cursor = 0;

  for (const span of spans) {
    expect(span.start).toBeGreaterThanOrEqual(cursor);
    expect(boundaries.has(span.start)).toBe(true);
    expect(boundaries.has(span.end)).toBe(true);
    cursor = span.end;
  }

  for (const token of tokens) {
    const covering = spans.filter((span) => token.start >= span.start && token.end <= span.end);

    expect(covering.length).toBe(1);
  }
}

test("labels an interval number and its unit, opening the clause on the value", () => {
  const text = "every 2 hours";
  const intent: ScheduleIntent = { frequency: { unit: "hour", interval: 2 } };
  const result = alignText(text, intent);

  expect(result.ok).toBe(true);
  if (!result.ok) return;

  expect(result.spans).toEqual([
    { start: 0, end: 5, label: "RECUR", clauseStart: false },
    { start: 6, end: 7, label: "INTERVAL_2", clauseStart: true, value: 2 },
    { start: 8, end: 13, label: "UNIT_HOUR", clauseStart: false, value: "hour" },
  ]);

  expect(result.slots).toEqual([
    { label: "INTERVAL_2", value: 2, start: 6, end: 7 },
    { label: "UNIT_HOUR", value: "hour", start: 8, end: 13 },
  ]);

  checkCoverage(text, result.spans);
});

test("multiple slots of the same role are each labeled", () => {
  const text = "every Monday and Wednesday";

  const intent: ScheduleIntent = {
    frequency: { unit: "week", interval: 1 },
    weekdays: ["monday", "wednesday"],
  };

  const result = alignText(text, intent);

  expect(result.ok).toBe(true);
  if (!result.ok) return;

  const weekdays = result.spans.filter((span) => span.label.startsWith("WEEKDAY_"));

  expect(weekdays.map((span) => span.value)).toEqual(["monday", "wednesday"]);
  expect(weekdays.map((span) => text.slice(span.start, span.end))).toEqual(["Monday", "Wednesday"]);
  expect(result.slots.filter((slot) => slot.label.startsWith("WEEKDAY_")).length).toBe(2);
  checkCoverage(text, result.spans);
});

test("carrier prose is O and never picks up a schedule label", () => {
  const text = "remind me to water the plants every Monday";

  const intent: ScheduleIntent = {
    frequency: { unit: "week", interval: 1 },
    weekdays: ["monday"],
  };

  const result = alignText(text, intent);

  expect(result.ok).toBe(true);
  if (!result.ok) return;

  const carrier = result.spans.filter((span) => span.label === "O");
  const carrierText = carrier.map((span) => text.slice(span.start, span.end)).join(" ");

  expect(carrierText).toContain("remind");
  expect(carrierText).toContain("water");
  expect(carrierText).toContain("plants");

  expect(
    result.spans.some(
      (span) => span.label.startsWith("WEEKDAY_") && text.slice(span.start, span.end) === "Monday",
    ),
  ).toBe(true);

  checkCoverage(text, result.spans);
});

test("a day group is one label that stands for the whole set", () => {
  const text = "every weekend";

  const intent: ScheduleIntent = {
    frequency: { unit: "week", interval: 1 },
    weekdays: ["saturday", "sunday"],
  };

  const result = alignText(text, intent);

  expect(result.ok).toBe(true);
  if (!result.ok) return;

  expect(result.spans[1]).toEqual({
    start: 6,
    end: 13,
    label: "DAYGROUP_WEEKEND",
    clauseStart: true,
    value: "weekend",
  });
});

test("multiple weekday slots in a time sentence order and do not overlap", () => {
  const text = "every Monday and Wednesday at 9am";

  const intent: ScheduleIntent = {
    frequency: { unit: "week", interval: 1 },
    weekdays: ["monday", "wednesday"],
    time: { hour: 9, minute: 0 },
  };

  const result = alignText(text, intent);

  expect(result.ok).toBe(true);
  if (!result.ok) return;
  checkCoverage(text, result.spans);
  expect(result.spans.filter((span) => span.label.startsWith("WEEKDAY_"))).toHaveLength(2);
  expect(result.spans.at(-1)?.label).toBe("MERIDIEM_AM");
});

test("spans cover whole tokens, are ordered, and do not overlap", () => {
  const examples: Array<{ text: string; intent: ScheduleIntent }> = [
    {
      text: "every 5 minutes",
      intent: { frequency: { unit: "minute", interval: 5 } },
    },
    {
      text: "Monday",
      intent: { frequency: { unit: "week", interval: 1 }, weekdays: ["monday"] },
    },
    {
      text: "every day from 2026-01-01 until 2027-03-04",
      intent: {
        frequency: { unit: "day", interval: 1 },
        bounds: { start: "2026-01-01", end: "2027-03-04" },
      },
    },
    {
      text: "the 15th of every month",
      intent: { frequency: { unit: "month", interval: 1 }, dayOfMonth: 15 },
    },
  ];

  for (const { text, intent } of examples) {
    const result = alignText(text, intent);

    expect(result.ok).toBe(true);
    if (!result.ok) continue;
    checkCoverage(text, result.spans);
    const boundaries = new Set(tokenize(text).flatMap((token) => [token.start, token.end]));

    for (const span of result.spans) {
      expect(boundaries.has(span.start)).toBe(true);
      expect(boundaries.has(span.end)).toBe(true);
    }
  }
});

test("a schedule word missing from the lexicon is rejected, never guessed", () => {
  const intent: ScheduleIntent = {
    frequency: { unit: "week", interval: 1 },
    weekdays: ["monday"],
  };

  const result = alignText("every Monday fortnightly", intent);

  expect(result.ok).toBe(false);
  if (result.ok) return;
  expect(result.reason).toBe("unknown-lexeme");
});

test("only content spans yield values; structural labels do not", () => {
  const intent: ScheduleIntent = {
    frequency: { unit: "week", interval: 1 },
    weekdays: ["monday"],
    time: { hour: 9, minute: 0 },
  };

  const result = alignText("every Monday at 9", intent);

  expect(result.ok).toBe(true);
  if (!result.ok) return;
  const labels: Label[] = result.spans.map((span) => span.label);

  expect(labels).toContain("RECUR");
  expect(labels).toContain("GLUE");
});

test("expectedValues derives required slots from the intent alone", () => {
  const intent: ScheduleIntent = {
    frequency: { unit: "week", interval: 1 },
    weekdays: ["monday"],
    time: { hour: 21, minute: 30 },
  };

  const values = expectedValues(intent);

  expect(values).toContainEqual({
    label: "WEEKDAY_MON",
    value: "monday",
    required: true,
  });

  expect(values).toContainEqual({
    label: "INTERVAL_LIT",
    value: 1,
    required: false,
  });

  expect(values).toContainEqual({ label: "HOUR_21", value: 21, required: true });
  expect(values).toContainEqual({ label: "MINUTE_30", value: 30, required: true });

  expect(values).toContainEqual({
    label: "MERIDIEM_PM",
    value: "pm",
    required: false,
  });
});
