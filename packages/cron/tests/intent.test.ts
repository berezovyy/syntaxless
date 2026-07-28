import { expect, test } from "bun:test";

import {
  compile,
  compileWeekdays,
  explain,
  IntentError,
  recurrences,
  validate,
  WEEKDAY_NUMBER,
} from "../src/index.ts";
import type { IntentFailureCode, ScheduleIntent } from "../src/index.ts";
import { labelId } from "../src/model/labels.ts";
import { tokenize } from "../src/model/tokenizer.ts";
import { decodeIntent } from "../src/parser/decode.ts";

const daily: ScheduleIntent = { frequency: { unit: "day", interval: 1 } };

test("caller intent validation rejects facts that compilation previously ignored", () => {
  expect(() => compileWeekdays([])).toThrow(IntentError);
  expect(Object.isFrozen(WEEKDAY_NUMBER)).toBe(true);

  const cases: [unknown, IntentFailureCode][] = [
    [{ ...daily, timezone: "UTC" }, "invalid-intent"],
    [{ frequency: { unit: "day", interval: 1, phase: 2 } }, "missing-anchor"],
    [{ frequency: { unit: { toString: () => "day" }, interval: 1 } }, "missing-anchor"],
    [{ frequency: { unit: "day", interval: Number.MAX_SAFE_INTEGER + 1 } }, "invalid-interval"],
    [{ ...daily, time: {} }, "invalid-time"],
    [{ ...daily, time: { minute: 15 } }, "invalid-time"],
    [{ ...daily, time: { hour: 9 } }, "invalid-time"],
    [{ ...daily, time: { hour: 9, minute: 0, second: 20 } }, "invalid-time"],
    [{ ...daily, day: { type: "holiday" } }, "invalid-day"],
    [{ ...daily, day: { type: "last_day_of_month" } }, "conflicting-selectors"],
    [
      {
        frequency: { unit: "month", interval: 1 },
        dayOfMonth: 3,
        day: { type: "last_day_of_month" },
      },
      "conflicting-selectors",
    ],
    [{ ...daily, bounds: {} }, "invalid-bounds"],
    [{ ...daily, bounds: { start: "2026-02-29" } }, "invalid-bounds"],
    [{ ...daily, bounds: { start: "2026-03-01", end: "2026-03-01" } }, "invalid-bounds"],
    [{ ...daily, bounds: { start: "2026-03-02", end: "2026-03-01" } }, "invalid-bounds"],
    [{ ...daily, bounds: { start: "2026-03-01", inclusive: true } }, "invalid-bounds"],
    [{ ...daily, exceptions: { dates: ["2026-02-30"] } }, "invalid-exceptions"],
    [{ ...daily, exceptions: { dates: ["2026-03-01", "2026-03-01"] } }, "invalid-exceptions"],
    [{ ...daily, exceptions: { dates: Array(1) } }, "invalid-exceptions"],
    [{ ...daily, exceptions: { weekdays: ["sunday", "sunday"] } }, "invalid-exceptions"],
    [{ ...daily, exceptions: { holidays: true } }, "invalid-exceptions"],
  ];

  for (const [value, code] of cases) {
    let failure: unknown;

    try {
      validate(value);
    } catch (error) {
      failure = error;
    }

    expect(failure).toBeInstanceOf(IntentError);
    expect((failure as IntentError).code).toBe(code);
    const result = compile(value as ScheduleIntent);

    expect(result.supported).toBe(false);
    if (!result.supported) expect(result.code).toBe(code);
  }
});

test("valid calendar intent facts survive explanation without cron approximation", () => {
  const intent: ScheduleIntent = {
    frequency: { unit: "month", interval: 2 },
    ordinalWeekday: { ordinal: "last", weekday: "friday" },
    time: { hour: 23, minute: 45 },
    bounds: { start: "0099-01-01", end: "2028-03-01" },
    exceptions: { dates: ["2028-02-25"], weekdays: ["sunday"] },
  };

  const before = JSON.stringify(intent);

  validate(intent);
  const explanation = explain(intent);

  for (const fact of [
    "2 months",
    "last friday",
    "23:45",
    "inclusive local start date is 0099-01-01",
    "exclusive local end date is 2028-03-01",
    "2028-02-25",
    "excluded weekdays are sunday",
    "Cron compilation is unsupported",
  ]) {
    expect(explanation).toContain(fact);
  }

  expect(JSON.stringify(intent)).toBe(before);

  expect(explain({ frequency: { unit: "hour", interval: 2 }, time: { minute: 15 } })).toContain(
    "15 */2 * * *",
  );

  expect(explain(daily)).toContain("00:00 by default");

  expect(explain({ frequency: { unit: "week", interval: 1 } })).toContain(
    "use the anchor's weekday",
  );

  expect(explain({ frequency: { unit: "month", interval: 1 } })).toContain(
    "use the anchor's day of the month",
  );

  expect(
    explain({
      frequency: { unit: "month", interval: 1 },
      day: { type: "last_business_day" },
    }),
  ).toContain("without holiday exclusions");

  expect(() => explain({ ...daily, time: {} } as ScheduleIntent)).toThrow(IntentError);
});

test(
  "literal intervals remain valid intents beyond a clock field, " +
    "and cron only compiles equal steps",
  () => {
    for (const [unit, interval, expression] of [
      ["minute", 59, null],
      ["minute", 60, "0 * * * *"],
      ["minute", 61, null],
      ["minute", 120, null],
      ["hour", 23, null],
      ["hour", 24, "0 0 * * *"],
      ["hour", 25, null],
      ["hour", Number.MAX_SAFE_INTEGER, null],
    ] as const) {
      const text = `every ${interval} ${unit}s`;

      const labeled = tokenize(text)
        .filter((token) => token.kind !== 3)
        .map((token) => ({
          token,
          role:
            token.text === "every"
              ? labelId.RECUR
              : /^\d+$/.test(token.text)
                ? labelId.INTERVAL_LIT
                : unit === "minute"
                  ? labelId.UNIT_MIN
                  : labelId.UNIT_HOUR,
          boundary: token.text === "every",
          score: 1,
          actionable: true,
        }));

      const decoded = decodeIntent(labeled, undefined, text, "request");
      const intent = { frequency: { unit, interval } };

      expect(decoded.intent).toEqual(intent);
      validate(intent);
      expect(explain(intent)).toContain(`interval is ${interval} ${unit}`);

      expect(compile(intent)).toMatchObject(
        expression === null
          ? { supported: false, code: "non-unit-interval" }
          : { supported: true, expression },
      );
    }

    for (const interval of [0, -1, 1.5, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
      const intent = { frequency: { unit: "minute" as const, interval } };

      expect(() => validate(intent)).toThrow(IntentError);

      expect(compile(intent)).toMatchObject({
        supported: false,
        code: "invalid-interval",
      });
    }

    for (const [unit, interval, expected] of [
      ["minute", 120, ["2026-01-01T02:00:00.000Z", "2026-01-01T04:00:00.000Z"]],
      ["hour", 25, ["2026-01-02T01:00:00.000Z", "2026-01-03T02:00:00.000Z"]],
    ] as const) {
      const result = recurrences(
        { frequency: { unit, interval } },
        { timezone: "UTC", reference: "2026-01-01T00:00:00Z", count: 2 },
      );

      expect(result.ok).toBe(true);

      if (result.ok) {
        expect(result.occurrences.map((occurrence) => occurrence.instant)).toEqual([...expected]);
      }
    }
  },
);
