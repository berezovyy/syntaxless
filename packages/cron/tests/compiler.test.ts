import { expect, test } from "bun:test";

import { compile } from "../src/index.ts";
import type {
  CompileFailureCode,
  CompileResult,
  Ordinal,
  ScheduleIntent,
  TimeOfDay,
  Weekday,
} from "../src/index.ts";

function ok(expression: string): CompileResult {
  return { supported: true, expression };
}

function fail(code: CompileFailureCode): CompileResult {
  return { supported: false, reason: code, code };
}

const weekdays = (days: Weekday[], time: TimeOfDay, interval = 1): ScheduleIntent => ({
  frequency: { unit: "week", interval },
  weekdays: days,
  time,
});

const cases: Array<{ name: string; intent: ScheduleIntent; expected: CompileResult }> = [
  {
    name: "every-minute",
    intent: { frequency: { unit: "minute", interval: 1 } },
    expected: ok("* * * * *"),
  },
  {
    name: "every-30-minutes",
    intent: { frequency: { unit: "minute", interval: 30 } },
    expected: ok("*/30 * * * *"),
  },
  {
    name: "every-hour",
    intent: { frequency: { unit: "hour", interval: 1 } },
    expected: ok("0 * * * *"),
  },
  {
    name: "every-2-hours-at-15",
    intent: {
      frequency: { unit: "hour", interval: 2 },
      time: { minute: 15 } as TimeOfDay,
    },
    expected: ok("15 */2 * * *"),
  },
  {
    name: "every-day",
    intent: { frequency: { unit: "day", interval: 1 } },
    expected: ok("0 0 * * *"),
  },
  {
    name: "every-day-at-9",
    intent: {
      frequency: { unit: "day", interval: 1 },
      time: { hour: 9, minute: 0 },
    },
    expected: ok("0 9 * * *"),
  },
  {
    name: "every-3-days",
    intent: { frequency: { unit: "day", interval: 3 } },
    expected: fail("non-unit-interval"),
  },
  {
    name: "monday",
    intent: weekdays(["monday"], { hour: 9, minute: 0 }),
    expected: ok("0 9 * * 1"),
  },
  {
    name: "monday-friday",
    intent: weekdays(["monday", "tuesday", "wednesday", "thursday", "friday"], {
      hour: 9,
      minute: 0,
    }),
    expected: ok("0 9 * * 1-5"),
  },
  {
    name: "monday-and-friday",
    intent: weekdays(["monday", "friday"], { hour: 9, minute: 0 }),
    expected: ok("0 9 * * 1,5"),
  },
  {
    name: "friday-sunday",
    intent: weekdays(["friday", "saturday", "sunday"], { hour: 9, minute: 0 }),
    expected: ok("0 9 * * 5,6,0"),
  },
  {
    name: "every-week",
    intent: { frequency: { unit: "week", interval: 1 } },
    expected: fail("missing-anchor"),
  },
  {
    name: "every-other-week",
    intent: { frequency: { unit: "week", interval: 2 } },
    expected: fail("non-unit-interval"),
  },
  {
    name: "first-of-month",
    intent: {
      frequency: { unit: "month", interval: 1 },
      dayOfMonth: 1,
      time: { hour: 9, minute: 0 },
    },
    expected: ok("0 9 1 * *"),
  },
  {
    name: "month-day-15",
    intent: {
      frequency: { unit: "month", interval: 1 },
      dayOfMonth: 15,
      time: { hour: 9, minute: 0 },
    },
    expected: ok("0 9 15 * *"),
  },
  {
    name: "every-month",
    intent: { frequency: { unit: "month", interval: 1 } },
    expected: fail("missing-anchor"),
  },
  {
    name: "every-2-months",
    intent: { frequency: { unit: "month", interval: 2 }, dayOfMonth: 1 },
    expected: fail("non-unit-interval"),
  },
  {
    name: "first-monday",
    intent: {
      frequency: { unit: "month", interval: 1 },
      ordinalWeekday: { ordinal: "first", weekday: "monday" },
    },
    expected: fail("ordinal-weekday"),
  },
  {
    name: "last-friday",
    intent: {
      frequency: { unit: "month", interval: 1 },
      ordinalWeekday: { ordinal: "last", weekday: "friday" },
    },
    expected: fail("ordinal-weekday"),
  },
  {
    name: "last-business-day",
    intent: {
      frequency: { unit: "day", interval: 1 },
      day: { type: "last_business_day" },
    },
    expected: fail("day-spec"),
  },
  {
    name: "bounded",
    intent: {
      frequency: { unit: "day", interval: 1 },
      bounds: { start: "2026-01-01" },
    },
    expected: fail("bounds"),
  },
  {
    name: "except-dates",
    intent: {
      frequency: { unit: "day", interval: 1 },
      exceptions: { dates: ["2026-01-01"] },
    },
    expected: fail("exceptions"),
  },
  {
    name: "missing-frequency",
    intent: {} as ScheduleIntent,
    expected: fail("missing-anchor"),
  },
  {
    name: "weekdays-and-dom",
    intent: {
      frequency: { unit: "week", interval: 1 },
      weekdays: ["monday"],
      dayOfMonth: 15,
    },
    expected: fail("conflicting-selectors"),
  },
  {
    name: "every-minute-at-9",
    intent: {
      frequency: { unit: "minute", interval: 1 },
      time: { hour: 9, minute: 0 },
    },
    expected: fail("conflicting-time"),
  },
  {
    name: "every-hour-at-9",
    intent: {
      frequency: { unit: "hour", interval: 1 },
      time: { hour: 9, minute: 0 },
    },
    expected: fail("conflicting-time"),
  },
  {
    name: "hour-24",
    intent: {
      frequency: { unit: "day", interval: 1 },
      time: { hour: 24, minute: 0 },
    },
    expected: fail("invalid-time"),
  },
  {
    name: "minute-60",
    intent: {
      frequency: { unit: "day", interval: 1 },
      time: { hour: 9, minute: 60 },
    },
    expected: fail("invalid-time"),
  },
  {
    name: "day-0",
    intent: { frequency: { unit: "month", interval: 1 }, dayOfMonth: 0 },
    expected: fail("invalid-day-of-month"),
  },
  {
    name: "day-32",
    intent: { frequency: { unit: "month", interval: 1 }, dayOfMonth: 32 },
    expected: fail("invalid-day-of-month"),
  },
  {
    name: "interval-0",
    intent: { frequency: { unit: "minute", interval: 0 } },
    expected: fail("invalid-interval"),
  },
  {
    name: "weekday-funday",
    intent: {
      frequency: { unit: "week", interval: 1 },
      weekdays: ["funday" as unknown as Weekday],
    },
    expected: fail("invalid-weekday"),
  },
  {
    name: "ordinal-sixth",
    intent: {
      frequency: { unit: "month", interval: 1 },
      ordinalWeekday: {
        ordinal: "sixth" as unknown as Ordinal,
        weekday: "monday",
      },
    },
    expected: fail("invalid-ordinal"),
  },
];

test.each(cases)("$name", ({ intent, expected }) => {
  const result = compile(intent);

  expect(result.supported).toBe(expected.supported);

  if (result.supported && expected.supported) {
    expect(result.expression).toBe(expected.expression);
  } else if (!result.supported && !expected.supported) {
    expect(result.code).toBe(expected.code);
  }
});

test("every weekday at 9 compiles to 0 9 * * 1-5", () => {
  const result = compile(
    weekdays(["monday", "tuesday", "wednesday", "thursday", "friday"], {
      hour: 9,
      minute: 0,
    }),
  );

  expect(result.supported).toBe(true);

  if (result.supported) {
    expect(result.expression).toBe("0 9 * * 1-5");
  }
});

test("null and non-object input are programming errors", () => {
  expect(() => compile(null as unknown as ScheduleIntent)).toThrow(TypeError);
  expect(() => compile("daily" as unknown as ScheduleIntent)).toThrow(TypeError);
});
