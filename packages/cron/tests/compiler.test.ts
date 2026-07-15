import { expect, test } from "bun:test";

import { compile } from "../src/index.ts";
import type {
  CompileFailureCode,
  CompileResult,
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
    name: "every-30-minutes",
    intent: { frequency: { unit: "minute", interval: 30 } },
    expected: ok("*/30 * * * *"),
  },
  {
    name: "every-2-hours-at-15",
    intent: {
      frequency: { unit: "hour", interval: 2 },
      time: { minute: 15 },
    },
    expected: ok("15 */2 * * *"),
  },
  {
    name: "every-day",
    intent: { frequency: { unit: "day", interval: 1 } },
    expected: ok("0 0 * * *"),
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
    name: "first-of-month",
    intent: {
      frequency: { unit: "month", interval: 1 },
      dayOfMonth: 1,
      time: { hour: 9, minute: 0 },
    },
    expected: ok("0 9 1 * *"),
  },
  {
    name: "first-monday",
    intent: {
      frequency: { unit: "month", interval: 1 },
      ordinalWeekday: { ordinal: "first", weekday: "monday" },
    },
    expected: fail("ordinal-weekday"),
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
