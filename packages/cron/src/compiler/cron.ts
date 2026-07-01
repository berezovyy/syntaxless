import type { FrequencyUnit, ScheduleIntent, Weekday } from "../types/intent.js";
import {
  unsupported,
  validateDayOfMonth,
  validateInterval,
  validateOrdinalWeekday,
  validateTime,
  validateWeekdays,
} from "./validate.js";

export type CompileResult =
  | { supported: true; expression: string }
  | { supported: false; reason: string; code: CompileFailureCode };

export type CompileFailureCode =
  | "invalid-interval"
  | "invalid-time"
  | "invalid-weekday"
  | "invalid-day-of-month"
  | "invalid-ordinal"
  | "conflicting-selectors"
  | "conflicting-time"
  | "missing-anchor"
  | "non-unit-interval"
  | "ordinal-weekday"
  | "day-spec"
  | "bounds"
  | "exceptions";

export const WEEKDAY_NUMBER: Record<Weekday, number> = {
  sunday: 0,
  monday: 1,
  tuesday: 2,
  wednesday: 3,
  thursday: 4,
  friday: 5,
  saturday: 6,
};

function weekOrder(value: number): number {
  return (value + 6) % 7;
}

function renderRun(run: number[]): string {
  if (run.length >= 3) {
    return `${run[0]}-${run[run.length - 1]}`;
  }

  return run.join(",");
}

export function compileWeekdays(days: Weekday[]): string {
  const numbers = Array.from(new Set(days.map((day) => WEEKDAY_NUMBER[day])));

  numbers.sort((a, b) => weekOrder(a) - weekOrder(b));
  const runs: number[][] = [];

  for (const value of numbers) {
    const current = runs[runs.length - 1];
    const previous = current?.[current.length - 1];

    if (current !== undefined && previous !== undefined && value - previous === 1) {
      current.push(value);
    } else {
      runs.push([value]);
    }
  }

  return runs.map(renderRun).join(",");
}

function supported(expression: string): CompileResult {
  return { supported: true, expression };
}

export function compile(intent: ScheduleIntent): CompileResult {
  if (intent === null || typeof intent !== "object") {
    throw new TypeError("compile() expects a ScheduleIntent object");
  }

  if (intent.day !== undefined) {
    return unsupported("day-spec", "cron cannot represent an explicit day rule");
  }

  if (intent.bounds !== undefined) {
    return unsupported("bounds", "cron cannot represent recurrence bounds");
  }

  if (intent.exceptions !== undefined) {
    return unsupported("exceptions", "cron cannot represent exceptions");
  }

  const frequency = (intent as { frequency?: unknown }).frequency;

  if (typeof frequency !== "object" || frequency === null) {
    return unsupported("missing-anchor", "a frequency is required");
  }

  const { unit, interval } = frequency as { unit: FrequencyUnit; interval: number };

  const intervalFailure = validateInterval(unit, interval);

  if (intervalFailure) return intervalFailure;

  const timeFailure = validateTime(intent.time);

  if (timeFailure) return timeFailure;

  const weekdaysFailure = validateWeekdays(intent.weekdays);

  if (weekdaysFailure) return weekdaysFailure;

  const dayOfMonthFailure = validateDayOfMonth(intent.dayOfMonth);

  if (dayOfMonthFailure) return dayOfMonthFailure;

  const ordinalFailure = validateOrdinalWeekday(intent.ordinalWeekday);

  if (ordinalFailure) return ordinalFailure;

  if (intent.weekdays !== undefined && intent.dayOfMonth !== undefined) {
    return unsupported(
      "conflicting-selectors",
      "weekdays and dayOfMonth cannot be combined in cron",
    );
  }

  const time = intent.time;
  const hasHour = time !== undefined && (time as { hour?: unknown }).hour !== undefined;

  if (unit === "minute" && time !== undefined) {
    return unsupported("conflicting-time", "a minute frequency cannot carry a time");
  }

  if (unit === "hour" && hasHour) {
    return unsupported("conflicting-time", "an hour frequency cannot carry an hour anchor");
  }

  if (intent.ordinalWeekday !== undefined) {
    return unsupported("ordinal-weekday", "cron cannot represent an ordinal weekday");
  }

  const minute = time?.minute ?? 0;
  const hour = (time as { hour?: number } | undefined)?.hour ?? 0;

  switch (unit) {
    case "minute": {
      const minuteField = interval === 1 ? "*" : `*/${interval}`;

      return supported(`${minuteField} * * * *`);
    }

    case "hour": {
      const hourField = interval === 1 ? "*" : `*/${interval}`;

      return supported(`${minute} ${hourField} * * *`);
    }

    case "day": {
      if (interval !== 1) {
        return unsupported("non-unit-interval", "day intervals greater than 1 are not expressible");
      }

      return supported(`${minute} ${hour} * * *`);
    }

    case "week": {
      if (interval !== 1) {
        return unsupported(
          "non-unit-interval",
          "week intervals greater than 1 are not expressible",
        );
      }

      if (intent.weekdays === undefined) {
        return unsupported("missing-anchor", "a weekly schedule needs weekdays");
      }

      return supported(`${minute} ${hour} * * ${compileWeekdays(intent.weekdays)}`);
    }

    case "month": {
      if (interval !== 1) {
        return unsupported(
          "non-unit-interval",
          "month intervals greater than 1 are not expressible",
        );
      }

      if (intent.dayOfMonth === undefined) {
        return unsupported("missing-anchor", "a monthly schedule needs a day of month");
      }

      return supported(`${minute} ${hour} ${intent.dayOfMonth} * *`);
    }

    default: {
      return unsupported("missing-anchor", `unsupported frequency unit: ${String(unit)}`);
    }
  }
}
