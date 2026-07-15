import type { IntentFailureCode } from "../types/errors.js";
import { IntentError } from "../types/errors.js";
import type { ScheduleIntent, Weekday } from "../types/intent.js";
import { intentFailure } from "../validate.js";
import { unsupported, validateWeekdays } from "./validate.js";

export type CompileResult =
  | { supported: true; expression: string }
  | { supported: false; reason: string; code: CompileFailureCode };

export type CompileFailureCode =
  | IntentFailureCode
  | "non-unit-interval"
  | "ordinal-weekday"
  | "day-spec"
  | "bounds"
  | "exceptions";

export const WEEKDAY_NUMBER: Readonly<Record<Weekday, number>> = Object.freeze({
  sunday: 0,
  monday: 1,
  tuesday: 2,
  wednesday: 3,
  thursday: 4,
  friday: 5,
  saturday: 6,
});

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
  if (!Array.isArray(days)) throw new IntentError("invalid-weekday", "weekdays must be an array");
  const failure = validateWeekdays(days);

  if (failure !== null) throw new IntentError(failure.code, failure.reason);
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

  const failure = intentFailure(intent);

  if (failure !== null) return failure;

  if (intent.day !== undefined) {
    return unsupported("day-spec", "cron cannot represent an explicit day rule");
  }

  if (intent.bounds !== undefined) {
    return unsupported("bounds", "cron cannot represent recurrence bounds");
  }

  if (intent.exceptions !== undefined) {
    return unsupported("exceptions", "cron cannot represent exceptions");
  }

  const { unit, interval } = intent.frequency;

  if (intent.ordinalWeekday !== undefined) {
    return unsupported("ordinal-weekday", "cron cannot represent an ordinal weekday");
  }

  const time = intent.time;
  const minute = time?.minute ?? 0;
  const hour = time?.hour ?? 0;

  switch (unit) {
    case "minute": {
      if (60 % interval !== 0) {
        return unsupported(
          "non-unit-interval",
          "cron minute steps would change the interval at hour boundaries",
        );
      }

      const minuteField = interval === 60 ? "0" : interval === 1 ? "*" : `*/${interval}`;

      return supported(`${minuteField} * * * *`);
    }

    case "hour": {
      if (24 % interval !== 0) {
        return unsupported(
          "non-unit-interval",
          "cron hour steps would change the interval at day boundaries",
        );
      }

      const hourField = interval === 24 ? "0" : interval === 1 ? "*" : `*/${interval}`;

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
