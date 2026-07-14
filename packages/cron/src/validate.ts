import {
  unsupported,
  validateDayOfMonth,
  validateInterval,
  validateOrdinalWeekday,
  validateTime,
  validateWeekdays,
} from "./compiler/validate.js";
import { parseDateLiteral } from "./schedule-semantics.js";
import { IntentError } from "./types/errors.js";
import type { IntentFailure } from "./types/errors.js";
import type { ScheduleIntent } from "./types/intent.js";

function record(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object") return false;
  const prototype = Object.getPrototypeOf(value);

  return prototype === Object.prototype || prototype === null;
}

function shape(value: unknown, keys: readonly string[]): value is Record<string, unknown> {
  return (
    record(value) &&
    Reflect.ownKeys(value).every((key) => typeof key === "string" && keys.includes(key))
  );
}

function date(value: unknown): value is string {
  return typeof value === "string" && parseDateLiteral(value) === value;
}

/** Internal result form shared by compilation and the public throwing validator. */
export function intentFailure(value: unknown): IntentFailure | null {
  if (
    !shape(value, [
      "frequency",
      "time",
      "weekdays",
      "dayOfMonth",
      "ordinalWeekday",
      "day",
      "bounds",
      "exceptions",
    ])
  ) {
    return unsupported("invalid-intent", "provide an intent object with only supported fields");
  }

  const frequency = value.frequency;

  if (
    !shape(frequency, ["unit", "interval"]) ||
    typeof frequency.unit !== "string" ||
    !["minute", "hour", "day", "week", "month"].includes(frequency.unit)
  ) {
    return unsupported("missing-anchor", "provide a supported frequency unit and interval");
  }

  const intervalFailure = validateInterval(frequency.interval);

  if (intervalFailure !== null) return intervalFailure;

  const failure =
    validateTime(value.time) ??
    validateWeekdays(value.weekdays) ??
    validateDayOfMonth(value.dayOfMonth) ??
    validateOrdinalWeekday(value.ordinalWeekday);

  if (failure !== null) return failure;

  if (value.ordinalWeekday !== undefined && !shape(value.ordinalWeekday, ["ordinal", "weekday"])) {
    return unsupported("invalid-ordinal", "ordinalWeekday must contain only ordinal and weekday");
  }

  if (
    value.day !== undefined &&
    (!shape(value.day, ["type"]) ||
      typeof value.day.type !== "string" ||
      !["last_business_day", "last_day_of_month"].includes(value.day.type))
  ) {
    return unsupported("invalid-day", "provide a supported day rule");
  }

  const selectors = [value.weekdays, value.dayOfMonth, value.ordinalWeekday, value.day].filter(
    (field) => field !== undefined,
  ).length;

  if (selectors > 1) {
    return unsupported("conflicting-selectors", "provide at most one day selector");
  }

  if (value.weekdays !== undefined && frequency.unit !== "week") {
    return unsupported("conflicting-selectors", "weekdays require a weekly frequency");
  }

  if (
    (value.dayOfMonth !== undefined ||
      value.ordinalWeekday !== undefined ||
      value.day !== undefined) &&
    frequency.unit !== "month"
  ) {
    return unsupported(
      "conflicting-selectors",
      "monthly day selectors require a monthly frequency",
    );
  }

  if (value.time !== undefined) {
    if (frequency.unit === "minute") {
      return unsupported("conflicting-time", "a minute frequency cannot carry a time");
    }

    if (!shape(value.time, ["hour", "minute"]) || value.time.minute === undefined) {
      return unsupported(
        "invalid-time",
        "time must contain a minute and only supported clock fields",
      );
    }

    if (frequency.unit === "hour" && value.time.hour !== undefined) {
      return unsupported("conflicting-time", "an hour frequency cannot carry an hour anchor");
    }

    if (frequency.unit !== "hour" && value.time.hour === undefined) {
      return unsupported(
        "invalid-time",
        "calendar frequencies require both hour and minute when time is supplied",
      );
    }
  }

  if (value.bounds !== undefined) {
    const bounds = value.bounds;

    if (
      !shape(bounds, ["start", "end"]) ||
      (bounds.start === undefined && bounds.end === undefined) ||
      (bounds.start !== undefined && !date(bounds.start)) ||
      (bounds.end !== undefined && !date(bounds.end))
    ) {
      return unsupported(
        "invalid-bounds",
        "provide inclusive start or exclusive end as valid YYYY-MM-DD dates",
      );
    }

    if (
      typeof bounds.start === "string" &&
      typeof bounds.end === "string" &&
      bounds.start >= bounds.end
    ) {
      return unsupported("invalid-bounds", "bounds.start must precede bounds.end");
    }
  }

  if (value.exceptions !== undefined) {
    const exceptions = value.exceptions;

    if (
      !shape(exceptions, ["dates", "weekdays"]) ||
      (exceptions.dates === undefined && exceptions.weekdays === undefined)
    ) {
      return unsupported("invalid-exceptions", "provide exception dates or weekdays");
    }

    if (
      exceptions.dates !== undefined &&
      (!Array.isArray(exceptions.dates) ||
        !Array.from(exceptions.dates).every(date) ||
        new Set(exceptions.dates).size !== exceptions.dates.length)
    ) {
      return unsupported(
        "invalid-exceptions",
        "exception dates must be distinct valid YYYY-MM-DD dates",
      );
    }

    const weekdaysFailure = validateWeekdays(exceptions.weekdays);

    if (weekdaysFailure !== null) return unsupported("invalid-exceptions", weekdaysFailure.reason);
  }

  return null;
}

/** Validate all serializable intent facts, without requiring cron support. */
export function validate(intent: unknown): asserts intent is ScheduleIntent {
  const failure = intentFailure(intent);

  if (failure !== null) throw new IntentError(failure.code, failure.reason);
}
