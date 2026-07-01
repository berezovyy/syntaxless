import type { CompileFailureCode, CompileResult } from "./cron.js";

export type Failure = Extract<CompileResult, { supported: false }>;

const WEEKDAY_NAMES = new Set<string>([
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
]);

const ORDINAL_NAMES = new Set<string>(["first", "second", "third", "fourth", "last"]);

export function unsupported(code: CompileFailureCode, reason: string): Failure {
  return { supported: false, reason, code };
}

function isInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value);
}

export function validateInterval(unit: unknown, interval: unknown): Failure | null {
  if (!isInteger(interval) || interval < 1) {
    return unsupported("invalid-interval", "frequency.interval must be an integer >= 1");
  }

  if (unit === "minute" && interval > 59) {
    return unsupported("invalid-interval", "minute interval must be at most 59");
  }

  if (unit === "hour" && interval > 23) {
    return unsupported("invalid-interval", "hour interval must be at most 23");
  }

  return null;
}

export function validateTime(time: unknown): Failure | null {
  if (time === undefined) return null;

  if (typeof time !== "object" || time === null) {
    return unsupported("invalid-time", "time must be an object");
  }

  const { hour, minute } = time as { hour?: unknown; minute?: unknown };

  if (hour !== undefined && (!isInteger(hour) || hour < 0 || hour > 23)) {
    return unsupported("invalid-time", "time.hour must be an integer from 0 to 23");
  }

  if (minute !== undefined && (!isInteger(minute) || minute < 0 || minute > 59)) {
    return unsupported("invalid-time", "time.minute must be an integer from 0 to 59");
  }

  return null;
}

export function validateWeekdays(weekdays: unknown): Failure | null {
  if (weekdays === undefined) return null;

  if (!Array.isArray(weekdays)) {
    return unsupported("invalid-weekday", "weekdays must be an array");
  }

  if (weekdays.length === 0) {
    return unsupported("invalid-weekday", "weekdays must not be empty");
  }

  if (weekdays.length > 7) {
    return unsupported("invalid-weekday", "weekdays must not exceed 7 entries");
  }

  const seen = new Set<string>();

  for (const day of weekdays) {
    if (typeof day !== "string" || !WEEKDAY_NAMES.has(day)) {
      return unsupported("invalid-weekday", `unknown weekday: ${String(day)}`);
    }

    if (seen.has(day)) {
      return unsupported("invalid-weekday", `duplicate weekday: ${day}`);
    }

    seen.add(day);
  }

  return null;
}

export function validateDayOfMonth(dayOfMonth: unknown): Failure | null {
  if (dayOfMonth === undefined) return null;

  if (!isInteger(dayOfMonth) || dayOfMonth < 1 || dayOfMonth > 31) {
    return unsupported("invalid-day-of-month", "dayOfMonth must be an integer from 1 to 31");
  }

  return null;
}

export function validateOrdinalWeekday(ordinalWeekday: unknown): Failure | null {
  if (ordinalWeekday === undefined) return null;

  if (typeof ordinalWeekday !== "object" || ordinalWeekday === null) {
    return unsupported("invalid-ordinal", "ordinalWeekday must be an object");
  }

  const { ordinal, weekday } = ordinalWeekday as {
    ordinal?: unknown;
    weekday?: unknown;
  };

  if (typeof ordinal !== "string" || !ORDINAL_NAMES.has(ordinal)) {
    return unsupported("invalid-ordinal", `unknown ordinal: ${String(ordinal)}`);
  }

  if (typeof weekday !== "string" || !WEEKDAY_NAMES.has(weekday)) {
    return unsupported("invalid-weekday", `unknown weekday: ${String(weekday)}`);
  }

  return null;
}
