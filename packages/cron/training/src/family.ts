/** Corpus families and the structural shape check shared by the generators. */
import type { Ordinal, ScheduleIntent, Weekday } from "../../src/index";

export type Family =
  | "interval"
  | "weekday-list"
  | "weekday-range"
  | "ordinal-weekday"
  | "day-of-month"
  | "time-of-day"
  | "bounds"
  | "exception"
  | "terse"
  | "negative";

const WEEKDAYS: readonly Weekday[] = [
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
];

const ORDINALS: readonly Ordinal[] = ["first", "second", "third", "fourth", "last"];

/**
 * Shape guard, not a second compiler. Checks only invariants independent of
 * cron expressibility: weekday lists non-empty/distinct/known, clock ranges,
 * and the ordinal/weekday enums. Representability bounds (interval, day of
 * month) are deliberately left to the compiler.
 */
export function assertValid(intent: ScheduleIntent): void {
  if (intent.weekdays !== undefined) {
    if (intent.weekdays.length === 0) {
      throw new Error("assertValid: weekdays must be non-empty");
    }

    const seen = new Set<Weekday>();

    for (const weekday of intent.weekdays) {
      if (!WEEKDAYS.includes(weekday)) {
        throw new Error(`assertValid: unknown weekday ${String(weekday)}`);
      }

      if (seen.has(weekday)) {
        throw new Error(`assertValid: duplicate weekday ${weekday}`);
      }

      seen.add(weekday);
    }
  }

  if (intent.time !== undefined) {
    const { hour, minute } = intent.time;

    if (
      hour === undefined
        ? intent.frequency.unit !== "hour"
        : !Number.isInteger(hour) || hour < 0 || hour > 23
    ) {
      throw new Error(`assertValid: hour out of range: ${String(hour)}`);
    }

    if (!Number.isInteger(minute) || minute < 0 || minute > 59) {
      throw new Error(`assertValid: minute out of range: ${String(minute)}`);
    }
  }

  if (intent.ordinalWeekday !== undefined) {
    const { ordinal, weekday } = intent.ordinalWeekday;

    if (!ORDINALS.includes(ordinal)) {
      throw new Error(`assertValid: unknown ordinal ${String(ordinal)}`);
    }

    if (!WEEKDAYS.includes(weekday)) {
      throw new Error(`assertValid: unknown ordinal weekday ${String(weekday)}`);
    }
  }
}
