/** Gregorian weekday names. No locale abbreviations belong in the AST. */
export type Weekday =
  "monday" | "tuesday" | "wednesday" | "thursday" | "friday" | "saturday" | "sunday";

/** The recurrence unit. `interval` multiplies this unit. */
export type FrequencyUnit = "minute" | "hour" | "day" | "week" | "month";

/** Ordinal positions used by monthly weekday rules. */
export type Ordinal = "first" | "second" | "third" | "fourth" | "last";

/** A wall-clock time resolved later by the caller's timezone. */
export interface TimeOfDay {
  hour: number; // 0-23
  minute: number; // 0-59
}

/** A minute within each selected hour, without a daily hour anchor. */
export interface MinuteOfHour {
  hour?: never;
  minute: number; // 0-59
}

/** e.g. "first Monday of every month". */
export interface OrdinalWeekday {
  ordinal: Ordinal;
  weekday: Weekday;
}

/** Day selectors cron cannot express; `compile()` reports `supported: false` for
 * these instead of forcing a wrong expression. It never invents a fallback. */
export interface DaySpec {
  type: "last_business_day" | "last_day_of_month";
}

/** ISO-8601 calendar dates (e.g. "2026-01-01"). Supplied by the caller context. */
export interface RecurrenceBounds {
  start?: string; // inclusive
  end?: string; // exclusive
}

/** Dates or weekdays to skip. Exceptions are data, not model predictions. */
export interface ScheduleExceptions {
  dates?: string[];
  weekdays?: Weekday[];
}

/**
 * Public schedule AST, more expressive than cron. Environment-independent: no `Date`,
 * functions, timezone, or locale. At most one day selector may be set (resolver-enforced).
 * `weekdays` always normalizes ranges into an ordered list.
 */
export interface ScheduleIntent {
  frequency: {
    unit: FrequencyUnit;
    interval: number; // >= 1; "every other" is interval 2
  };
  time?: TimeOfDay | MinuteOfHour;
  weekdays?: Weekday[];
  dayOfMonth?: number; // 1-31
  ordinalWeekday?: OrdinalWeekday;
  day?: DaySpec;
  bounds?: RecurrenceBounds;
  exceptions?: ScheduleExceptions;
}
