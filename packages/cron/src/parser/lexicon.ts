import type { FrequencyUnit, Ordinal, Weekday } from "../types/intent.js";

/** Canonical weekday order used whenever a parsed list is normalized. Monday-first,
 * Sunday last, matching how cron renders day-of-week ranges. */
export const WEEKDAY_SEQUENCE: Weekday[] = [
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
];

/** Full weekday names plus the accepted abbreviations. */
export const WEEKDAY_ALIASES: Record<string, Weekday> = {
  monday: "monday",
  mon: "monday",
  tuesday: "tuesday",
  tue: "tuesday",
  wednesday: "wednesday",
  wed: "wednesday",
  thursday: "thursday",
  thu: "thursday",
  thurs: "thursday",
  friday: "friday",
  fri: "friday",
  saturday: "saturday",
  sat: "saturday",
  sunday: "sunday",
  sun: "sunday",
};

/** Words that expand to a fixed set of weekdays. */
export const WEEKDAY_GROUPS: Record<string, Weekday[]> = {
  weekday: ["monday", "tuesday", "wednesday", "thursday", "friday"],
  weekdays: ["monday", "tuesday", "wednesday", "thursday", "friday"],
  workday: ["monday", "tuesday", "wednesday", "thursday", "friday"],
  workdays: ["monday", "tuesday", "wednesday", "thursday", "friday"],
  weekend: ["saturday", "sunday"],
  weekends: ["saturday", "sunday"],
};

/** Spelled-out numbers accepted as intervals and clock hours. */
export const NUMBER_WORDS: Record<string, number> = {
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
  ten: 10,
  eleven: 11,
  twelve: 12,
  fifteen: 15,
  twenty: 20,
  thirty: 30,
  forty: 40,
  fifty: 50,
};

/** Recurrence units in singular and plural form. */
export const UNIT_ALIASES: Record<string, FrequencyUnit> = {
  minute: "minute",
  minutes: "minute",
  hour: "hour",
  hours: "hour",
  day: "day",
  days: "day",
  week: "week",
  weeks: "week",
  month: "month",
  months: "month",
};

/** Ordinal positions used by weekday rules. */
export const ORDINAL_ALIASES: Record<string, Ordinal> = {
  first: "first",
  second: "second",
  third: "third",
  fourth: "fourth",
  last: "last",
};

/** Adverbs that pin a single interval-1 unit without an `every` connective. */
export const ADVERB_FREQUENCIES: Record<string, FrequencyUnit> = {
  daily: "day",
  hourly: "hour",
};

/** Structural words the grammar consumes but never turns into intent fields. */
export const CONNECTIVES: ReadonlySet<string> = new Set([
  "every",
  "each",
  "other",
  "and",
  "through",
  "to",
  "at",
  "the",
  "on",
  "&",
]);
