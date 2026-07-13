/**
 * Canonical semantic form used for evaluation and oracle comparison.
 * Input arrays are never mutated.
 */
import type { ScheduleIntent } from "../../src/types/intent";

const WEEKDAY_ORDER = [
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
] as const;

function canonicalStrings<T extends string>(values: readonly T[] | undefined): T[] | undefined {
  return values === undefined ? undefined : [...new Set(values)].sort();
}

export function canonicalizeIntent(intent: ScheduleIntent): ScheduleIntent {
  const weekdayRank = new Map(WEEKDAY_ORDER.map((day, index) => [day, index]));

  const weekdays =
    intent.weekdays === undefined
      ? undefined
      : [...new Set(intent.weekdays)].sort(
          (left, right) => (weekdayRank.get(left) ?? 99) - (weekdayRank.get(right) ?? 99),
        );

  const exceptionWeekdays =
    intent.exceptions?.weekdays === undefined
      ? undefined
      : [...new Set(intent.exceptions.weekdays)].sort(
          (left, right) => (weekdayRank.get(left) ?? 99) - (weekdayRank.get(right) ?? 99),
        );

  const exceptionDates = canonicalStrings(intent.exceptions?.dates);

  return {
    ...intent,
    ...(weekdays === undefined ? {} : { weekdays }),
    ...(intent.exceptions === undefined
      ? {}
      : {
          exceptions: {
            ...(exceptionDates === undefined ? {} : { dates: exceptionDates }),
            ...(exceptionWeekdays === undefined ? {} : { weekdays: exceptionWeekdays }),
          },
        }),
  };
}
