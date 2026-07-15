import type { LocalDateTime, RecurrenceCalendar } from "./recurrence/calendar.js";

/** The small Temporal API used by the adapter. A host may supply native Temporal or a polyfill. */
export interface TemporalProvider {
  Instant: {
    fromEpochMilliseconds(value: number): {
      toZonedDateTimeISO(timezone: string): LocalDateTime;
    };
  };
  ZonedDateTime: {
    from(
      value: LocalDateTime & { timeZone: string; calendar: "iso8601" },
      options: { disambiguation: "earlier" | "later"; overflow: "reject" },
    ): { epochMilliseconds: number };
  };
}

/** No Temporal import or global initialization occurs inside this package. */
export function createTemporalCalendar(temporal: TemporalProvider): RecurrenceCalendar {
  if (!temporal?.Instant?.fromEpochMilliseconds || !temporal?.ZonedDateTime?.from) {
    throw new TypeError("provide the Temporal namespace");
  }

  return Object.freeze({
    toLocal(instant: number, timezone: string): LocalDateTime {
      const value = temporal.Instant.fromEpochMilliseconds(instant).toZonedDateTimeISO(timezone);

      return {
        year: value.year,
        month: value.month,
        day: value.day,
        hour: value.hour,
        minute: value.minute,
        second: value.second,
        millisecond: value.millisecond,
      };
    },
    resolve(local: LocalDateTime, timezone: string, disambiguation: "earlier" | "later"): number {
      return temporal.ZonedDateTime.from(
        { ...local, timeZone: timezone, calendar: "iso8601" },
        { disambiguation, overflow: "reject" },
      ).epochMilliseconds;
    },
  });
}
