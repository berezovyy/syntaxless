import { parseDateLiteral } from "../schedule-semantics.js";

export interface LocalDateTime {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
  millisecond: number;
}

/** Gregorian civil fields and timezone conversions, with offsets strictly below 24 hours. */
export interface RecurrenceCalendar {
  toLocal(instant: number, timezone: string): LocalDateTime;
  resolve(local: LocalDateTime, timezone: string, disambiguation: "earlier" | "later"): number;
}

export type RecurrenceFailureCode =
  | "invalid-context"
  | "missing-calendar"
  | "invalid-calendar"
  | "nonexistent-time"
  | "repeated-time"
  | "month-end"
  | "work-limit"
  | "output-limit";

export class RecurrenceError extends Error {
  constructor(
    public readonly code: RecurrenceFailureCode,
    message: string,
  ) {
    super(message);
    this.name = "RecurrenceError";
  }
}

export const DAY = 86_400_000;
export const MIN_DATE = -62_135_596_800_000; // 0001-01-01T00:00:00Z
export const MAX_DATE = 253_402_300_800_000; // 10000-01-01T00:00:00Z, exclusive

export function fromEpoch(value: number): LocalDateTime {
  const date = new Date(value);

  return {
    year: date.getUTCFullYear(),
    month: date.getUTCMonth() + 1,
    day: date.getUTCDate(),
    hour: date.getUTCHours(),
    minute: date.getUTCMinutes(),
    second: date.getUTCSeconds(),
    millisecond: date.getUTCMilliseconds(),
  };
}

const zeroPadded = (value: number, width: number): string => String(value).padStart(width, "0");

export function dateText(local: Pick<LocalDateTime, "year" | "month" | "day">): string {
  return `${zeroPadded(local.year, 4)}-${zeroPadded(local.month, 2)}-${zeroPadded(local.day, 2)}`;
}

export function civilEpoch(local: LocalDateTime): number {
  if (
    !local ||
    parseDateLiteral(dateText(local)) !== dateText(local) ||
    ![
      local.year,
      local.month,
      local.day,
      local.hour,
      local.minute,
      local.second,
      local.millisecond,
    ].every(Number.isSafeInteger) ||
    local.hour < 0 ||
    local.hour > 23 ||
    local.minute < 0 ||
    local.minute > 59 ||
    local.second < 0 ||
    local.second > 59 ||
    local.millisecond < 0 ||
    local.millisecond > 999
  ) {
    throw new RecurrenceError(
      "invalid-calendar",
      "calendar conversions must return valid Gregorian civil fields",
    );
  }

  return (
    new Date(`${dateText(local)}T00:00:00Z`).getTime() +
    local.hour * 3_600_000 +
    local.minute * 60_000 +
    local.second * 1000 +
    local.millisecond
  );
}

export function sameLocal(a: LocalDateTime, b: LocalDateTime): boolean {
  return civilEpoch(a) === civilEpoch(b);
}

function offset(timezone: string): number {
  if (["UTC", "Etc/UTC", "Z"].includes(timezone)) return 0;
  const match = /^([+-])(\d{2}):(\d{2})$/.exec(timezone);

  if (match && Number(match[2]) <= 23 && Number(match[3]) <= 59) {
    return (match[1] === "-" ? -1 : 1) * (Number(match[2]) * 60 + Number(match[3])) * 60_000;
  }

  throw new RecurrenceError(
    "missing-calendar",
    "IANA timezones require an explicit recurrence calendar adapter",
  );
}

export const FIXED_CALENDAR: RecurrenceCalendar = Object.freeze({
  toLocal: (instant: number, timezone: string) => fromEpoch(instant + offset(timezone)),
  resolve: (local: LocalDateTime, timezone: string) => civilEpoch(local) - offset(timezone),
});

/**
 * Absolute millisecond precision only, without Date.parse rollover or an implicit host
 * timezone.
 */
export function instantValue(value: unknown): number {
  if (
    typeof value === "number" &&
    Number.isSafeInteger(value) &&
    value >= MIN_DATE &&
    value < MAX_DATE
  ) {
    return value;
  }

  if (typeof value !== "string") {
    throw new RecurrenceError(
      "invalid-context",
      "provide an absolute ISO instant or safe epoch milliseconds",
    );
  }

  const match =
    /^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,3}))?)?(Z|[+-]\d{2}:\d{2})$/.exec(
      value,
    );

  if (!match || parseDateLiteral(match[1]!) !== match[1]) {
    throw new RecurrenceError(
      "invalid-context",
      "instant must be a valid ISO date and time with an explicit offset",
    );
  }

  const [year, month, day] = match[1]!.split("-").map(Number);

  try {
    const result =
      civilEpoch({
        year: year!,
        month: month!,
        day: day!,
        hour: Number(match[2]),
        minute: Number(match[3]),
        second: Number(match[4] ?? 0),
        millisecond: Number((match[5] ?? "0").padEnd(3, "0")),
      }) - offset(match[6]!);

    if (result >= MIN_DATE && result < MAX_DATE) return result;
  } catch {
    /* Report input errors as context errors, rather than adapter failures. */
  }

  throw new RecurrenceError(
    "invalid-context",
    "instant is outside the supported date or time range",
  );
}
