import { InputError } from "./types.js";
import type { Bucket, Window } from "./types.js";

export type Unit = "millisecond" | "second" | "minute" | "hour" | Bucket;

export const FIXED_MS: Partial<Record<Unit, number>> = {
  millisecond: 1,
  second: 1000,
  minute: 60000,
  hour: 3600000,
  day: 86400000,
  week: 604800000,
};

/** Convert the literal through exact fractions, accepting only safe whole output units. */
function scaled(literal: string, numerator: number, denominator = 1): number {
  const match = literal.match(/^\+?(\d+)(?:\.(\d+))?$/u);

  if (!match) throw new InputError("invalid-input", "provide a positive ordinary decimal window");
  const digits = match[2] ?? "";

  const top = BigInt(match[1]! + digits) * BigInt(numerator);
  const bottom = 10n ** BigInt(digits.length) * BigInt(denominator);

  if (top <= 0n || top % bottom || top / bottom > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new InputError("invalid-input", "window must resolve to positive safe whole units");
  }

  return Number(top / bottom);
}

export function calendarWindow(
  literal: string,
  unit: Unit,
  bucket: Bucket,
): Extract<Window, { kind: "calendar" }> {
  if (!["day", "week", "month", "year"].includes(unit)) {
    throw new InputError("unsupported", "calendar windows require days, weeks, months or years");
  }

  // Use calendar units, never a fixed number of days for months or years.
  if (unit === "week" && bucket !== "week") {
    return { kind: "calendar", unit: "day", count: scaled(literal, 7) };
  }

  if (unit === "year" && bucket !== "year") {
    return { kind: "calendar", unit: "month", count: scaled(literal, 12) };
  }

  return { kind: "calendar", unit: unit as Bucket, count: scaled(literal, 1) };
}

export function elapsedMs(literal: string, unit: Unit): number {
  const scale = FIXED_MS[unit];

  if (scale === undefined) {
    throw new InputError("unsupported", "calendar months and years are not elapsed durations");
  }

  return scaled(literal, scale);
}
