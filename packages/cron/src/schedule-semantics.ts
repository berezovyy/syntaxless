/**
 * Shared surface ↔ intent rules for clocks and inclusive bounds.
 *
 * Renderer, decoder, and tests import this module so gold never renders a
 * clock or `through` date the decoder cannot invert. Nothing here reads raw
 * user text to assign labels.
 */
export function shiftIsoDate(iso: string, days: number): string | undefined {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);

  if (match === null) return undefined;
  const date = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]) + days));

  return `${String(date.getUTCFullYear()).padStart(4, "0")}-${String(
    date.getUTCMonth() + 1,
  ).padStart(2, "0")}-${String(date.getUTCDate()).padStart(2, "0")}`;
}

/** `through 2026-01-31` names an inclusive last day; intent bounds.end is exclusive. */
export function exclusiveEndFromInclusiveShown(iso: string): string | undefined {
  return shiftIsoDate(iso, 1);
}

/** Inverse of {@link exclusiveEndFromInclusiveShown} for gold rendering. */
export function inclusiveShownFromExclusiveEnd(iso: string): string | undefined {
  return shiftIsoDate(iso, -1);
}

export function hour12From24(hour24: number): number {
  return hour24 % 12 === 0 ? 12 : hour24 % 12;
}

export function meridiemFrom24(hour24: number): "am" | "pm" {
  return hour24 >= 12 ? "pm" : "am";
}

export function hour24From12(hour12: number, meridiem: "am" | "pm"): number {
  if (hour12 < 1 || hour12 > 12) throw new RangeError(`invalid 12-hour time: ${hour12}`);
  if (meridiem === "am") return hour12 === 12 ? 0 : hour12;

  return hour12 === 12 ? 12 : hour12 + 12;
}

/**
 * A shown hour without a meridiem is invertible iff it is already the 24h
 * value. `1:30` cannot name 13:30; render `13:30` or `1:30 pm`.
 */
export function hourWithoutMeridiemIsInvertible(shown: number, intentHour: number): boolean {
  return Number.isFinite(shown) && shown === intentHour;
}

/** DOM_LIT calendar payload: ISO `YYYY-MM-DD` digits only. */
export function parseDateLiteral(text: string): string | undefined {
  const iso = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text.trim());

  if (iso === null) return undefined;
  const year = Number(iso[1]);
  const month = Number(iso[2]);
  const day = Number(iso[3]);
  const candidate = new Date(Date.UTC(year, month - 1, day));

  if (
    year < 1 ||
    candidate.getUTCFullYear() !== year ||
    candidate.getUTCMonth() !== month - 1 ||
    candidate.getUTCDate() !== day
  ) {
    return undefined;
  }

  return `${iso[1]}-${iso[2]}-${iso[3]}`;
}
