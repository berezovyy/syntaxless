import type { FormatIntent } from "./types.js";
import { validate } from "./validate.js";

export function explain(intent: FormatIntent): string {
  validate(intent);
  const parts: string[] = [];

  if (intent.kind === "date-time") {
    if (intent.dateStyle) parts.push(`Format the date with ${intent.dateStyle} style.`);
    if (intent.timeStyle) parts.push(`Format the time with ${intent.timeStyle} style.`);

    if (intent.hourCycle) {
      parts.push(`Use ${intent.hourCycle === "h12" ? "12-hour" : "24-hour"} time.`);
    } else if (intent.timeStyle) {
      parts.push("Use the locale's default hour cycle.");
    }

    parts.push("Medium is the default style for a requested date or time without a named style.");
    parts.push("Use the Gregorian calendar, Latin digits and an exact millisecond timestamp.");
  } else {
    parts.push(
      intent.kind === "currency"
        ? `Format currency in ${intent.currency} with ${intent.currencyDisplay} display.`
        : intent.kind === "percent"
          ? "Format a percentage from a fractional value, so 0.125 represents 12.5 percent."
          : "Format a number with standard notation.",
    );

    const p = intent.precision;

    if (p?.minimum !== undefined) parts.push(`Show at least ${p.minimum} decimal places.`);
    if (p?.maximum !== undefined) parts.push(`Show at most ${p.maximum} decimal places.`);

    if (p?.minimum === undefined || p?.maximum === undefined) {
      parts.push("Unspecified precision uses Intl defaults for the selected style.");
    }

    parts.push(
      intent.grouping === undefined
        ? "Use locale grouping defaults."
        : intent.grouping
          ? "Enable locale grouping separators."
          : "Disable grouping separators.",
    );

    parts.push(
      intent.negative === "parentheses"
        ? "Wrap negative magnitudes in parentheses, including negative zero."
        : "Use the default minus presentation for negatives, including negative zero.",
    );

    parts.push("Use Latin digits and standard Intl rounding.");
  }

  parts.push(
    intent.locale ? `Use locale ${intent.locale}.` : "The caller must supply a supported locale.",
  );

  if (intent.kind === "date-time") {
    parts.push(
      intent.timeZone
        ? `Use timezone ${intent.timeZone}.`
        : "The caller must supply a supported timezone.",
    );
  }

  parts.push(
    "Instruction and caller context must agree. " +
      "Output names and punctuation depend on runtime locale data.",
  );

  return parts.join(" ");
}
