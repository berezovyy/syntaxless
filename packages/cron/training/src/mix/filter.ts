/** Keep ordinary English that is safe to label as an easy negative. */
import {
  MONTH_ALIASES,
  UNIT_ABBREVIATIONS,
  WEEKDAY_ALIASES,
  WEEKDAY_PLURAL_ALIASES,
} from "../../../src/parser/lexicon.js";
import { normalize } from "../../../src/parser/normalize";
import { normalizeText, RESERVED_PHRASES } from "../background";

const TIME_SHAPED: readonly RegExp[] = [
  /\d{1,2}\s*:\s*\d{2}/,
  /\b\d{1,2}\s*(?:am|pm|a\.m|p\.m)\b/i,
  /\b\d{1,2}\s*[\/.-]\s*\d{1,2}\b/,
  /\b(?:19|20)\d{2}/,
  /\b\d{1,3}(?:st|nd|rd|th)\b/i,
];

const EXTRA_TIME_WORDS = [
  "afternoon",
  "ago",
  "annually",
  "biweekly",
  "century",
  "daily",
  "decade",
  "evening",
  "fortnight",
  "fortnightly",
  "hourly",
  "immediately",
  "later",
  "midday",
  "midnight",
  "monthly",
  "morning",
  "night",
  "nightly",
  "noon",
  "now",
  "o'clock",
  "oclock",
  "pm",
  "am",
  "second",
  "today",
  "tomorrow",
  "tonight",
  "upcoming",
  "weekday",
  "weekend",
  "weekly",
  "yearly",
  "yesterday",
  "every",
  "each",
  "remind",
  "reminder",
  "schedule",
  "cron",
  "recurrence",
  "recurring",
] as const;

const UNIT_WORDS = [
  "minute",
  "minutes",
  "hour",
  "hours",
  "day",
  "days",
  "week",
  "weeks",
  "month",
  "months",
  "year",
  "years",
  UNIT_ABBREVIATIONS.minute.singular,
  UNIT_ABBREVIATIONS.minute.plural,
  UNIT_ABBREVIATIONS.hour.singular,
  UNIT_ABBREVIATIONS.hour.plural,
  UNIT_ABBREVIATIONS.week.singular,
  UNIT_ABBREVIATIONS.week.plural,
  UNIT_ABBREVIATIONS.month.singular,
  UNIT_ABBREVIATIONS.month.plural,
] as const;

const TIME_WORD = new Set<string>([
  ...EXTRA_TIME_WORDS,
  ...Object.keys(WEEKDAY_ALIASES),
  ...Object.keys(WEEKDAY_PLURAL_ALIASES),
  ...Object.keys(MONTH_ALIASES),
  ...UNIT_WORDS,
]);

export type EasyDropReason =
  | "shape"
  | "length"
  | "normalization"
  | "timeShaped"
  | "timeWord"
  | "reserved"
  | "blocked"
  | "duplicate";

export interface EasyFilterOptions {
  /** Normalized texts that must not appear (human-v*, already kept). */
  blocked?: ReadonlySet<string>;
  seen?: Set<string>;
}

const SENTENCE_SHAPE = /^\p{Lu}[\p{L}\p{M}\d ,.;:'"!?()-]*[.!?]$/u;

export function timeShapedNumber(text: string): boolean {
  return TIME_SHAPED.some((pattern) => pattern.test(text));
}

export function hasTimeWord(text: string): boolean {
  for (const token of text.toLowerCase().match(/[\p{L}][\p{L}'’]*/gu) ?? []) {
    const normalized = token.replace(/’/g, "'");

    for (const word of [normalized, ...normalized.split("'")]) {
      if (word.length === 0) continue;
      if (TIME_WORD.has(word) || TIME_WORD.has(word.replace(/s$/, ""))) return true;
    }
  }

  return false;
}

export function withinEasyLength(text: string): boolean {
  const words = text.split(" ").filter(Boolean).length;

  return text.length >= 20 && text.length <= 160 && words >= 4 && words <= 26;
}

/** Why this sentence is not easy-no, or null if it may be kept. */
export function easyNoDropReason(
  text: string,
  options: EasyFilterOptions = {},
): EasyDropReason | null {
  if (!SENTENCE_SHAPE.test(text)) return "shape";
  if (!withinEasyLength(text)) return "length";
  // E labels describe the text seen by the model, not just the raw export.
  // Dotted meridiems and compatibility characters can expose time evidence
  // only after runtime normalization. Do not silently give it all-O gold.
  const runtimeText = normalize(text);

  if (normalize(runtimeText) !== runtimeText) return "normalization";
  if (timeShapedNumber(text) || timeShapedNumber(runtimeText)) return "timeShaped";
  if (hasTimeWord(text) || hasTimeWord(runtimeText)) return "timeWord";
  const normalized = normalizeText(text);

  if (RESERVED_PHRASES.has(normalized)) return "reserved";
  if (options.blocked?.has(normalized)) return "blocked";

  if (options.seen !== undefined) {
    if (options.seen.has(normalized)) return "duplicate";
    options.seen.add(normalized);
  }

  return null;
}

export function acceptEasyNo(text: string, options: EasyFilterOptions = {}): boolean {
  return easyNoDropReason(text, options) === null;
}
