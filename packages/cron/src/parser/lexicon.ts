import type { Label } from "../model/labels.js";
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

/**
 * Plural weekday names. The renderer emits these (e.g. "Mondays") and the
 * aligner resolves them, so they belong to the surface vocabulary. The v0 rule
 * parser keeps its singular/abbreviated grammar and does not consume this map.
 */
export const WEEKDAY_PLURAL_ALIASES: Record<string, Weekday> = {
  mondays: "monday",
  tuesdays: "tuesday",
  wednesdays: "wednesday",
  thursdays: "thursday",
  fridays: "friday",
  saturdays: "saturday",
  sundays: "sunday",
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

/** Cardinal number words, so both the parser and the renderer spell numbers the
 * same way. `NUMBER_WORDS` is the lookup form; {@link numberWord} is the renderer
 * form and composes tens with ones. */
const NUMBER_ONES = [
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "eleven",
  "twelve",
  "thirteen",
  "fourteen",
  "fifteen",
  "sixteen",
  "seventeen",
  "eighteen",
  "nineteen",
] as const;

const NUMBER_TENS: Readonly<Record<number, string>> = {
  20: "twenty",
  30: "thirty",
  40: "forty",
  50: "fifty",
};

/** Spelled-out numbers accepted as intervals and clock hours. */
export const NUMBER_WORDS: Record<string, number> = (() => {
  const words: Record<string, number> = {};

  NUMBER_ONES.forEach((word, value) => {
    words[word] = value;
  });

  for (const [tens, word] of Object.entries(NUMBER_TENS)) {
    words[word] = Number(tens);
  }

  return words;
})();

/** Spell a non-negative integer the way the renderer emits it. */
export function numberWord(value: number): string {
  if (value < 20) return NUMBER_ONES[value]!;
  const tens = Math.floor(value / 10) * 10;
  const ones = value % 10;

  return ones === 0 ? NUMBER_TENS[tens]! : `${NUMBER_TENS[tens]!} ${NUMBER_ONES[ones]!}`;
}

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

/** Ordinal words in declaration order; carrier prose may draw on them. */
export const ORDINAL_WORDS: readonly Ordinal[] = Object.keys(ORDINAL_ALIASES) as Ordinal[];

/** Ordinal rank for the aligner: positions 1..4, `last` searches from the end. */
export const ORDINAL_RANK: Record<Ordinal, number> = {
  first: 1,
  second: 2,
  third: 3,
  fourth: 4,
  last: -1,
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

/**
 * Canonical surface strings the renderer draws from. Kept here, next to the
 * parser aliases, so there is exactly one copy of the runtime vocabulary; the
 * training renderer imports this module directly and never re-declares a word.
 */
export const WORDS = {
  recur: {
    every: "every",
    each: "each",
    recurring: "recurring",
    monthly: "monthly",
  },
  glue: {
    on: "on",
    the: "the",
    of: "of",
    at: "at",
    and: "and",
    apart: "apart",
    other: "other",
    dash: "-",
    colon: ":",
    comma: ",",
    ampersand: "&",
  },
  unit: {
    minute: "minute",
    hour: "hour",
    day: "day",
    week: "week",
    month: "month",
  },
  range: { from: "from", to: "to", through: "through" },
  bounds: {
    from: "from",
    to: "to",
    starting: "starting",
    until: "until",
    between: "between",
  },
  exception: {
    except: "except",
    excluding: "excluding",
    but: "but",
    not: "not",
  },
  meridiem: { am: "am", pm: "pm" },
  suffix: { st: "st", nd: "nd", rd: "rd", th: "th" },
  qualifier: {
    morning: "in the morning",
    afternoon: "in the afternoon",
    evening: "in the evening",
    night: "at night",
  },
} as const;

/** Canonical weekday display forms, derived from the one weekday sequence. */
function capitalize(word: string): string {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

export const WEEKDAY_NAME: Record<Weekday, string> = Object.fromEntries(
  WEEKDAY_SEQUENCE.map((weekday) => [weekday, capitalize(weekday)]),
) as Record<Weekday, string>;

export const WEEKDAY_PLURAL: Record<Weekday, string> = Object.fromEntries(
  WEEKDAY_SEQUENCE.map((weekday) => [weekday, `${capitalize(weekday)}s`]),
) as Record<Weekday, string>;

export const WEEKDAY_ABBR: Record<Weekday, string> = Object.fromEntries(
  WEEKDAY_SEQUENCE.map((weekday) => [weekday, capitalize(weekday).slice(0, 3)]),
) as Record<Weekday, string>;

/** One surface form and the runtime label it carries, with an optional value. */
export type SurfaceEntry = { label: Label; value?: string | number };

function surface(label: Label, value?: string | number): SurfaceEntry {
  return value === undefined ? { label } : { label, value };
}

/**
 * The single surface vocabulary. Ids are frozen once a corpus is built: adding a
 * key later is a corpus-format change. Built from the parser aliases and
 * {@link WORDS} so there is no second copy to drift.
 *
 * Vocabulary policy: a new schedule word is added here with a parser test
 * **before** it is generated, never after. A surface the generator can emit but
 * the lexicon cannot resolve is how labels drift.
 */
export const LEXICON: Record<string, SurfaceEntry> = (() => {
  const entries: Record<string, SurfaceEntry> = {};

  for (const [word, weekday] of Object.entries(WEEKDAY_ALIASES)) {
    entries[word] = surface("WEEKDAY", weekday);
  }

  for (const [word, weekday] of Object.entries(WEEKDAY_PLURAL_ALIASES)) {
    entries[word] = surface("WEEKDAY", weekday);
  }

  for (const word of Object.keys(WEEKDAY_GROUPS)) {
    entries[word] = surface("DAYGROUP", word);
  }

  for (const [word, unit] of Object.entries(UNIT_ALIASES)) {
    entries[word] = surface("INTERVAL", unit);
  }

  for (const [word, unit] of Object.entries(ADVERB_FREQUENCIES)) {
    entries[word] = surface("INTERVAL", unit);
  }

  for (const [word, ordinal] of Object.entries(ORDINAL_ALIASES)) {
    entries[word] = surface("ORDINAL", ORDINAL_RANK[ordinal]);
  }

  for (const [word, value] of Object.entries(NUMBER_WORDS)) {
    entries[word] = surface("COUNT", value);
  }

  for (const word of Object.values(WORDS.recur)) {
    entries[word] = surface("RECUR");
  }

  for (const word of Object.values(WORDS.glue)) {
    entries[word] = surface("GLUE");
  }

  for (const word of Object.values(WORDS.range)) {
    entries[word] = surface("GLUE");
  }

  for (const word of Object.values(WORDS.bounds)) {
    entries[word] = surface("BOUND");
  }

  for (const word of Object.values(WORDS.exception)) {
    entries[word] = surface("EXCEPT");
  }

  for (const word of Object.values(WORDS.meridiem)) {
    entries[word] = surface("MERIDIEM", word);
  }

  return entries;
})();

/** Read-only lookup over {@link LEXICON} for the aligner and verifier. */
export const LOOKUP: ReadonlyMap<string, SurfaceEntry> = new Map(Object.entries(LEXICON));
