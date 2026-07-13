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
  mnday: "monday",
  mon: "monday",
  "mon.": "monday",
  tuesday: "tuesday",
  tue: "tuesday",
  tues: "tuesday",
  "tue.": "tuesday",
  "tues.": "tuesday",
  wednesday: "wednesday",
  wensday: "wednesday",
  wed: "wednesday",
  "wed.": "wednesday",
  thursday: "thursday",
  thu: "thursday",
  thur: "thursday",
  thurs: "thursday",
  "thu.": "thursday",
  "thur.": "thursday",
  "thurs.": "thursday",
  friday: "friday",
  firday: "friday",
  fri: "friday",
  "fri.": "friday",
  saturday: "saturday",
  satrday: "saturday",
  sat: "saturday",
  "sat.": "saturday",
  sunday: "sunday",
  sun: "sunday",
  "sun.": "sunday",
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

/** Descriptive weekday aliases. This package's canonical week is explicitly
 * Monday-first (the same order used by cron rendering), so positional phrases
 * such as "the first day of the week" are deterministic rather than guessed. */
export const WEEKDAY_DESCRIPTIONS: Readonly<Record<string, Weekday>> = (() => {
  const entries: Record<string, Weekday> = {};
  const positions = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"];
  const cardinals = ["one", "two", "three", "four", "five", "six", "seven"];

  WEEKDAY_SEQUENCE.forEach((day, index) => {
    const previous = WEEKDAY_SEQUENCE[(index + 6) % 7]!;
    const next = WEEKDAY_SEQUENCE[(index + 1) % 7]!;

    entries[`the day after ${previous}`] = day;
    entries[`the day following ${previous}`] = day;
    entries[`the day immediately after ${previous}`] = day;
    entries[`the day directly after ${previous}`] = day;
    entries[`day directly after ${previous}`] = day;
    entries[`the day before ${next}`] = day;
    entries[`the day preceding ${next}`] = day;
    entries[`the day immediately before ${next}`] = day;
    entries[`the day directly before ${next}`] = day;
    entries[`day directly before ${next}`] = day;
    entries[`the ${positions[index]} day of the ISO week`] = day;
    entries[`the ${positions[index]} day in an ISO week`] = day;
    entries[`the ${positions[index]} day of a Monday-first week`] = day;
    entries[`the ${positions[index]} day of the week`] = day;
    entries[`${positions[index]} day of the week`] = day;
    entries[`day ${index + 1} of the week`] = day;
    entries[`day ${cardinals[index]} of the week`] = day;
    entries[`the week's ${positions[index]} day`] = day;
  });

  ["first", "second", "third", "fourth", "fifth"].forEach((position, index) => {
    const day = WEEKDAY_SEQUENCE[index]!;

    entries[`the ${position} workday of the week`] = day;
    entries[`${position} workday of the week`] = day;
    entries[`the ${position} working day of the week`] = day;
    entries[`the ${position} day of the working week`] = day;
  });

  entries["the start of the week"] = "monday";
  entries["the beginning of the week"] = "monday";
  entries["the start of the workweek"] = "monday";
  entries["the beginning of the workweek"] = "monday";
  entries["the first day of the weekend"] = "saturday";
  entries["the start of the weekend"] = "saturday";
  entries["the second day of the weekend"] = "sunday";
  entries["the last day of the weekend"] = "sunday";
  entries["the final day of the weekend"] = "sunday";
  entries["the end of the weekend"] = "sunday";
  entries["the last day of the week"] = "sunday";
  entries["the final day of the week"] = "sunday";
  entries["the end of the week"] = "sunday";

  return Object.fromEntries(
    Object.entries(entries).map(([text, day]) => [text.toLowerCase(), day]),
  );
})();

/** Words that expand to a fixed set of weekdays. */
export const WEEKDAY_GROUPS: Record<string, Weekday[]> = {
  weekday: ["monday", "tuesday", "wednesday", "thursday", "friday"],
  weekdays: ["monday", "tuesday", "wednesday", "thursday", "friday"],
  workday: ["monday", "tuesday", "wednesday", "thursday", "friday"],
  workdays: ["monday", "tuesday", "wednesday", "thursday", "friday"],
  "business day": ["monday", "tuesday", "wednesday", "thursday", "friday"],
  "business days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
  "working day": ["monday", "tuesday", "wednesday", "thursday", "friday"],
  "working days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
  "working week": ["monday", "tuesday", "wednesday", "thursday", "friday"],
  wkdays: ["monday", "tuesday", "wednesday", "thursday", "friday"],
  weekend: ["saturday", "sunday"],
  weekends: ["saturday", "sunday"],
  "weekend day": ["saturday", "sunday"],
  "weekend days": ["saturday", "sunday"],
  "monday to friday": ["monday", "tuesday", "wednesday", "thursday", "friday"],
  "monday through friday": ["monday", "tuesday", "wednesday", "thursday", "friday"],
};

export interface DayGroupExpression {
  text: string;
  prefix: string;
  surface: string;
  weekdays: Weekday[];
}

/** Complete, grammatical phrasings for the two built-in weekday groups. */
export const DAYGROUP_EXPRESSIONS: readonly DayGroupExpression[] = [
  "weekday",
  "workday",
  "business day",
  "working day",
]
  .flatMap((surface) => [
    {
      text: `every ${surface}`,
      prefix: "every",
      surface,
      weekdays: WEEKDAY_GROUPS[surface]!,
    },
    {
      text: `each ${surface}`,
      prefix: "each",
      surface,
      weekdays: WEEKDAY_GROUPS[surface]!,
    },
  ])
  .concat(
    ["weekdays", "workdays", "business days", "working days"].flatMap((surface) => [
      {
        text: `on ${surface}`,
        prefix: "on",
        surface,
        weekdays: WEEKDAY_GROUPS[surface]!,
      },
      {
        text: `weekly on ${surface}`,
        prefix: "weekly on",
        surface,
        weekdays: WEEKDAY_GROUPS[surface]!,
      },
      {
        text: `every week on ${surface}`,
        prefix: "every week on",
        surface,
        weekdays: WEEKDAY_GROUPS[surface]!,
      },
      {
        text: `each week on ${surface}`,
        prefix: "each week on",
        surface,
        weekdays: WEEKDAY_GROUPS[surface]!,
      },
    ]),
    ["weekend", "weekend day"].flatMap((surface) => [
      {
        text: `every ${surface}`,
        prefix: "every",
        surface,
        weekdays: WEEKDAY_GROUPS[surface]!,
      },
      {
        text: `each ${surface}`,
        prefix: "each",
        surface,
        weekdays: WEEKDAY_GROUPS[surface]!,
      },
    ]),
    ["weekends", "weekend days"].flatMap((surface) => [
      {
        text: `on ${surface}`,
        prefix: "on",
        surface,
        weekdays: WEEKDAY_GROUPS[surface]!,
      },
      {
        text: `every ${surface}`,
        prefix: "every",
        surface,
        weekdays: WEEKDAY_GROUPS[surface]!,
      },
      {
        text: `each ${surface}`,
        prefix: "each",
        surface,
        weekdays: WEEKDAY_GROUPS[surface]!,
      },
    ]),
    ["monday to friday", "monday through friday"].map((surface) => ({
      text: surface,
      prefix: "",
      surface,
      weekdays: WEEKDAY_GROUPS[surface]!,
    })),
  );

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
  minits: "minute",
  min: "minute",
  mins: "minute",
  hour: "hour",
  hours: "hour",
  hr: "hour",
  hrs: "hour",
  day: "day",
  days: "day",
  week: "week",
  weeks: "week",
  wk: "week",
  wks: "week",
  month: "month",
  months: "month",
  mo: "month",
  mos: "month",
};

export const UNIT_ABBREVIATIONS: Readonly<
  Record<FrequencyUnit, { singular: string; plural: string }>
> = {
  minute: { singular: "min", plural: "mins" },
  hour: { singular: "hr", plural: "hrs" },
  day: { singular: "day", plural: "days" },
  week: { singular: "wk", plural: "wks" },
  month: { singular: "mo", plural: "mos" },
};

/** Ordinal positions used by weekday rules. */
export const ORDINAL_ALIASES: Record<string, Ordinal> = {
  first: "first",
  "1st": "first",
  second: "second",
  "2nd": "second",
  third: "third",
  "3rd": "third",
  fourth: "fourth",
  "4th": "fourth",
  last: "last",
  final: "last",
};

/** Ordinal words in declaration order; carrier prose may draw on them. */
export const ORDINAL_WORDS: readonly string[] = Object.keys(ORDINAL_ALIASES);

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
  weekly: "week",
};

/** Idioms whose meaning cannot be obtained by reading one integer + unit. */
export const FREQUENCY_IDIOMS: Readonly<Record<string, { unit: FrequencyUnit; interval: number }>> =
  {
    "once a minute": { unit: "minute", interval: 1 },
    "once an hour": { unit: "hour", interval: 1 },
    "once a day": { unit: "day", interval: 1 },
    "once a week": { unit: "week", interval: 1 },
    "once a month": { unit: "month", interval: 1 },
    "once per minute": { unit: "minute", interval: 1 },
    "once per hour": { unit: "hour", interval: 1 },
    "once per day": { unit: "day", interval: 1 },
    "once per week": { unit: "week", interval: 1 },
    "once per month": { unit: "month", interval: 1 },
    "once each minute": { unit: "minute", interval: 1 },
    "once each hour": { unit: "hour", interval: 1 },
    "once each day": { unit: "day", interval: 1 },
    "once each week": { unit: "week", interval: 1 },
    "once each month": { unit: "month", interval: 1 },
    "one time per minute": { unit: "minute", interval: 1 },
    "one time per hour": { unit: "hour", interval: 1 },
    "one time per day": { unit: "day", interval: 1 },
    "one time per week": { unit: "week", interval: 1 },
    "one time per month": { unit: "month", interval: 1 },
    "one time a minute": { unit: "minute", interval: 1 },
    "one time an hour": { unit: "hour", interval: 1 },
    "one time a day": { unit: "day", interval: 1 },
    "one time a week": { unit: "week", interval: 1 },
    "one time a month": { unit: "month", interval: 1 },
    "each and every minute": { unit: "minute", interval: 1 },
    "each and every hour": { unit: "hour", interval: 1 },
    "each and every day": { unit: "day", interval: 1 },
    "each and every week": { unit: "week", interval: 1 },
    "each and every month": { unit: "month", interval: 1 },
    "per minute": { unit: "minute", interval: 1 },
    "per hour": { unit: "hour", interval: 1 },
    "per day": { unit: "day", interval: 1 },
    "per week": { unit: "week", interval: 1 },
    "per month": { unit: "month", interval: 1 },
    "once hourly": { unit: "hour", interval: 1 },
    "once daily": { unit: "day", interval: 1 },
    "once weekly": { unit: "week", interval: 1 },
    "once monthly": { unit: "month", interval: 1 },
    "on an hourly basis": { unit: "hour", interval: 1 },
    "on a daily basis": { unit: "day", interval: 1 },
    "on a weekly basis": { unit: "week", interval: 1 },
    "on a monthly basis": { unit: "month", interval: 1 },
    "minute by minute": { unit: "minute", interval: 1 },
    "hour by hour": { unit: "hour", interval: 1 },
    "day by day": { unit: "day", interval: 1 },
    "week by week": { unit: "week", interval: 1 },
    "month by month": { unit: "month", interval: 1 },
    "every sixty minutes": { unit: "hour", interval: 1 },
    "every twenty-four hours": { unit: "day", interval: 1 },
    "every seven days": { unit: "week", interval: 1 },
    "every fourteen days": { unit: "week", interval: 2 },
    "every other minute": { unit: "minute", interval: 2 },
    "every other hour": { unit: "hour", interval: 2 },
    "every other day": { unit: "day", interval: 2 },
    "every half hour": { unit: "minute", interval: 30 },
    "every half an hour": { unit: "minute", interval: 30 },
    "every half-hour": { unit: "minute", interval: 30 },
    "half-hourly": { unit: "minute", interval: 30 },
    "every quarter hour": { unit: "minute", interval: 15 },
    "every quarter of an hour": { unit: "minute", interval: 15 },
    "every quarter-hour": { unit: "minute", interval: 15 },
    "quarter-hourly": { unit: "minute", interval: 15 },
    fortnightly: { unit: "week", interval: 2 },
    "every fortnight": { unit: "week", interval: 2 },
    "once a fortnight": { unit: "week", interval: 2 },
    "once every fortnight": { unit: "week", interval: 2 },
    "every other week": { unit: "week", interval: 2 },
    "every other month": { unit: "month", interval: 2 },
    "alternate minutes": { unit: "minute", interval: 2 },
    "alternate hours": { unit: "hour", interval: 2 },
    "alternate days": { unit: "day", interval: 2 },
    "alternate weeks": { unit: "week", interval: 2 },
    "alternate months": { unit: "month", interval: 2 },
    "alternating minutes": { unit: "minute", interval: 2 },
    "alternating hours": { unit: "hour", interval: 2 },
    "alternating days": { unit: "day", interval: 2 },
    "alternating months": { unit: "month", interval: 2 },
    "on alternate weeks": { unit: "week", interval: 2 },
    "on alternating weeks": { unit: "week", interval: 2 },
  };

export const NAMED_HOURS: Readonly<Record<string, number>> = {
  noon: 12,
  midday: 12,
  "twelve noon": 12,
  midnight: 0,
  midnite: 0,
  "twelve midnight": 0,
  lunchtime: 12,
};

export const MINUTE_IDIOMS: Readonly<Record<string, number>> = {
  "a quarter": 15,
  quarter: 15,
  half: 30,
  "a half": 30,
};

/** Ordinals in calendar dates, distinct from an ordinal weekday selector. */
export const DAY_ORDINALS: readonly string[] = [
  "first",
  "second",
  "third",
  "fourth",
  "fifth",
  "sixth",
  "seventh",
  "eighth",
  "ninth",
  "tenth",
  "eleventh",
  "twelfth",
  "thirteenth",
  "fourteenth",
  "fifteenth",
  "sixteenth",
  "seventeenth",
  "eighteenth",
  "nineteenth",
  "twentieth",
  "twenty-first",
  "twenty-second",
  "twenty-third",
  "twenty-fourth",
  "twenty-fifth",
  "twenty-sixth",
  "twenty-seventh",
  "twenty-eighth",
  "twenty-ninth",
  "thirtieth",
  "thirty-first",
];

/** Unambiguous English month names used inside labeled calendar dates. Numeric
 * slash dates are intentionally absent because their day/month order is locale
 * dependent. A trailing period on an abbreviation is accepted by the date
 * normalizer rather than duplicated in this table. */
export const MONTH_ALIASES: Readonly<Record<string, number>> = {
  january: 1,
  jan: 1,
  february: 2,
  feb: 2,
  march: 3,
  mar: 3,
  april: 4,
  apr: 4,
  may: 5,
  june: 6,
  jun: 6,
  july: 7,
  jul: 7,
  august: 8,
  aug: 8,
  september: 9,
  sep: 9,
  sept: 9,
  october: 10,
  oct: 10,
  november: 11,
  nov: 11,
  december: 12,
  dec: 12,
};

export const MONTH_NAMES: readonly string[] = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

export const MONTH_ABBREVIATIONS: readonly string[] = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

export const DAY_RULES = {
  "last day": "last_day_of_month",
  "very last day": "last_day_of_month",
  "final day": "last_day_of_month",
  "last calendar day": "last_day_of_month",
  "final calendar day": "last_day_of_month",
  "last business day": "last_business_day",
  "final business day": "last_business_day",
  "last working day": "last_business_day",
  "final working day": "last_business_day",
} as const;

export const BOUND_ALIASES: Readonly<
  Record<string, "start" | "end" | "end-inclusive" | "between">
> = {
  from: "start",
  starting: "start",
  "starting on": "start",
  "beginning on": "start",
  "on or after": "start",
  "starting from": "start",
  "beginning from": "start",
  effective: "start",
  "effective on": "start",
  commencing: "start",
  "commencing on": "start",
  to: "end",
  until: "end",
  before: "end",
  "ending before": "end",
  "stopping before": "end",
  "up until": "end",
  "but not on or after": "end",
  "ending on": "end-inclusive",
  ending: "end",
  "through and including": "end-inclusive",
  between: "between",
};

/** Wording inventories used by deterministic training templates. Entries are
 * intentionally explicit: similar-looking ambiguous phrases do not belong. */
export const RECURRENCE_PREFIXES = [
  "every",
  "each",
  "once every",
  "once per",
  "repeating every",
  "recurring every",
  "at intervals of",
] as const;

export const UNIT_INTERVAL_PREFIXES = [
  "every",
  "each",
  "once every",
  "once per",
  "every single",
] as const;

export const WEEKDAY_LIST_PREFIXES = [
  "on",
  "every",
  "each",
  "weekly on",
  "every week on",
  "each week on",
] as const;

export const WEEKDAY_RANGE_CONNECTORS = ["through", "to", "thru", "up to and including"] as const;

export const EXCEPTION_PREFIXES = [
  "except",
  "except on",
  "excluding",
  "skip",
  "skipping",
  "omit",
  "omitting",
  "but not",
  "but not on",
  "other than",
  "apart from",
  "save for",
  "although not on",
  "unless the day is",
] as const;

export const START_BOUND_PREFIXES = [
  "from",
  "starting",
  "starting on",
  "starting from",
  "beginning on",
  "beginning from",
  "on or after",
  "effective",
  "effective on",
  "commencing",
  "commencing on",
] as const;

export const END_BOUND_PREFIXES = [
  "until",
  "before",
  "up until",
  "ending before",
  "stopping before",
  "but not on or after",
  "ending",
] as const;

export const INCLUSIVE_END_BOUND_PREFIXES = ["ending on", "through and including"] as const;

export const CLOCK_PREFIXES = [
  "at",
  "for",
  "scheduled for",
  "at exactly",
  "exactly at",
  "precisely at",
] as const;

export const CLOCK_EXACT_SUFFIXES = ["sharp", "on the dot"] as const;
export const INTERVAL_FRAME_SUFFIXES = ["intervals"] as const;
export const PAST_CONNECTORS = ["past", "after"] as const;
export const TO_CONNECTORS = ["to", "before"] as const;

export const MONTHLY_PREFIXES = [
  "monthly on",
  "every month on",
  "each month on",
  "once a month on",
] as const;

export const ORDINAL_MONTH_FRAMES = [
  "the {ordinal} {weekday} of every month",
  "the {ordinal} {weekday} of each month",
  "every month's {ordinal} {weekday}",
  "each month's {ordinal} {weekday}",
  "monthly on the {ordinal} {weekday}",
  "on the {ordinal} {weekday} every month",
] as const;

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
    once: "once",
    repeating: "repeating",
    recurring: "recurring",
    monthly: "monthly",
  },
  glue: {
    on: "on",
    the: "the",
    of: "of",
    at: "at",
    in: "in",
    and: "and",
    apart: "apart",
    after: "after",
    before: "before",
    past: "past",
    per: "per",
    single: "single",
    scheduled: "scheduled",
    for: "for",
    other: "other",
    alternate: "alternate",
    alternating: "alternating",
    dash: "-",
    colon: ":",
    comma: ",",
    ampersand: "&",
    slash: "/",
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
  correction: {
    iMeant: "i meant",
    ratherThan: "rather than",
    insteadOf: "instead of",
    use: "use",
    sorry: "sorry",
    not: "not",
    makeThat: "make that",
    correction: "correction",
    dash: "—",
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

  for (const [word, weekday] of Object.entries(WEEKDAY_DESCRIPTIONS)) {
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

  for (const word of Object.keys(MONTH_ALIASES)) entries[word] = surface("DOM");

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

  for (const word of Object.values(WORDS.qualifier)) {
    entries[word] = surface("MERIDIEM");
  }

  entries["a.m."] = surface("MERIDIEM", "am");
  entries["p.m."] = surface("MERIDIEM", "pm");
  entries["a.m"] = surface("MERIDIEM", "am");
  entries["p.m"] = surface("MERIDIEM", "pm");
  entries["a. m."] = surface("MERIDIEM", "am");
  entries["p. m."] = surface("MERIDIEM", "pm");

  for (const word of Object.keys(FREQUENCY_IDIOMS)) entries[word] = surface("INTERVAL");
  for (const [word, hour] of Object.entries(NAMED_HOURS)) entries[word] = surface("HOUR", hour);

  for (const [word, minute] of Object.entries(MINUTE_IDIOMS)) {
    entries[word] = surface("MINUTE", minute);
  }

  for (const word of Object.keys(DAY_RULES)) {
    entries[word] = surface("DOM", DAY_RULES[word as keyof typeof DAY_RULES]);
  }

  for (const word of Object.keys(BOUND_ALIASES)) entries[word] = surface("BOUND");

  DAY_ORDINALS.forEach((word, index) => {
    entries[word] ??= surface("DOM", index + 1);
  });

  for (const phrase of RECURRENCE_PREFIXES) entries[phrase] ??= surface("RECUR");
  for (const phrase of UNIT_INTERVAL_PREFIXES) entries[phrase] ??= surface("RECUR");
  for (const phrase of WEEKDAY_LIST_PREFIXES) entries[phrase] ??= surface("RECUR");
  for (const phrase of WEEKDAY_RANGE_CONNECTORS) entries[phrase] ??= surface("GLUE");
  for (const phrase of EXCEPTION_PREFIXES) entries[phrase] ??= surface("EXCEPT");
  for (const phrase of CLOCK_PREFIXES) entries[phrase] ??= surface("GLUE");
  for (const phrase of CLOCK_EXACT_SUFFIXES) entries[phrase] = surface("EDGE");
  for (const phrase of INTERVAL_FRAME_SUFFIXES) entries[phrase] = surface("EDGE");
  entries["the interval is"] = surface("GLUE");
  entries["the cadence is"] = surface("GLUE");
  entries["the cycle is"] = surface("GLUE");
  entries["the period is"] = surface("GLUE");
  entries["the spacing is"] = surface("GLUE");
  entries["the stride is"] = surface("GLUE");
  entries["the step is"] = surface("GLUE");
  entries["the gap is"] = surface("GLUE");
  entries["the rhythm is"] = surface("GLUE");
  entries["the days are"] = surface("GLUE");
  entries["days are"] = surface("GLUE");
  entries["the selected days are"] = surface("GLUE");
  entries["the weekdays are"] = surface("GLUE");
  entries["the active days are"] = surface("GLUE");
  entries["the chosen days are"] = surface("GLUE");
  entries["the span runs"] = surface("GLUE");
  entries["the window runs"] = surface("GLUE");
  entries["the range runs"] = surface("GLUE");
  entries["the stretch runs"] = surface("GLUE");
  entries["the coverage runs"] = surface("GLUE");
  entries["the omitted days are"] = surface("EXCEPT");
  entries["the skipped days are"] = surface("EXCEPT");
  entries["the exceptions are"] = surface("EXCEPT");
  entries["a cycle of"] = surface("GLUE");
  entries["between runs"] = surface("EDGE");
  entries["the recurrence interval is"] = surface("GLUE");
  entries["a repeat interval of"] = surface("GLUE");
  entries["runs on a"] = surface("RECUR");
  entries["on a cadence of"] = surface("RECUR");
  entries["cycle"] = surface("EDGE");
  entries["apart"] = surface("EDGE");
  for (const phrase of PAST_CONNECTORS) entries[phrase] ??= surface("GLUE");
  for (const phrase of TO_CONNECTORS) entries[phrase] ??= surface("GLUE");

  return entries;
})();

/** Read-only lookup over {@link LEXICON} for the aligner and verifier. */
export const LOOKUP: ReadonlyMap<string, SurfaceEntry> = new Map(Object.entries(LEXICON));
