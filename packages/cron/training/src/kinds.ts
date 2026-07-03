/**
 * Sentence-kind catalog for the backward corpus generator.
 *
 * Coverage is kinds, not paraphrases of one frame. A kind is one way a
 * schedule is said (unit interval, day group, every-other, compound time,
 * …) together with the intents that kind can name. Templates are alternate
 * wordings of that kind; held-out templates use a different frame so train
 * and held-out fingerprints cannot collide.
 *
 * Surfaces come only from the runtime lexicon. Labels match the Phase 07
 * renderer / Phase 17 decoder contract: unit words are `GLUE`, numbers are
 * `INTERVAL`, adverbs `hourly`/`daily` are `INTERVAL`, groups are `DAYGROUP`.
 *
 * No LLM is consulted here. Wording variety is the template list.
 */
import {
  ADVERB_FREQUENCIES,
  CLOCK_EXACT_SUFFIXES,
  CLOCK_PREFIXES,
  DAY_ORDINALS,
  DAY_RULES,
  DAYGROUP_EXPRESSIONS,
  END_BOUND_PREFIXES,
  EXCEPTION_PREFIXES,
  FREQUENCY_IDIOMS,
  INCLUSIVE_END_BOUND_PREFIXES,
  INTERVAL_FRAME_SUFFIXES,
  MONTH_ABBREVIATIONS,
  MONTH_NAMES,
  MONTHLY_PREFIXES,
  numberWord,
  PAST_CONNECTORS,
  RECURRENCE_PREFIXES,
  START_BOUND_PREFIXES,
  TO_CONNECTORS,
  UNIT_ABBREVIATIONS,
  UNIT_INTERVAL_PREFIXES,
  WEEKDAY_ABBR,
  WEEKDAY_ALIASES,
  WEEKDAY_DESCRIPTIONS,
  WEEKDAY_GROUPS,
  WEEKDAY_LIST_PREFIXES,
  WEEKDAY_NAME,
  WEEKDAY_PLURAL,
  WEEKDAY_RANGE_CONNECTORS,
  WEEKDAY_SEQUENCE,
  WORDS,
} from "../../src/parser/lexicon.js";
import type { FrequencyUnit, ScheduleIntent, TimeOfDay, Weekday } from "../../src/types/intent";
import type { RenderedExample } from "./renderer";
import type { Family } from "./sampler";
import { assertValid } from "./sampler";
import { Sentence } from "./sentence";

export type KindId =
  | "unit-interval"
  | "n-interval"
  | "daygroup"
  | "single-weekday"
  | "weekday-list"
  | "weekday-range"
  | "every-other"
  | "every-other-at-time"
  | "time-of-day"
  | "day-at-time"
  | "interval-at-time"
  | "daygroup-at-time"
  | "weekday-at-time"
  | "day-of-month"
  | "day-of-month-at-time"
  | "ordinal-weekday"
  | "ordinal-weekday-at-time"
  | "bounds"
  | "bounds-at-time"
  | "exception-group"
  | "exception-days"
  | "exception-date"
  | "exception-date-at-time"
  | "exception-at-time"
  | "selector-exception-at-time"
  | "selector-exception"
  | "frequency-idiom"
  | "day-rule"
  | "day-rule-at-time"
  | "negative";

export type KindSplit = "train" | "heldout";
export type ExampleSplit = "train" | "validation" | "heldout";

export interface KindTemplate {
  id: string;
  split: KindSplit;
}

export interface KindExample {
  kind: KindId;
  family: Family;
  template: string;
  split: ExampleSplit;
  intent: ScheduleIntent | null;
  example: RenderedExample;
}

const UNITS: readonly FrequencyUnit[] = ["minute", "hour", "day", "week", "month"];

const integers = (first: number, last: number): number[] =>
  Array.from({ length: last - first + 1 }, (_, index) => first + index);

/** Full valid minute/hour intervals and a practical finite domain for calendar
 * units whose AST permits arbitrary positive integers. */
const N_INTERVALS: Readonly<Record<FrequencyUnit, readonly number[]>> = {
  minute: integers(2, 59),
  hour: integers(2, 23),
  day: integers(2, 31),
  week: integers(2, 12),
  month: integers(2, 12),
};

const CLOCKS: readonly TimeOfDay[] = integers(0, 1439).map((value) => ({
  hour: Math.floor(value / 60),
  minute: value % 60,
}));

const COMPOUND_CLOCKS = CLOCKS;

const WEEKDAY_SETS: readonly Weekday[][] = integers(1, 127).map((mask) =>
  WEEKDAY_SEQUENCE.filter((_, index) => (mask & (1 << index)) !== 0),
);

const WEEKDAY_INTENTS = WEEKDAY_SETS.filter((weekdays) => weekdays.length > 1);

const WEEKDAY_RUNS: readonly Weekday[][] = WEEKDAY_SEQUENCE.flatMap((_, start) =>
  integers(start + 1, 6).map((end) => WEEKDAY_SEQUENCE.slice(start, end + 1)),
);

const DAYGROUPS: readonly { surface: string; weekdays: Weekday[] }[] = [
  { surface: "weekday", weekdays: WEEKDAY_GROUPS.weekday! },
  { surface: "weekend", weekdays: WEEKDAY_GROUPS.weekend! },
];

const HELD_OUT_DAYGROUPS: readonly { surface: string; weekdays: Weekday[] }[] = [
  { surface: "weekdays", weekdays: WEEKDAY_GROUPS.weekdays! },
  { surface: "weekends", weekdays: WEEKDAY_GROUPS.weekends! },
];

const DOM_DAYS: readonly number[] = integers(1, 31);

const WEEKDAY_ALIAS_ENTRIES = Object.entries(WEEKDAY_ALIASES) as Array<[string, Weekday]>;

const WEEKDAY_DESCRIPTION_ENTRIES = Object.entries(WEEKDAY_DESCRIPTIONS) as Array<
  [string, Weekday]
>;

function calendarDates(year: number): string[] {
  const result: string[] = [];

  for (let month = 1; month <= 12; month++) {
    const days = new Date(Date.UTC(year, month, 0)).getUTCDate();

    for (let day = 1; day <= days; day++) {
      result.push(`${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`);
    }
  }

  return result;
}

const CALENDAR_DATE_LATTICE: readonly string[] = [...calendarDates(2026), "2028-02-29"];

const MONTH_RANGES_2026: readonly { start: string; end: string }[] = integers(1, 12).map(
  (month) => ({
    start: `2026-${String(month).padStart(2, "0")}-01`,
    end: month === 12 ? "2027-01-01" : `2026-${String(month + 1).padStart(2, "0")}-01`,
  }),
);

const NEGATIVE_GROUPS = {
  ordinary: [
    "Please open the file.",
    "Could you review the document for Alex?",
    "The report contains twelve examples.",
    "Sam wrote the next page.",
    "Choose option four from section two.",
    "Send the invoice to the finance team.",
    "Archive this conversation.",
    "Turn on dark mode.",
    "Rename the project to Apollo.",
    "Add Priya to the workspace.",
    "The build completed successfully.",
    "I prefer the second proposal.",
  ],
  incomplete: [
    "every",
    "every other",
    "once per",
    "at",
    "at half past",
    "on Monday at",
    "from Tuesday through",
    "the last Friday of",
    "weekdays except",
    "starting on",
    "until the",
    "monthly on the",
    "repeat this every",
    "schedule it for",
  ],
  ambiguous: [
    "Monday works for me.",
    "Friday might be better.",
    "Morning is probably fine.",
    "Sometime after nine would work.",
    "Let's discuss it next week.",
    "The first day is the hardest.",
    "We could do this monthly.",
    "Every option has drawbacks.",
    "I visit often on weekends.",
    "Nine people joined on Tuesday.",
    "The deadline moved from Monday to Friday.",
    "The shop is open daily.",
    "Hourly billing is expensive.",
    "This report is produced every month by another system.",
  ],
  corrective: [
    "No, not Monday.",
    "Forget what I said about Friday.",
    "Cancel the weekday idea.",
    "That time was a mistake.",
    "I did not mean every hour.",
    "Ignore the previous schedule.",
    "Actually, never mind.",
    "Do not create that recurrence.",
    "Scratch the nine o'clock suggestion.",
    "The monthly option is wrong.",
    "I am correcting the date, not scheduling anything.",
    "Stop; the schedule is still undecided.",
  ],
  conversational: [
    "When would you like to meet?",
    "Does Monday work for everyone?",
    "What time is the appointment?",
    "Should this happen every week?",
    "Are weekends included?",
    "Which Friday did you mean?",
    "Do you want morning or afternoon?",
    "Is this a recurring task?",
    "Can we decide the schedule later?",
    "Why was the reminder moved?",
    "Who changed it to Tuesday?",
    "Where can I see the current schedule?",
  ],
  quoted: [
    "The documentation uses the example 'every Monday'.",
    "She wrote 'at nine' in the message.",
    "The phrase 'once a month' appears twice.",
    "Translate 'every weekday' into French.",
    "Explain what 'every other Friday' means.",
    "Search for messages containing 'daily at noon'.",
    "The button is labeled 'schedule weekly'.",
    "Remove the words 'at midnight' from the title.",
    "He asked whether 'Monday through Friday' was clear.",
    "The test case named 'every hour' is failing.",
    "Copy 'the last Friday' to the clipboard.",
    "Bold the text 'weekdays except Friday'.",
  ],
  meta: [
    "Show me the existing schedule.",
    "Delete the current schedule.",
    "Is there already a reminder?",
    "Why did this run at midnight?",
    "List all recurring jobs.",
    "Disable scheduling for this task.",
    "The cron expression is invalid.",
    "This parser does not understand Tuesdays.",
    "How does an hourly schedule work?",
    "Compare daily and weekly recurrence.",
    "Do not change any dates.",
    "Keep the schedule exactly as it is.",
  ],
  temporal: [
    "The meeting happened Monday at nine.",
    "I finished the report last Friday.",
    "We met at noon yesterday.",
    "The server failed twice this morning.",
    "Alex arrived at half past eight.",
    "The office closes on weekends.",
    "Sales increased every month last year.",
    "The train leaves at six, according to the timetable.",
    "I used to call every Sunday.",
    "The alarm rang hourly during the test.",
    "They worked Monday through Friday in June.",
    "The invoice was due on the first of May.",
  ],
} as const;

const NEGATIVES: readonly { category: string; text: string }[] = Object.entries(
  NEGATIVE_GROUPS,
).flatMap(([category, texts]) => texts.map((text) => ({ category, text })));

/** Natural carriers deliberately distinct from the reserved promotion corpus. */
const POSITIVE_CARRIERS = [
  "set a reminder for",
  "can you schedule this for",
  "please make this happen",
  "I'd like this to run",
  "actually, change it to",
  "instead, make the schedule",
] as const;

/** Semantically complete base frames that receive conversational carriers.
 * Clock-only schedules deliberately cover several surface families: otherwise
 * a frame-disjoint split can leave the model with no training example where
 * prose is followed directly by an HOUR clause start. Carrier text is O; the
 * schedule spans and intent remain unchanged. */
const CARRIER_BASES: Readonly<Partial<Record<KindId, readonly string[]>>> = {
  "unit-interval": ["every-unit"],
  "n-interval": [
    "every-n",
    "interval-is",
    "cadence-is",
    "cycle-of",
    "between-runs",
    "recurrence-interval-is",
    "repeat-interval-of",
    "runs-on-cycle",
    "cadence-of",
    "spaced-apart",
  ],
  daygroup: ["expression-0"],
  "single-weekday": ["every-day"],
  "weekday-list": ["prefix-0-full"],
  "weekday-range": ["from-to"],
  "every-other": ["every-other"],
  "time-of-day": [
    "digits",
    "hour-meridiem",
    "spoken",
    "24h",
    "compact",
    "dotted-meridiem",
    "oclock",
  ],
  "day-at-time": [
    "every-day-at",
    "daily-at",
    "time-daily",
    "digits-daily",
    "compact-daily",
    "24h-daily",
    "digits-every-day",
    "digits-each-day",
    "daily-digits",
    "daily-basis-at",
  ],
  "daygroup-at-time": ["expression-0"],
  "day-of-month": ["on-the"],
  "day-of-month-at-time": ["the-at"],
  "ordinal-weekday": ["the-ordinal-of"],
  "ordinal-weekday-at-time": ["ordinal-of-each-at"],
  bounds: ["from-to"],
  "exception-group": ["except-weekends"],
};

function receivesCarrier(kind: KindId, template: string): boolean {
  if (CARRIER_BASES[kind]?.includes(template)) return true;

  return (
    kind === "single-weekday" &&
    (template.includes("-alias-") || template.includes("-description-surface-"))
  );
}

function plural(unit: FrequencyUnit, count: number): string {
  return count === 1 ? unit : `${unit}s`;
}

function abbreviatedUnit(unit: FrequencyUnit, count: number): string {
  const abbreviation = UNIT_ABBREVIATIONS[unit];

  return count === 1 ? abbreviation.singular : abbreviation.plural;
}

function hour12(hour24: number): number {
  return hour24 % 12 === 0 ? 12 : hour24 % 12;
}

function meridiem(hour24: number): string {
  return hour24 >= 12 ? WORDS.meridiem.pm : WORDS.meridiem.am;
}

function paddedMinute(minute: number): string {
  return String(minute).padStart(2, "0");
}

function ordinalSuffix(day: number): string {
  if (day % 100 >= 11 && day % 100 <= 13) return WORDS.suffix.th;

  switch (day % 10) {
    case 1:
      return WORDS.suffix.st;
    case 2:
      return WORDS.suffix.nd;
    case 3:
      return WORDS.suffix.rd;
    default:
      return WORDS.suffix.th;
  }
}

type WeekdayMode = "full" | "plural" | "abbrev" | "dotted-abbrev";

function weekdayText(weekday: Weekday, mode: WeekdayMode): string {
  switch (mode) {
    case "full":
      return WEEKDAY_NAME[weekday];
    case "plural":
      return WEEKDAY_PLURAL[weekday];
    case "abbrev":
      return WEEKDAY_ABBR[weekday];
    case "dotted-abbrev":
      return `${WEEKDAY_ABBR[weekday]}.`;
  }
}

function addWeekdays(
  sentence: Sentence,
  weekdays: readonly Weekday[],
  mode: WeekdayMode,
  connector: string,
): void {
  weekdays.forEach((weekday, index) => {
    if (index > 0) {
      sentence.add(connector, "GLUE", connector === WORDS.glue.comma ? "" : " ");
    }

    sentence.add(weekdayText(weekday, mode), "WEEKDAY");
  });
}

function addSlashWeekdays(sentence: Sentence, weekdays: readonly Weekday[]): void {
  weekdays.forEach((weekday, index) => {
    if (index > 0) sentence.add(WORDS.glue.slash, "GLUE", "");
    sentence.add(WEEKDAY_ABBR[weekday], "WEEKDAY", index > 0 ? "" : " ");
  });
}

type WeekdayListStyle = WeekdayMode | "comma-and" | "ampersand";

function addWeekdayList(
  sentence: Sentence,
  weekdays: readonly Weekday[],
  style: WeekdayListStyle,
): void {
  if (style !== "comma-and" && style !== "ampersand") {
    addWeekdays(sentence, weekdays, style, WORDS.glue.and);

    return;
  }

  weekdays.forEach((weekday, index) => {
    if (index > 0) {
      const last = index === weekdays.length - 1;

      sentence.add(
        style === "ampersand" && last
          ? WORDS.glue.ampersand
          : last
            ? WORDS.glue.and
            : WORDS.glue.comma,
        "GLUE",
        !last ? "" : " ",
      );
    }

    sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
  });
}

function addDom(sentence: Sentence, day: number): void {
  sentence.add(String(day), "DOM");
  sentence.add(ordinalSuffix(day), "GLUE", "");
}

type DateStyle =
  "iso" | "month-day" | "month-ordinal" | "abbreviated" | "dotted-abbreviated" | "day-first";

const DATE_STYLES: readonly DateStyle[] = [
  "iso",
  "month-day",
  "month-ordinal",
  "abbreviated",
  "dotted-abbreviated",
  "day-first",
];

function calendarDateSurface(iso: string, style: DateStyle): string | undefined {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);

  if (match === null) return undefined;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const name = MONTH_NAMES[month - 1];
  const abbreviation = MONTH_ABBREVIATIONS[month - 1];

  if (name === undefined || abbreviation === undefined || day < 1 || day > 31) return undefined;

  switch (style) {
    case "iso":
      return iso;
    case "month-day":
      return `${name} ${day}, ${year}`;
    case "month-ordinal":
      return `${name} ${day}${ordinalSuffix(day)}, ${year}`;
    case "abbreviated":
      return `${abbreviation} ${day}, ${year}`;
    case "dotted-abbreviated":
      return `${abbreviation}. ${day}, ${year}`;
    case "day-first":
      return `${day} ${name} ${year}`;
  }
}

function offsetIsoDate(iso: string, days: number): string | undefined {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);

  if (match === null) return undefined;
  const date = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]) + days));

  return `${String(date.getUTCFullYear()).padStart(4, "0")}-${String(
    date.getUTCMonth() + 1,
  ).padStart(2, "0")}-${String(date.getUTCDate()).padStart(2, "0")}`;
}

function addCalendarDate(sentence: Sentence, iso: string, style: DateStyle): boolean {
  const surface = calendarDateSurface(iso, style);

  if (surface === undefined) return false;
  sentence.add(surface, "DOM");

  return true;
}

function addInclusiveEndDate(
  sentence: Sentence,
  exclusiveEnd: string,
  style: DateStyle,
  prefix: string,
): boolean {
  const inclusiveDate = offsetIsoDate(exclusiveEnd, -1);

  if (inclusiveDate === undefined) return false;
  sentence.add(prefix, "BOUND");

  return addCalendarDate(sentence, inclusiveDate, style);
}

function sameWeekdays(actual: readonly Weekday[], expected: readonly Weekday[]): boolean {
  return (
    actual.length === expected.length &&
    actual.every((weekday, index) => weekday === expected[index])
  );
}

/** Render the selector half of a compound schedule without assuming that it is
 * daily. The decoder derives weekly frequency from an explicit weekday set. */
function addScheduleSelector(sentence: Sentence, intent: ScheduleIntent): boolean {
  const weekdays = intent.weekdays;

  if (weekdays === undefined) {
    if (intent.frequency.unit !== "day" || intent.frequency.interval !== 1) return false;
    sentence.add(WORDS.recur.every, "RECUR");
    sentence.add(WORDS.unit.day, "GLUE");

    return true;
  }

  if (intent.frequency.unit !== "week" || intent.frequency.interval !== 1) return false;
  sentence.add(WORDS.glue.on, "GLUE");

  if (sameWeekdays(weekdays, WEEKDAY_GROUPS.weekdays!)) {
    sentence.add("weekdays", "DAYGROUP");
  } else if (sameWeekdays(weekdays, WEEKDAY_GROUPS.weekends!)) {
    sentence.add("weekends", "DAYGROUP");
  } else {
    addWeekdayList(sentence, weekdays, "comma-and");
  }

  return true;
}

type ClockStyle =
  | "digits"
  | "hour-meridiem"
  | "spoken"
  | "24h"
  | "compact"
  | "military"
  | "named"
  | "quarter-past"
  | "quarter-after"
  | "half-past"
  | "half-after"
  | "quarter-to"
  | "quarter-before"
  | "minute-past"
  | "minute-after"
  | "minute-to"
  | "minute-before"
  | "dotted-meridiem"
  | "compact-dotted-meridiem"
  | "spaced-dotted-meridiem"
  | "oclock"
  | "sharp"
  | "on-the-dot"
  | "qualified";

function qualifierForHour(hour: number): string | undefined {
  if (hour >= 5 && hour <= 11) return WORDS.qualifier.morning;
  if (hour >= 13 && hour <= 16) return WORDS.qualifier.afternoon;
  if (hour >= 17 && hour <= 21) return WORDS.qualifier.evening;
  if (hour >= 22 && hour <= 23) return WORDS.qualifier.night;

  return undefined;
}

function addClock(
  sentence: Sentence,
  time: TimeOfDay,
  style: ClockStyle,
  prefix: string = WORDS.glue.at,
): void {
  const { hour, minute } = time;

  switch (style) {
    case "digits":
      sentence.add(prefix, "GLUE");
      sentence.add(String(hour12(hour)), "HOUR");
      sentence.add(WORDS.glue.colon, "GLUE", "");
      sentence.add(paddedMinute(minute), "MINUTE", "");
      sentence.add(meridiem(hour), "MERIDIEM");
      break;
    case "hour-meridiem":
      if (minute !== 0) return;
      sentence.add(prefix, "GLUE");
      sentence.add(String(hour12(hour)), "HOUR");
      sentence.add(meridiem(hour), "MERIDIEM");
      break;
    case "spoken":
      sentence.add(prefix, "GLUE");
      sentence.add(numberWord(hour12(hour)), "HOUR");

      if (minute !== 0) {
        sentence.add(minute < 10 ? `oh ${numberWord(minute)}` : numberWord(minute), "MINUTE");
      }

      sentence.add(meridiem(hour), "MERIDIEM");
      break;
    case "24h":
      sentence.add(prefix, "GLUE");
      sentence.add(String(hour), "HOUR");
      sentence.add(WORDS.glue.colon, "GLUE", "");
      sentence.add(paddedMinute(minute), "MINUTE", "");
      break;
    case "compact":
      sentence.add(prefix, "GLUE");
      sentence.add(String(hour12(hour)), "HOUR");
      sentence.add(WORDS.glue.colon, "GLUE", "");
      sentence.add(paddedMinute(minute), "MINUTE", "");
      sentence.add(meridiem(hour), "MERIDIEM", "");
      break;
    case "military":
      sentence.add(prefix, "GLUE");
      sentence.add(`${String(hour).padStart(2, "0")}${paddedMinute(minute)}`, "HOUR");
      sentence.add("hours", "EDGE");
      break;
    case "named":
      if (minute !== 0 || (hour !== 0 && hour !== 12)) return;
      sentence.add(prefix, "GLUE");
      sentence.add(hour === 0 ? "midnight" : "noon", "HOUR");
      break;
    case "quarter-past":
      if (minute !== 15) return;
      sentence.add(prefix, "GLUE");
      sentence.add("quarter", "MINUTE");
      sentence.add(PAST_CONNECTORS[0], "GLUE");
      sentence.add(numberWord(hour12(hour)), "HOUR");
      sentence.add(meridiem(hour), "MERIDIEM");
      break;
    case "quarter-after":
      if (minute !== 15) return;
      sentence.add(prefix, "GLUE");
      sentence.add("quarter", "MINUTE");
      sentence.add(PAST_CONNECTORS[1], "GLUE");
      sentence.add(numberWord(hour12(hour)), "HOUR");
      sentence.add(meridiem(hour), "MERIDIEM");
      break;
    case "half-past":
      if (minute !== 30) return;
      sentence.add(prefix, "GLUE");
      sentence.add("half", "MINUTE");
      sentence.add(PAST_CONNECTORS[0], "GLUE");
      sentence.add(numberWord(hour12(hour)), "HOUR");
      sentence.add(meridiem(hour), "MERIDIEM");
      break;
    case "half-after":
      if (minute !== 30) return;
      sentence.add(prefix, "GLUE");
      sentence.add("half", "MINUTE");
      sentence.add(PAST_CONNECTORS[1], "GLUE");
      sentence.add(numberWord(hour12(hour)), "HOUR");
      sentence.add(meridiem(hour), "MERIDIEM");
      break;

    case "quarter-to": {
      if (minute !== 45) return;
      const targetHour = (hour + 1) % 24;

      sentence.add(prefix, "GLUE");
      sentence.add("quarter", "MINUTE");
      sentence.add(TO_CONNECTORS[0], "GLUE");
      sentence.add(numberWord(hour12(targetHour)), "HOUR");
      sentence.add(meridiem(targetHour), "MERIDIEM");
      break;
    }

    case "quarter-before": {
      if (minute !== 45) return;
      const targetHour = (hour + 1) % 24;

      sentence.add(prefix, "GLUE");
      sentence.add("quarter", "MINUTE");
      sentence.add(TO_CONNECTORS[1], "GLUE");
      sentence.add(numberWord(hour12(targetHour)), "HOUR");
      sentence.add(meridiem(targetHour), "MERIDIEM");
      break;
    }

    case "minute-past":
      if (minute < 1 || minute > 30) return;
      sentence.add(prefix, "GLUE");
      sentence.add(numberWord(minute), "MINUTE");
      sentence.add(`${plural("minute", minute)} past`, "GLUE");
      sentence.add(numberWord(hour12(hour)), "HOUR");
      sentence.add(meridiem(hour), "MERIDIEM");
      break;
    case "minute-after":
      if (minute < 1 || minute > 30) return;
      sentence.add(prefix, "GLUE");
      sentence.add(numberWord(minute), "MINUTE");
      sentence.add(`${plural("minute", minute)} after`, "GLUE");
      sentence.add(numberWord(hour12(hour)), "HOUR");
      sentence.add(meridiem(hour), "MERIDIEM");
      break;

    case "minute-to": {
      if (minute < 31) return;
      const remaining = 60 - minute;
      const targetHour = (hour + 1) % 24;

      sentence.add(prefix, "GLUE");
      sentence.add(numberWord(remaining), "MINUTE");
      sentence.add(`${plural("minute", remaining)} to`, "GLUE");
      sentence.add(numberWord(hour12(targetHour)), "HOUR");
      sentence.add(meridiem(targetHour), "MERIDIEM");
      break;
    }

    case "minute-before": {
      if (minute < 31) return;
      const remaining = 60 - minute;
      const targetHour = (hour + 1) % 24;

      sentence.add(prefix, "GLUE");
      sentence.add(numberWord(remaining), "MINUTE");
      sentence.add(`${plural("minute", remaining)} before`, "GLUE");
      sentence.add(numberWord(hour12(targetHour)), "HOUR");
      sentence.add(meridiem(targetHour), "MERIDIEM");
      break;
    }

    case "dotted-meridiem":
      sentence.add(prefix, "GLUE");
      sentence.add(String(hour12(hour)), "HOUR");
      sentence.add(WORDS.glue.colon, "GLUE", "");
      sentence.add(paddedMinute(minute), "MINUTE", "");
      sentence.add(meridiem(hour) === "am" ? "a.m." : "p.m.", "MERIDIEM");
      break;
    case "compact-dotted-meridiem":
      sentence.add(prefix, "GLUE");
      sentence.add(String(hour12(hour)), "HOUR");
      sentence.add(WORDS.glue.colon, "GLUE", "");
      sentence.add(paddedMinute(minute), "MINUTE", "");
      sentence.add(meridiem(hour) === "am" ? "a.m." : "p.m.", "MERIDIEM", "");
      break;
    case "spaced-dotted-meridiem":
      sentence.add(prefix, "GLUE");
      sentence.add(String(hour12(hour)), "HOUR");
      sentence.add(WORDS.glue.colon, "GLUE", "");
      sentence.add(paddedMinute(minute), "MINUTE", "");
      sentence.add(meridiem(hour) === "am" ? "a. m." : "p. m.", "MERIDIEM");
      break;
    case "oclock":
      if (minute !== 0 || qualifierForHour(hour) === undefined) return;
      sentence.add(prefix, "GLUE");
      sentence.add(numberWord(hour12(hour)), "HOUR");
      sentence.add("o'clock", "EDGE");
      sentence.add(qualifierForHour(hour)!, "MERIDIEM");
      break;
    case "sharp":
      sentence.add(prefix, "GLUE");
      sentence.add(String(hour12(hour)), "HOUR");
      sentence.add(WORDS.glue.colon, "GLUE", "");
      sentence.add(paddedMinute(minute), "MINUTE", "");
      sentence.add(meridiem(hour), "MERIDIEM");
      sentence.add(CLOCK_EXACT_SUFFIXES[0], "EDGE");
      break;
    case "on-the-dot":
      if (minute !== 0) return;
      sentence.add(prefix, "GLUE");

      if (hour === 0 || hour === 12) {
        sentence.add(hour === 0 ? "midnight" : "noon", "HOUR");
      } else {
        sentence.add(numberWord(hour12(hour)), "HOUR");
        sentence.add(meridiem(hour), "MERIDIEM");
      }

      sentence.add(CLOCK_EXACT_SUFFIXES[1], "EDGE");
      break;

    case "qualified": {
      const qualifier = qualifierForHour(hour);

      if (qualifier === undefined) return;
      sentence.add(prefix, "GLUE");
      sentence.add(numberWord(hour12(hour)), "HOUR");

      if (minute !== 0) {
        sentence.add(minute < 10 ? `oh ${numberWord(minute)}` : numberWord(minute), "MINUTE");
      }

      sentence.add(qualifier, "MERIDIEM");
      break;
    }
  }
}

function clockCanRender(time: TimeOfDay, style: ClockStyle): boolean {
  if (style === "hour-meridiem") return time.minute === 0;
  if (style === "named") return time.minute === 0 && (time.hour === 0 || time.hour === 12);
  if (style === "quarter-past") return time.minute === 15;
  if (style === "quarter-after") return time.minute === 15;
  if (style === "half-past") return time.minute === 30;
  if (style === "half-after") return time.minute === 30;
  if (style === "quarter-to") return time.minute === 45;
  if (style === "quarter-before") return time.minute === 45;
  if (style === "minute-past") return time.minute >= 1 && time.minute <= 30;
  if (style === "minute-after") return time.minute >= 1 && time.minute <= 30;
  if (style === "minute-to") return time.minute >= 31;
  if (style === "minute-before") return time.minute >= 31;
  if (style === "oclock") return time.minute === 0 && qualifierForHour(time.hour) !== undefined;
  if (style === "on-the-dot") return time.minute === 0;
  if (style === "qualified") return (time.hour >= 5 && time.hour <= 11) || time.hour >= 13;

  return true;
}

function weekly(weekdays: readonly Weekday[], time?: TimeOfDay): ScheduleIntent {
  return {
    frequency: { unit: "week", interval: 1 },
    weekdays: [...weekdays],
    ...(time === undefined ? {} : { time }),
  };
}

function daily(time?: TimeOfDay): ScheduleIntent {
  return {
    frequency: { unit: "day", interval: 1 },
    ...(time === undefined ? {} : { time }),
  };
}

function adverbFor(unit: FrequencyUnit): string | undefined {
  for (const [word, value] of Object.entries(ADVERB_FREQUENCIES)) {
    if (value === unit) return word;
  }

  return undefined;
}

function weekdayDescription(weekday: Weekday, marker: string): string | undefined {
  return Object.entries(WEEKDAY_DESCRIPTIONS).find(
    ([text, value]) => value === weekday && text.includes(marker),
  )?.[0];
}

function finish(sentence: Sentence, template: string): RenderedExample {
  return {
    text: sentence.text,
    spans: sentence.spans,
    template,
    style: template.slice(template.indexOf("/") + 1),
  };
}

function renderUnitInterval(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const unit = intent.frequency.unit;

  if (templateId.startsWith("prefix-")) {
    const prefix = UNIT_INTERVAL_PREFIXES[Number(templateId.slice(7))];

    if (prefix === undefined) return false;
    sentence.add(prefix, "RECUR");
    sentence.add(plural(unit, 1), "GLUE");

    return true;
  }

  switch (templateId) {
    case "every-unit":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(plural(unit, 1), "GLUE");

      return true;
    case "each-unit":
      sentence.add(WORDS.recur.each, "RECUR");
      sentence.add(plural(unit, 1), "GLUE");

      return true;

    case "adverb": {
      const adverb = adverbFor(unit);

      if (adverb !== undefined) {
        sentence.add(adverb, "INTERVAL");

        return true;
      }

      if (unit === "month") {
        sentence.add(WORDS.recur.monthly, "RECUR");

        return true;
      }

      sentence.add(WORDS.recur.recurring, "RECUR");
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(plural(unit, 1), "GLUE");

      return true;
    }

    default:
      return false;
  }
}

function renderNInterval(sentence: Sentence, intent: ScheduleIntent, templateId: string): boolean {
  const { unit, interval } = intent.frequency;

  if (templateId === "every-abbrev" || templateId === "every-word-abbrev") {
    sentence.add(WORDS.recur.every, "RECUR");

    sentence.add(
      templateId === "every-word-abbrev" ? numberWord(interval) : String(interval),
      "INTERVAL",
    );

    sentence.add(abbreviatedUnit(unit, interval), "GLUE");

    return true;
  }

  if (
    templateId === "at-hyphenated" ||
    templateId === "in-hyphenated" ||
    templateId === "every-hyphenated" ||
    templateId === "on-a-hyphenated-cycle"
  ) {
    if (templateId === "at-hyphenated") sentence.add(WORDS.glue.at, "GLUE");
    else if (templateId === "in-hyphenated") sentence.add(WORDS.glue.in, "GLUE");
    else if (templateId === "every-hyphenated") sentence.add(WORDS.recur.every, "RECUR");
    else sentence.add("on a", "RECUR");
    sentence.add(String(interval), "INTERVAL");
    sentence.add(WORDS.glue.dash, "GLUE", "");
    sentence.add(unit, "GLUE", "");

    sentence.add(
      templateId === "on-a-hyphenated-cycle" ? "cycle" : INTERVAL_FRAME_SUFFIXES[0],
      "EDGE",
    );

    return true;
  }

  if (templateId.endsWith("-word")) {
    const prefix = templateId.startsWith("once-")
      ? "once every"
      : templateId.startsWith("intervals-")
        ? "at intervals of"
        : WORDS.recur.every;

    sentence.add(prefix, "RECUR");
    sentence.add(numberWord(interval), "INTERVAL");
    sentence.add(plural(unit, interval), "GLUE");

    return true;
  }

  if (templateId.startsWith("prefix-")) {
    const prefix = RECURRENCE_PREFIXES[Number(templateId.slice(7))];

    if (prefix === undefined || prefix === "once per") return false;
    sentence.add(prefix, "RECUR");
    sentence.add(String(interval), "INTERVAL");
    sentence.add(plural(unit, interval), "GLUE");

    return true;
  }

  if (
    templateId === "interval-is" ||
    templateId === "cadence-is" ||
    templateId === "cycle-of" ||
    templateId === "between-runs" ||
    templateId === "recurrence-interval-is" ||
    templateId === "repeat-interval-of" ||
    templateId === "runs-on-cycle" ||
    templateId === "cadence-of" ||
    templateId === "spaced-apart"
  ) {
    if (templateId === "interval-is") sentence.add("the interval is", "GLUE");
    if (templateId === "cadence-is") sentence.add("the cadence is", "GLUE");
    if (templateId === "cycle-of") sentence.add("a cycle of", "GLUE");
    if (templateId === "recurrence-interval-is") sentence.add("the recurrence interval is", "GLUE");
    if (templateId === "repeat-interval-of") sentence.add("a repeat interval of", "GLUE");
    if (templateId === "runs-on-cycle") sentence.add("runs on a", "RECUR");
    if (templateId === "cadence-of") sentence.add("on a cadence of", "RECUR");
    sentence.add(String(interval), "INTERVAL");
    sentence.add(plural(unit, interval), "GLUE");
    if (templateId === "between-runs") sentence.add("between runs", "EDGE");
    if (templateId === "runs-on-cycle") sentence.add("cycle", "EDGE");
    if (templateId === "spaced-apart") sentence.add("apart", "EDGE");

    return true;
  }

  switch (templateId) {
    case "every-n":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(String(interval), "INTERVAL");
      sentence.add(plural(unit, interval), "GLUE");

      return true;
    case "each-n":
      sentence.add(WORDS.recur.each, "RECUR");
      sentence.add(String(interval), "INTERVAL");
      sentence.add(plural(unit, interval), "GLUE");

      return true;
    case "recurring-every-n":
      sentence.add(WORDS.recur.recurring, "RECUR");
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(String(interval), "INTERVAL");
      sentence.add(plural(unit, interval), "GLUE");

      return true;
    case "once-per-n":
      sentence.add("once per", "RECUR");
      sentence.add(String(interval), "INTERVAL");
      sentence.add(plural(unit, interval), "GLUE");

      return true;
    case "every-n-on-repeat":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(String(interval), "INTERVAL");
      sentence.add(plural(unit, interval), "GLUE");
      sentence.add("on repeat", "EDGE");

      return true;
    default:
      return false;
  }
}

function renderDaygroup(sentence: Sentence, intent: ScheduleIntent, templateId: string): boolean {
  const weekdays = intent.weekdays!;

  if (templateId.startsWith("expression-")) {
    const expression = DAYGROUP_EXPRESSIONS[Number(templateId.slice(11))];

    if (
      expression === undefined ||
      expression.weekdays.length !== weekdays.length ||
      !expression.weekdays.every((day, index) => day === weekdays[index])
    ) {
      return false;
    }

    if (expression.prefix !== "") {
      sentence.add(expression.prefix, expression.prefix === "on" ? "GLUE" : "RECUR");
    }

    sentence.add(expression.surface, "DAYGROUP");

    return true;
  }

  const train = DAYGROUPS.find(
    (group) =>
      group.weekdays.length === weekdays.length &&
      group.weekdays.every((day, index) => day === weekdays[index]),
  );

  const held = HELD_OUT_DAYGROUPS.find(
    (group) =>
      group.weekdays.length === weekdays.length &&
      group.weekdays.every((day, index) => day === weekdays[index]),
  );

  switch (templateId) {
    case "every-group":
      if (train === undefined) return false;
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(train.surface, "DAYGROUP");

      return true;
    case "each-group":
      if (held === undefined) return false;
      sentence.add(WORDS.recur.each, "RECUR");
      sentence.add(held.surface, "DAYGROUP");

      return true;
    default:
      return false;
  }
}

function renderSingleWeekday(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const weekday = intent.weekdays![0]!;
  const alias = /^(every|on|bare)-alias-(\d+)$/.exec(templateId);

  if (alias !== null) {
    const entry = WEEKDAY_ALIAS_ENTRIES[Number(alias[2])];

    if (entry === undefined || entry[1] !== weekday) return false;
    if (alias[1] === "every") sentence.add(WORDS.recur.every, "RECUR");
    if (alias[1] === "on") sentence.add(WORDS.glue.on, "GLUE");
    sentence.add(entry[0], "WEEKDAY");

    return true;
  }

  const descriptionSurface = /^(every|on|bare)-description-surface-(\d+)$/.exec(templateId);

  if (descriptionSurface !== null) {
    const entry = WEEKDAY_DESCRIPTION_ENTRIES[Number(descriptionSurface[2])];

    if (entry === undefined || entry[1] !== weekday) return false;
    if (descriptionSurface[1] === "every") sentence.add(WORDS.recur.every, "RECUR");
    if (descriptionSurface[1] === "on") sentence.add(WORDS.glue.on, "GLUE");
    sentence.add(entry[0], "WEEKDAY");

    return true;
  }

  if (templateId.startsWith("description-")) {
    const marker =
      templateId === "description-after"
        ? "day after"
        : templateId === "description-following"
          ? "day following"
          : templateId === "description-immediately-after"
            ? "day immediately after"
            : templateId === "description-before"
              ? "day before"
              : templateId === "description-preceding"
                ? "day preceding"
                : templateId === "description-immediately-before"
                  ? "day immediately before"
                  : templateId === "description-iso"
                    ? "ISO week"
                    : "Monday-first week";

    const description = weekdayDescription(weekday, marker);

    if (description === undefined) return false;
    sentence.add(WORDS.recur.every, "RECUR");
    sentence.add(description, "WEEKDAY");

    return true;
  }

  switch (templateId) {
    case "every-day":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");

      return true;
    case "on-day":
      sentence.add(WORDS.glue.on, "GLUE");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");

      return true;
    case "each-day":
      sentence.add(WORDS.recur.each, "RECUR");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");

      return true;
    case "plural":
      sentence.add(WEEKDAY_PLURAL[weekday], "WEEKDAY");

      return true;
    case "on-plural":
      sentence.add(WORDS.glue.on, "GLUE");
      sentence.add(WEEKDAY_PLURAL[weekday], "WEEKDAY");

      return true;
    case "bare":
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");

      return true;
    default:
      return false;
  }
}

function renderWeekdayList(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const weekdays = intent.weekdays!;

  if (templateId.startsWith("prefix-")) {
    const match = /^prefix-(\d+)-(.+)$/.exec(templateId);

    if (match === null) return false;
    const prefix = WEEKDAY_LIST_PREFIXES[Number(match[1])];

    if (prefix === undefined) return false;
    sentence.add(prefix, prefix === WORDS.glue.on ? "GLUE" : "RECUR");
    addWeekdayList(sentence, weekdays, match[2] as WeekdayListStyle);

    return true;
  }

  switch (templateId) {
    case "on-days":
      sentence.add(WORDS.glue.on, "GLUE");
      addWeekdays(sentence, weekdays, "full", WORDS.glue.and);

      return true;
    case "every-days":
      sentence.add(WORDS.recur.every, "RECUR");
      addWeekdays(sentence, weekdays, "plural", WORDS.glue.and);

      return true;
    case "abbrev":
      sentence.add(WORDS.glue.on, "GLUE");
      addWeekdays(sentence, weekdays, "abbrev", WORDS.glue.comma);

      return true;
    case "bare-plural":
      addWeekdays(sentence, weekdays, "plural", WORDS.glue.and);

      return true;
    case "slash-abbrev":
      addSlashWeekdays(sentence, weekdays);

      return true;
    default:
      return false;
  }
}

function renderWeekdayRange(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const weekdays = intent.weekdays!;
  const first = weekdays[0]!;
  const last = weekdays[weekdays.length - 1]!;

  if (templateId.startsWith("connector-")) {
    const connector = WEEKDAY_RANGE_CONNECTORS[Number(templateId.slice(10))];

    if (connector === undefined) return false;
    sentence.add(WEEKDAY_NAME[first], "WEEKDAY");
    sentence.add(connector, "GLUE");
    sentence.add(WEEKDAY_NAME[last], "WEEKDAY");

    return true;
  }

  switch (templateId) {
    case "through":
      sentence.add(WEEKDAY_NAME[first], "WEEKDAY");
      sentence.add(WORDS.range.through, "GLUE");
      sentence.add(WEEKDAY_NAME[last], "WEEKDAY");

      return true;
    case "from-to":
      sentence.add(WORDS.range.from, "GLUE");
      sentence.add(WEEKDAY_NAME[first], "WEEKDAY");
      sentence.add(WORDS.range.to, "GLUE");
      sentence.add(WEEKDAY_NAME[last], "WEEKDAY");

      return true;
    case "on-through":
      sentence.add(WORDS.glue.on, "GLUE");
      sentence.add(WEEKDAY_NAME[first], "WEEKDAY");
      sentence.add(WORDS.range.through, "GLUE");
      sentence.add(WEEKDAY_NAME[last], "WEEKDAY");

      return true;
    case "weekly-from-to":
      sentence.add("weekly", "INTERVAL");
      sentence.add(WORDS.range.from, "GLUE");
      sentence.add(WEEKDAY_NAME[first], "WEEKDAY");
      sentence.add(WORDS.range.to, "GLUE");
      sentence.add(WEEKDAY_NAME[last], "WEEKDAY");

      return true;
    case "each-week-up-to-inclusive":
      sentence.add("each week", "RECUR");
      sentence.add(WEEKDAY_NAME[first], "WEEKDAY");
      sentence.add("up to and including", "GLUE");
      sentence.add(WEEKDAY_NAME[last], "WEEKDAY");

      return true;
    case "dash":
      sentence.add(WEEKDAY_ABBR[first], "WEEKDAY");
      sentence.add(WORDS.glue.dash, "GLUE", "");
      sentence.add(WEEKDAY_ABBR[last], "WEEKDAY");

      return true;
    case "full-dash":
      sentence.add(WEEKDAY_NAME[first], "WEEKDAY");
      sentence.add(WORDS.glue.dash, "GLUE", "");
      sentence.add(WEEKDAY_NAME[last], "WEEKDAY");

      return true;
    case "dotted-dash":
      sentence.add(`${WEEKDAY_ABBR[first]}.`, "WEEKDAY");
      sentence.add(WORDS.glue.dash, "GLUE", "");
      sentence.add(`${WEEKDAY_ABBR[last]}.`, "WEEKDAY");

      return true;
    default:
      return false;
  }
}

function renderEveryOther(sentence: Sentence, intent: ScheduleIntent, templateId: string): boolean {
  const weekday = intent.weekdays![0]!;

  switch (templateId) {
    case "every-other":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.glue.other, "GLUE");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");

      return true;
    case "every-two-weeks":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(String(2), "INTERVAL");
      sentence.add(plural("week", 2), "GLUE");
      sentence.add(WORDS.glue.on, "GLUE");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");

      return true;
    default:
      return false;
  }
}

function naturalClockStyle(time: TimeOfDay): ClockStyle {
  if (time.minute === 15) return "quarter-past";
  if (time.minute === 30) return "half-past";
  if (time.minute === 45) return "quarter-to";
  if (time.minute === 0 && (time.hour === 0 || time.hour === 12)) return "named";
  if (time.minute >= 1 && time.minute <= 30) return "minute-past";
  if (time.minute >= 31) return "minute-to";

  return "digits";
}

function alternativeClockStyle(time: TimeOfDay): ClockStyle {
  if (time.minute === 15) return "quarter-after";
  if (time.minute === 30) return "half-after";
  if (time.minute === 45) return "quarter-before";
  if (time.minute >= 1 && time.minute <= 30) return "minute-after";
  if (time.minute >= 31) return "minute-before";
  if (time.hour === 0 || time.hour === 12) return "named";

  return "oclock";
}

function renderEveryOtherAtTime(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const weekday = intent.weekdays![0]!;
  const time = intent.time!;

  switch (templateId) {
    case "every-other-at":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.glue.other, "GLUE");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "time-every-other":
      addClock(sentence, time, naturalClockStyle(time));
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.glue.other, "GLUE");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");

      return true;
    case "alternate-at":
      sentence.add(WORDS.glue.alternate, "GLUE");
      sentence.add(WEEKDAY_PLURAL[weekday], "WEEKDAY");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "two-weeks-on-at":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(String(2), "INTERVAL");
      sentence.add("weeks", "GLUE");
      sentence.add(WORDS.glue.on, "GLUE");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    default:
      return false;
  }
}

function renderTimeOfDay(sentence: Sentence, intent: ScheduleIntent, templateId: string): boolean {
  const time = intent.time!;
  const [styleText, prefixText] = templateId.split("|");
  const style = styleText as ClockStyle;

  const styles: readonly ClockStyle[] = [
    "digits",
    "hour-meridiem",
    "spoken",
    "24h",
    "compact",
    "military",
    "named",
    "quarter-past",
    "quarter-after",
    "half-past",
    "half-after",
    "quarter-to",
    "quarter-before",
    "minute-past",
    "minute-after",
    "minute-to",
    "minute-before",
    "dotted-meridiem",
    "compact-dotted-meridiem",
    "spaced-dotted-meridiem",
    "oclock",
    "sharp",
    "on-the-dot",
    "qualified",
  ];

  if (!styles.includes(style) || !clockCanRender(time, style)) return false;
  const prefix = prefixText === undefined ? WORDS.glue.at : CLOCK_PREFIXES[Number(prefixText)];

  if (prefix === undefined) return false;
  addClock(sentence, time, style, prefix);

  return true;
}

function renderDayAtTime(sentence: Sentence, intent: ScheduleIntent, templateId: string): boolean {
  const time = intent.time!;

  switch (templateId) {
    case "every-day-at":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.unit.day, "GLUE");
      addClock(sentence, time, "digits");

      return true;
    case "daily-at":
      sentence.add("daily", "INTERVAL");
      addClock(sentence, time, "digits");

      return true;
    case "each-day-at":
      sentence.add(WORDS.recur.each, "RECUR");
      sentence.add(WORDS.unit.day, "GLUE");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "once-a-day-at":
      sentence.add("once a", "RECUR");
      sentence.add(WORDS.unit.day, "GLUE");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "time-every-day":
      addClock(sentence, time, naturalClockStyle(time));
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.unit.day, "GLUE");

      return true;
    case "time-daily":
      addClock(sentence, time, naturalClockStyle(time));
      sentence.add("daily", "INTERVAL");

      return true;
    case "digits-daily":
      addClock(sentence, time, "digits", "");
      sentence.add("daily", "INTERVAL");

      return true;
    case "compact-daily":
      addClock(sentence, time, "compact", "");
      sentence.add("daily", "INTERVAL");

      return true;
    case "24h-daily":
      addClock(sentence, time, "24h", "");
      sentence.add("daily", "INTERVAL");

      return true;
    case "digits-every-day":
      addClock(sentence, time, "digits", "");
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.unit.day, "GLUE");

      return true;
    case "digits-each-day":
      addClock(sentence, time, "digits", "");
      sentence.add(WORDS.recur.each, "RECUR");
      sentence.add(WORDS.unit.day, "GLUE");

      return true;
    case "daily-digits":
      sentence.add("daily", "INTERVAL");
      addClock(sentence, time, "digits", "");

      return true;
    case "daily-basis-at":
      sentence.add("on a daily basis", "INTERVAL");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "every-day-qualified":
      if (!clockCanRender(time, "qualified")) return false;
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.unit.day, "GLUE");
      addClock(sentence, time, "qualified");

      return true;
    case "every-day-alternative-clock":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.unit.day, "GLUE");
      addClock(sentence, time, alternativeClockStyle(time));

      return true;
    case "every-day-dotted":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.unit.day, "GLUE");
      addClock(sentence, time, "dotted-meridiem");

      return true;
    case "each-day-compact-dotted":
      sentence.add(WORDS.recur.each, "RECUR");
      sentence.add(WORDS.unit.day, "GLUE");
      addClock(sentence, time, "compact-dotted-meridiem");

      return true;
    case "once-a-day-spaced-dotted":
      sentence.add("once a", "RECUR");
      sentence.add(WORDS.unit.day, "GLUE");
      addClock(sentence, time, "spaced-dotted-meridiem");

      return true;
    default:
      return false;
  }
}

function renderIntervalAtTime(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const { interval, unit } = intent.frequency;
  const time = intent.time!;

  const addInterval = (prefix: string, words = false): void => {
    sentence.add(prefix, "RECUR");
    sentence.add(words ? numberWord(interval) : String(interval), "INTERVAL");
    sentence.add(plural(unit, interval), "GLUE");
  };

  switch (templateId) {
    case "every-at":
      addInterval(WORDS.recur.every);
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "time-every":
      addClock(sentence, time, naturalClockStyle(time));
      addInterval(WORDS.recur.every);

      return true;
    case "once-every-at":
      addInterval("once every");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "repeating-every-at":
      addInterval("repeating every");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "every-word-at":
      addInterval(WORDS.recur.every, true);
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "time-every-word":
      addClock(sentence, time, naturalClockStyle(time));
      addInterval(WORDS.recur.every, true);

      return true;
    default:
      return false;
  }
}

function renderDaygroupAtTime(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const weekdays = intent.weekdays!;
  const time = intent.time!;
  const dynamic = /^(expression|time-expression|qualified-expression)-(\d+)$/.exec(templateId);

  if (dynamic !== null) {
    const expression = DAYGROUP_EXPRESSIONS[Number(dynamic[2])];

    if (
      expression === undefined ||
      expression.weekdays.length !== weekdays.length ||
      !expression.weekdays.every((day, index) => day === weekdays[index])
    ) {
      return false;
    }

    if (dynamic[1] === "time-expression") addClock(sentence, time, naturalClockStyle(time));

    if (expression.prefix !== "") {
      sentence.add(expression.prefix, expression.prefix === "on" ? "GLUE" : "RECUR");
    }

    sentence.add(expression.surface, "DAYGROUP");
    if (dynamic[1] === "expression") addClock(sentence, time, naturalClockStyle(time));

    if (dynamic[1] === "qualified-expression") {
      if (!clockCanRender(time, "qualified")) return false;
      addClock(sentence, time, "qualified");
    }

    return true;
  }

  const train = DAYGROUPS.find(
    (group) =>
      group.weekdays.length === weekdays.length &&
      group.weekdays.every((day, index) => day === weekdays[index]),
  );

  const held = HELD_OUT_DAYGROUPS.find(
    (group) =>
      group.weekdays.length === weekdays.length &&
      group.weekdays.every((day, index) => day === weekdays[index]),
  );

  switch (templateId) {
    case "every-group-at":
      if (train === undefined) return false;
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(train.surface, "DAYGROUP");
      addClock(sentence, time, "digits");

      return true;
    case "on-group-at":
      if (held === undefined) return false;
      sentence.add(WORDS.glue.on, "GLUE");
      sentence.add(held.surface, "DAYGROUP");
      addClock(sentence, time, "digits");

      return true;
    default:
      return false;
  }
}

function renderWeekdayAtTime(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const weekdays = intent.weekdays!;
  const time = intent.time!;

  switch (templateId) {
    case "every-day-at":
      sentence.add(WORDS.recur.every, "RECUR");
      addWeekdays(sentence, weekdays, "full", WORDS.glue.and);
      addClock(sentence, time, "digits");

      return true;
    case "on-day-at":
      sentence.add(WORDS.glue.on, "GLUE");
      addWeekdays(sentence, weekdays, "full", WORDS.glue.and);
      addClock(sentence, time, "digits");

      return true;
    case "time-every-day":
      addClock(sentence, time, "digits");
      sentence.add(WORDS.recur.every, "RECUR");
      addWeekdays(sentence, weekdays, "full", WORDS.glue.and);

      return true;
    case "time-on-days":
      addClock(sentence, time, "digits");
      sentence.add(WORDS.glue.on, "GLUE");
      addWeekdays(sentence, weekdays, "full", WORDS.glue.and);

      return true;
    case "each-week-on-at":
      sentence.add("each week on", "RECUR");
      addWeekdays(sentence, weekdays, "full", WORDS.glue.and);
      addClock(sentence, time, "digits");

      return true;
    case "days-at-each-week":
      addWeekdays(sentence, weekdays, "full", WORDS.glue.and);
      addClock(sentence, time, "digits");
      sentence.add("each week", "RECUR");

      return true;
    case "weekdays-qualified":
      if (!clockCanRender(time, "qualified")) return false;
      sentence.add(WORDS.glue.on, "GLUE");
      addWeekdays(sentence, weekdays, "full", WORDS.glue.and);
      addClock(sentence, time, "qualified");

      return true;
    case "weekdays-alternative-clock":
      sentence.add(WORDS.glue.on, "GLUE");
      addWeekdays(sentence, weekdays, "full", WORDS.glue.and);
      addClock(sentence, time, alternativeClockStyle(time));

      return true;
    case "plural-at":
      addWeekdays(sentence, weekdays, "plural", WORDS.glue.and);
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "on-plural-at":
      sentence.add(WORDS.glue.on, "GLUE");
      addWeekdays(sentence, weekdays, "plural", WORDS.glue.and);
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "slash-at":
      if (weekdays.length < 2) return false;
      addSlashWeekdays(sentence, weekdays);
      addClock(sentence, time, "digits");

      return true;
    case "plural-exact":
      addWeekdays(sentence, weekdays, "plural", WORDS.glue.and);
      addClock(sentence, time, "digits", "at exactly");

      return true;
    default:
      return false;
  }
}

function renderDayOfMonth(sentence: Sentence, intent: ScheduleIntent, templateId: string): boolean {
  const day = intent.dayOfMonth!;

  switch (templateId) {
    case "on-word":
      sentence.add("on the", "GLUE");
      sentence.add(DAY_ORDINALS[day - 1]!, "DOM");
      sentence.add("of every month", "RECUR");

      return true;
    case "every-month-word":
      sentence.add("every month", "RECUR");
      sentence.add("on the", "GLUE");
      sentence.add(DAY_ORDINALS[day - 1]!, "DOM");

      return true;
    case "on-the":
      sentence.add(WORDS.glue.on, "GLUE");
      sentence.add(WORDS.glue.the, "GLUE");
      addDom(sentence, day);
      sentence.add(WORDS.glue.of, "GLUE");
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.unit.month, "GLUE");

      return true;
    case "monthly-on":
      sentence.add(WORDS.recur.monthly, "RECUR");
      sentence.add(WORDS.glue.on, "GLUE");
      sentence.add(WORDS.glue.the, "GLUE");
      addDom(sentence, day);

      return true;
    case "day-of-month":
      sentence.add("day", "GLUE");
      sentence.add(String(day), "DOM");
      sentence.add("of every month", "RECUR");

      return true;
    case "every-month-on-day":
      sentence.add("every month", "RECUR");
      sentence.add("on day", "GLUE");
      sentence.add(String(day), "DOM");

      return true;
    case "ordinal-each-month":
      addDom(sentence, day);
      sentence.add("of each month", "RECUR");

      return true;
    case "on-day-each-month":
      sentence.add("on day", "GLUE");
      sentence.add(String(day), "DOM");
      sentence.add("each month", "RECUR");

      return true;
    case "once-a-month-on":
      sentence.add("once a month", "RECUR");
      sentence.add("on the", "GLUE");
      addDom(sentence, day);

      return true;
    default:
      return false;
  }
}

function renderDayOfMonthAtTime(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const day = intent.dayOfMonth!;
  const time = intent.time!;

  switch (templateId) {
    case "word-at":
      sentence.add("on the", "GLUE");
      sentence.add(DAY_ORDINALS[day - 1]!, "DOM");
      sentence.add("of every month", "RECUR");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "time-word-monthly":
      addClock(sentence, time, naturalClockStyle(time));
      sentence.add("on the", "GLUE");
      sentence.add(DAY_ORDINALS[day - 1]!, "DOM");
      sentence.add("of each month", "RECUR");

      return true;
    case "the-at":
      sentence.add(WORDS.glue.the, "GLUE");
      addDom(sentence, day);
      addClock(sentence, time, "digits");

      return true;
    case "monthly-on-at":
      sentence.add(WORDS.recur.monthly, "RECUR");
      sentence.add(WORDS.glue.on, "GLUE");
      sentence.add(WORDS.glue.the, "GLUE");
      addDom(sentence, day);
      addClock(sentence, time, "digits");

      return true;
    case "day-of-month-at":
      sentence.add("day", "GLUE");
      sentence.add(String(day), "DOM");
      sentence.add("of every month", "RECUR");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "every-month-day-at":
      sentence.add("every month", "RECUR");
      sentence.add("on day", "GLUE");
      sentence.add(String(day), "DOM");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "on-day-each-month-at":
      sentence.add("on day", "GLUE");
      sentence.add(String(day), "DOM");
      sentence.add("each month", "RECUR");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "once-a-month-on-at":
      sentence.add("once a month", "RECUR");
      sentence.add("on the", "GLUE");
      addDom(sentence, day);
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    default:
      return false;
  }
}

function renderOrdinalWeekday(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const { ordinal, weekday } = intent.ordinalWeekday!;

  switch (templateId) {
    case "the-ordinal-of":
      sentence.add(WORDS.glue.the, "GLUE");
      sentence.add(ordinal, "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      sentence.add(WORDS.glue.of, "GLUE");
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.unit.month, "GLUE");

      return true;
    case "every-ordinal":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(ordinal, "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      sentence.add(WORDS.glue.of, "GLUE");
      sentence.add(WORDS.glue.the, "GLUE");
      sentence.add(WORDS.unit.month, "GLUE");

      return true;
    case "of-each-month":
      sentence.add(WORDS.glue.the, "GLUE");
      sentence.add(ordinal, "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      sentence.add("of each month", "RECUR");

      return true;
    case "monthly-on":
      sentence.add(WORDS.recur.monthly, "RECUR");
      sentence.add("on the", "GLUE");
      sentence.add(ordinal, "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");

      return true;
    case "every-months":
      sentence.add("every month's", "RECUR");
      sentence.add(ordinal, "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");

      return true;
    case "on-every-month":
      sentence.add(WORDS.glue.on, "GLUE");
      sentence.add(WORDS.glue.the, "GLUE");
      sentence.add(ordinal, "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      sentence.add("every month", "RECUR");

      return true;

    case "numeric-of-each": {
      if (ordinal === "last") return false;
      const rank = { first: 1, second: 2, third: 3, fourth: 4 }[ordinal];

      sentence.add(`${rank}${ordinalSuffix(rank)}`, "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      sentence.add("of each month", "RECUR");

      return true;
    }

    default:
      return false;
  }
}

function renderOrdinalWeekdayAtTime(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const { ordinal, weekday } = intent.ordinalWeekday!;
  const time = intent.time!;
  const ordinalSurface = ordinal === "last" && templateId.includes("final") ? "final" : ordinal;

  switch (templateId) {
    case "final-of-each-at":
    case "ordinal-of-each-at":
      if (templateId === "final-of-each-at" && ordinal !== "last") return false;
      sentence.add(WORDS.glue.the, "GLUE");
      sentence.add(ordinalSurface, "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      sentence.add("of each month", "RECUR");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "monthly-on-at":
      sentence.add(WORDS.recur.monthly, "RECUR");
      sentence.add("on the", "GLUE");
      sentence.add(ordinal, "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    case "time-on-ordinal":
      addClock(sentence, time, naturalClockStyle(time));
      sentence.add(WORDS.glue.on, "GLUE");
      sentence.add(WORDS.glue.the, "GLUE");
      sentence.add(ordinal, "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      sentence.add("every month", "RECUR");

      return true;

    case "numeric-of-each-at": {
      if (ordinal === "last") return false;
      const rank = { first: 1, second: 2, third: 3, fourth: 4 }[ordinal];

      sentence.add(`${rank}${ordinalSuffix(rank)}`, "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      sentence.add("of each month", "RECUR");
      addClock(sentence, time, naturalClockStyle(time));

      return true;
    }

    default:
      return false;
  }
}

function renderBounds(sentence: Sentence, intent: ScheduleIntent, templateId: string): boolean {
  const { start, end } = intent.bounds!;

  sentence.add(WORDS.recur.every, "RECUR");
  sentence.add(WORDS.unit.day, "GLUE");
  const inclusiveEnd = /^inclusive-end-(\d+)-(\d+)$/.exec(templateId);

  if (inclusiveEnd !== null) {
    if (end === undefined || start !== undefined) return false;
    const prefix = INCLUSIVE_END_BOUND_PREFIXES[Number(inclusiveEnd[1])];
    const style = DATE_STYLES[Number(inclusiveEnd[2])];

    return (
      prefix !== undefined &&
      style !== undefined &&
      addInclusiveEndDate(sentence, end, style, prefix)
    );
  }

  const naturalStart = /^start-natural-(\d+)-(\d+)$/.exec(templateId);

  if (naturalStart !== null) {
    if (start === undefined || end !== undefined) return false;
    const prefix = START_BOUND_PREFIXES[Number(naturalStart[1])];
    const style = DATE_STYLES[Number(naturalStart[2])];

    if (prefix === undefined || style === undefined) return false;
    sentence.add(prefix, "BOUND");

    return addCalendarDate(sentence, start, style);
  }

  const naturalEnd = /^end-natural-(\d+)-(\d+)$/.exec(templateId);

  if (naturalEnd !== null) {
    if (end === undefined || start !== undefined) return false;
    const prefix = END_BOUND_PREFIXES[Number(naturalEnd[1])];
    const style = DATE_STYLES[Number(naturalEnd[2])];

    if (prefix === undefined || style === undefined) return false;
    sentence.add(prefix, "BOUND");

    return addCalendarDate(sentence, end, style);
  }

  const between = /^between-natural-(\d+)$/.exec(templateId);

  if (between !== null) {
    if (start === undefined || end === undefined) return false;
    const style = DATE_STYLES[Number(between[1])];

    if (style === undefined) return false;
    sentence.add(WORDS.bounds.between, "BOUND");
    if (!addCalendarDate(sentence, start, style)) return false;
    sentence.add(WORDS.glue.and, "GLUE");

    return addCalendarDate(sentence, end, style);
  }

  if (templateId.startsWith("start-")) {
    if (start === undefined || end !== undefined) return false;
    const prefix = START_BOUND_PREFIXES[Number(templateId.slice(6))];

    if (prefix === undefined) return false;
    sentence.add(prefix, "BOUND");
    sentence.add(start, "DOM");

    return true;
  }

  if (templateId.startsWith("end-")) {
    if (end === undefined || start !== undefined) return false;
    const prefix = END_BOUND_PREFIXES[Number(templateId.slice(4))];

    if (prefix === undefined) return false;
    sentence.add(prefix, "BOUND");
    sentence.add(end, "DOM");

    return true;
  }

  switch (templateId) {
    case "from-to":
      if (start !== undefined) {
        sentence.add(WORDS.bounds.from, "BOUND");
        sentence.add(start, "DOM");
      }

      if (end !== undefined) {
        sentence.add(WORDS.bounds.to, "BOUND");
        sentence.add(end, "DOM");
      }

      return true;
    case "until":
      if (start !== undefined && end !== undefined) {
        sentence.add(WORDS.bounds.starting, "BOUND");
        sentence.add(start, "DOM");
      }

      if (end !== undefined) {
        sentence.add(WORDS.bounds.until, "BOUND");
        sentence.add(end, "DOM");
      } else if (start !== undefined) {
        sentence.add(WORDS.bounds.starting, "BOUND");
        sentence.add(start, "DOM");
      }

      return true;
    default:
      return false;
  }
}

function renderBoundsAtTime(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const inclusive = /^(base-inclusive|inclusive-base)-(\d+)-(\d+)$/.exec(templateId);

  if (inclusive !== null) {
    const { start, end } = intent.bounds!;
    const prefix = INCLUSIVE_END_BOUND_PREFIXES[Number(inclusive[2])];
    const style = DATE_STYLES[Number(inclusive[3])];

    if (start !== undefined || end === undefined || prefix === undefined || style === undefined) {
      return false;
    }

    const addBase = (): boolean => {
      if (!addScheduleSelector(sentence, intent)) return false;
      addClock(sentence, intent.time!, naturalClockStyle(intent.time!));

      return true;
    };

    if (inclusive[1] === "base-inclusive") {
      return addBase() && addInclusiveEndDate(sentence, end, style, prefix);
    }

    if (!addInclusiveEndDate(sentence, end, style, prefix)) return false;
    sentence.add(WORDS.glue.comma, "GLUE", "");
    sentence.clause();

    return addBase();
  }

  const match = /^(base-start|start-base|base-end|base-between|between-base)-(\d+)$/.exec(
    templateId,
  );

  if (match === null) return false;
  const style = DATE_STYLES[Number(match[2])];

  if (style === undefined) return false;
  const { start, end } = intent.bounds!;

  const addBase = (): boolean => {
    if (!addScheduleSelector(sentence, intent)) return false;
    addClock(sentence, intent.time!, naturalClockStyle(intent.time!));

    return true;
  };

  const addStart = (): boolean => {
    sentence.add("starting on", "BOUND");

    return start !== undefined && addCalendarDate(sentence, start, style);
  };

  const addEnd = (): boolean => {
    sentence.add(WORDS.bounds.until, "BOUND");

    return end !== undefined && addCalendarDate(sentence, end, style);
  };

  const addBetween = (): boolean => {
    if (start === undefined || end === undefined) return false;
    sentence.add(WORDS.bounds.between, "BOUND");
    if (!addCalendarDate(sentence, start, style)) return false;
    sentence.add(WORDS.glue.and, "GLUE");

    return addCalendarDate(sentence, end, style);
  };

  switch (match[1]) {
    case "base-start":
      return start !== undefined && end === undefined && addBase() && addStart();
    case "start-base":
      if (start === undefined || end !== undefined || !addStart()) return false;
      sentence.add(WORDS.glue.comma, "GLUE", "");
      sentence.clause();

      return addBase();
    case "base-end":
      return end !== undefined && start === undefined && addBase() && addEnd();
    case "base-between":
      return start !== undefined && end !== undefined && addBase() && addBetween();
    case "between-base":
      if (start === undefined || end === undefined || !addBetween()) return false;
      sentence.add(WORDS.glue.comma, "GLUE", "");
      sentence.clause();

      return addBase();
    default:
      return false;
  }
}

function renderExceptionGroup(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const group = intent.exceptions!.weekdays!;

  const surface =
    group.length === WEEKDAY_GROUPS.weekend!.length &&
    group.every((day, index) => day === WEEKDAY_GROUPS.weekend![index])
      ? templateId === "except-weekends"
        ? "weekends"
        : "weekend"
      : undefined;

  if (surface === undefined) return false;
  sentence.add(WORDS.recur.every, "RECUR");
  sentence.add(WORDS.unit.day, "GLUE");

  switch (templateId) {
    case "except-weekend":
      sentence.add(WORDS.exception.except, "EXCEPT");
      sentence.add(surface, "DAYGROUP");

      return true;
    case "except-weekends":
      sentence.add(WORDS.exception.excluding, "EXCEPT");
      sentence.add(surface, "DAYGROUP");

      return true;
    case "except-the-weekend":
      sentence.add(WORDS.exception.except, "EXCEPT");
      sentence.add(WORDS.glue.the, "GLUE");
      sentence.add("weekend", "DAYGROUP");

      return true;
    case "but-not-weekends":
      sentence.add("but not", "EXCEPT");
      sentence.add("weekends", "DAYGROUP");

      return true;
    case "excluding-weekend-days":
      sentence.add(WORDS.exception.excluding, "EXCEPT");
      sentence.add("weekend days", "DAYGROUP");

      return true;
    default:
      return false;
  }
}

function renderExceptionDays(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const weekdays = intent.exceptions!.weekdays!;

  sentence.add(WORDS.recur.every, "RECUR");
  sentence.add(WORDS.unit.day, "GLUE");
  const surface = /^surface-(\d+)-(plural|abbrev|dotted-abbrev)$/.exec(templateId);

  if (surface !== null) {
    const prefix = EXCEPTION_PREFIXES[Number(surface[1])];

    if (prefix === undefined) return false;
    sentence.add(prefix, "EXCEPT");

    addWeekdays(
      sentence,
      weekdays,
      surface[2] as "plural" | "abbrev" | "dotted-abbrev",
      WORDS.glue.and,
    );

    return true;
  }

  if (templateId.startsWith("prefix-")) {
    const prefix = EXCEPTION_PREFIXES[Number(templateId.slice(7))];

    if (prefix === undefined) return false;
    sentence.add(prefix, "EXCEPT");
    addWeekdays(sentence, weekdays, "full", WORDS.glue.and);

    return true;
  }

  switch (templateId) {
    case "except":
      sentence.add(WORDS.exception.except, "EXCEPT");
      addWeekdays(sentence, weekdays, "full", WORDS.glue.and);

      return true;
    case "excluding":
      sentence.add(WORDS.exception.excluding, "EXCEPT");
      addWeekdays(sentence, weekdays, "full", WORDS.glue.and);

      return true;
    default:
      return false;
  }
}

function renderSelectorException(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const excluded = intent.exceptions!.weekdays!;
  const surface = /^surface-(\d+)-(plural|abbrev|dotted-abbrev)$/.exec(templateId);
  const prefix = EXCEPTION_PREFIXES[Number(surface === null ? templateId.slice(7) : surface[1])];

  if (prefix === undefined) return false;
  sentence.add("weekdays", "DAYGROUP");
  sentence.add(prefix, "EXCEPT");

  addWeekdays(
    sentence,
    excluded,
    surface === null ? "full" : (surface[2] as "plural" | "abbrev" | "dotted-abbrev"),
    WORDS.glue.and,
  );

  return true;
}

function renderExceptionDate(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const match = /^prefix-(\d+)-(\d+)$/.exec(templateId);

  if (match === null) return false;
  const prefix = EXCEPTION_PREFIXES[Number(match[1])];
  const style = DATE_STYLES[Number(match[2])];

  if (prefix === undefined || style === undefined) return false;
  sentence.add(WORDS.recur.every, "RECUR");
  sentence.add(WORDS.unit.day, "GLUE");
  sentence.add(prefix, "EXCEPT");
  const dates = intent.exceptions!.dates!;

  for (let index = 0; index < dates.length; index++) {
    if (index > 0) sentence.add(WORDS.glue.and, "GLUE");
    if (!addCalendarDate(sentence, dates[index]!, style)) return false;
  }

  return true;
}

function renderExceptionDateAtTime(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const match = /^(after|before)-(\d+)-(\d+)$/.exec(templateId);

  if (match === null) return false;
  const prefix = EXCEPTION_PREFIXES[Number(match[2])];
  const style = DATE_STYLES[Number(match[3])];

  if (prefix === undefined || style === undefined) return false;

  const addBase = (): boolean => {
    if (!addScheduleSelector(sentence, intent)) return false;
    addClock(sentence, intent.time!, naturalClockStyle(intent.time!));

    return true;
  };

  const addExceptions = (): boolean => {
    sentence.add(prefix, "EXCEPT");
    const dates = intent.exceptions!.dates!;

    for (let index = 0; index < dates.length; index++) {
      if (index > 0) sentence.add(WORDS.glue.and, "GLUE");
      if (!addCalendarDate(sentence, dates[index]!, style)) return false;
    }

    return true;
  };

  if (match[1] === "after") return addBase() && addExceptions();
  if (!addExceptions()) return false;
  sentence.add(WORDS.glue.comma, "GLUE", "");
  sentence.clause();

  return addBase();
}

function renderExceptionAtTime(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const time = intent.time!;
  const excluded = intent.exceptions!.weekdays!;
  const match = /^(before|after|tail)-(\d+)$/.exec(templateId);

  if (match === null) return false;
  const prefix = EXCEPTION_PREFIXES[Number(match[2])];

  if (prefix === undefined) return false;
  if (match[1] === "before") addClock(sentence, time, naturalClockStyle(time));
  sentence.add(WORDS.recur.every, "RECUR");
  sentence.add(WORDS.unit.day, "GLUE");
  if (match[1] === "after") addClock(sentence, time, naturalClockStyle(time));
  sentence.add(prefix, "EXCEPT");
  addWeekdays(sentence, excluded, "full", WORDS.glue.and);
  if (match[1] === "tail") addClock(sentence, time, naturalClockStyle(time));

  return true;
}

function renderSelectorExceptionAtTime(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const time = intent.time!;
  const excluded = intent.exceptions!.weekdays!;
  const match = /^(before|after|tail)-(\d+)$/.exec(templateId);

  if (match === null) return false;
  const prefix = EXCEPTION_PREFIXES[Number(match[2])];

  if (prefix === undefined) return false;
  if (match[1] === "before") addClock(sentence, time, naturalClockStyle(time));
  sentence.add("weekdays", "DAYGROUP");
  if (match[1] === "after") addClock(sentence, time, naturalClockStyle(time));
  sentence.add(prefix, "EXCEPT");
  addWeekdays(sentence, excluded, "full", WORDS.glue.and);
  if (match[1] === "tail") addClock(sentence, time, naturalClockStyle(time));

  return true;
}

function renderFrequencyIdiom(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const phrase = Object.keys(FREQUENCY_IDIOMS)[Number(templateId.slice(6))];

  if (phrase === undefined) return false;
  const value = FREQUENCY_IDIOMS[phrase];

  if (
    value === undefined ||
    value.unit !== intent.frequency.unit ||
    value.interval !== intent.frequency.interval
  ) {
    return false;
  }

  sentence.add(phrase, "INTERVAL");

  return true;
}

function renderDayRule(sentence: Sentence, intent: ScheduleIntent, templateId: string): boolean {
  const phrase = Object.keys(DAY_RULES)[Number(templateId.slice(5))];

  if (phrase === undefined || DAY_RULES[phrase as keyof typeof DAY_RULES] !== intent.day?.type) {
    return false;
  }

  sentence.add("on the", "GLUE");
  sentence.add(phrase, "DOM");
  sentence.add("of every month", "RECUR");

  return true;
}

function renderDayRuleAtTime(
  sentence: Sentence,
  intent: ScheduleIntent,
  templateId: string,
): boolean {
  const index = Number(templateId.slice(templateId.lastIndexOf("-") + 1));
  const phrase = Object.keys(DAY_RULES)[index];

  if (phrase === undefined || DAY_RULES[phrase as keyof typeof DAY_RULES] !== intent.day?.type) {
    return false;
  }

  if (templateId.startsWith("time-")) {
    addClock(sentence, intent.time!, naturalClockStyle(intent.time!));
  }

  sentence.add("on the", "GLUE");
  sentence.add(phrase, "DOM");
  sentence.add("of every month", "RECUR");

  if (templateId.startsWith("rule-")) {
    addClock(sentence, intent.time!, naturalClockStyle(intent.time!));
  }

  return true;
}

interface KindDef {
  id: KindId;
  family: Family;
  templates: readonly KindTemplate[];
  intents: Array<ScheduleIntent | null>;
  render: (sentence: Sentence, intent: ScheduleIntent | null, templateId: string) => boolean;
}

function nIntervalIntents(): ScheduleIntent[] {
  const intents: ScheduleIntent[] = [];

  for (const unit of UNITS) {
    for (const interval of N_INTERVALS[unit]) {
      intents.push({ frequency: { unit, interval } });
    }
  }

  return intents;
}

function frequencyIdiomIntents(): ScheduleIntent[] {
  const seen = new Set<string>();
  const intents: ScheduleIntent[] = [];

  for (const frequency of Object.values(FREQUENCY_IDIOMS)) {
    const key = `${frequency.unit}:${frequency.interval}`;

    if (seen.has(key)) continue;
    seen.add(key);
    intents.push({ frequency: { ...frequency } });
  }

  return intents;
}

function timedSelectorIntents(): ScheduleIntent[] {
  return COMPOUND_CLOCKS.flatMap((time) => [
    daily(time),
    ...WEEKDAY_SEQUENCE.map((weekday) => weekly([weekday], time)),
    weekly(WEEKDAY_GROUPS.weekdays!, time),
    weekly(WEEKDAY_GROUPS.weekends!, time),
  ]);
}

function boundedAtTimeIntents(): ScheduleIntent[] {
  const bounds = [
    { start: "2026-01-01" },
    { end: "2027-03-04" },
    { start: "2026-01-01", end: "2027-03-04" },
  ] as const;

  return timedSelectorIntents().flatMap((intent) =>
    bounds.map((value) => ({ ...intent, bounds: { ...value } })),
  );
}

function exceptionDateAtTimeIntents(): ScheduleIntent[] {
  const dateSets: readonly (readonly string[])[] = [
    ["2026-01-01"],
    ["2026-07-04"],
    ["2026-12-25"],
    ["2026-01-01", "2026-12-25"],
  ];

  return COMPOUND_CLOCKS.flatMap((time) =>
    dateSets.map((dates) => ({
      ...daily(time),
      exceptions: { dates: [...dates] },
    })),
  );
}

/** The closed catalog. Adding a kind here is how the corpus grows. */
export const KINDS: readonly KindDef[] = [
  {
    id: "unit-interval",
    family: "interval",
    templates: [
      { id: "every-unit", split: "train" },
      { id: "each-unit", split: "train" },
      ...[2, 3, 4].map((index) => ({
        id: `prefix-${index}`,
        split: "train" as const,
      })),
      { id: "adverb", split: "heldout" },
    ],
    intents: UNITS.map((unit) => ({ frequency: { unit, interval: 1 } })),
    render: (sentence, intent, templateId) => renderUnitInterval(sentence, intent!, templateId),
  },
  {
    id: "n-interval",
    family: "interval",
    templates: [
      { id: "every-n", split: "train" },
      { id: "each-n", split: "train" },
      { id: "every-n-word", split: "train" },
      { id: "once-n-word", split: "train" },
      { id: "intervals-n-word", split: "train" },
      ...[2, 4, 6].map((index) => ({
        id: `prefix-${index}`,
        split: "train" as const,
      })),
      { id: "recurring-every-n", split: "heldout" },
      { id: "once-per-n", split: "train" },
      { id: "every-n-on-repeat", split: "train" },
      { id: "every-abbrev", split: "train" },
      { id: "every-word-abbrev", split: "heldout" },
      { id: "at-hyphenated", split: "train" },
      { id: "in-hyphenated", split: "heldout" },
      { id: "every-hyphenated", split: "train" },
      { id: "on-a-hyphenated-cycle", split: "train" },
      { id: "interval-is", split: "train" },
      { id: "cadence-is", split: "train" },
      { id: "cycle-of", split: "train" },
      { id: "between-runs", split: "train" },
      { id: "recurrence-interval-is", split: "train" },
      { id: "repeat-interval-of", split: "train" },
      { id: "runs-on-cycle", split: "train" },
      { id: "cadence-of", split: "train" },
      { id: "spaced-apart", split: "train" },
    ],
    intents: nIntervalIntents(),
    render: (sentence, intent, templateId) => renderNInterval(sentence, intent!, templateId),
  },
  {
    id: "daygroup",
    family: "weekday-list",
    templates: DAYGROUP_EXPRESSIONS.map((_, index) => ({
      id: `expression-${index}`,
      split: "train" as const,
    })),
    intents: DAYGROUPS.map((group) => weekly(group.weekdays)),
    render: (sentence, intent, templateId) => renderDaygroup(sentence, intent!, templateId),
  },
  {
    id: "single-weekday",
    family: "terse",
    templates: [
      { id: "every-day", split: "train" },
      { id: "on-day", split: "train" },
      { id: "each-day", split: "train" },
      { id: "plural", split: "train" },
      { id: "on-plural", split: "heldout" },
      { id: "description-after", split: "train" },
      { id: "description-following", split: "train" },
      { id: "description-immediately-after", split: "train" },
      { id: "description-before", split: "train" },
      { id: "description-preceding", split: "train" },
      { id: "description-immediately-before", split: "train" },
      { id: "description-iso", split: "train" },
      { id: "description-monday-first", split: "train" },
      ...WEEKDAY_ALIAS_ENTRIES.flatMap((_, index) =>
        (["every", "on", "bare"] as const).map((prefix) => ({
          id: `${prefix}-alias-${index}`,
          split: "train" as const,
        })),
      ),
      ...WEEKDAY_DESCRIPTION_ENTRIES.flatMap((_, index) =>
        (["every", "on", "bare"] as const).map((prefix) => ({
          id: `${prefix}-description-surface-${index}`,
          split: "train" as const,
        })),
      ),
      { id: "bare", split: "heldout" },
    ],
    intents: WEEKDAY_SEQUENCE.map((weekday) => weekly([weekday])),
    render: (sentence, intent, templateId) => renderSingleWeekday(sentence, intent!, templateId),
  },
  {
    id: "weekday-list",
    family: "weekday-list",
    templates: [
      "prefix-0-full",
      "prefix-0-abbrev",
      "prefix-0-dotted-abbrev",
      "prefix-0-comma-and",
      "prefix-0-ampersand",
      "prefix-1-full",
      "prefix-1-plural",
      "prefix-1-abbrev",
      "prefix-2-full",
      "prefix-3-full",
      "prefix-3-abbrev",
      "prefix-4-full",
      "prefix-4-abbrev",
      "prefix-5-full",
      "prefix-5-abbrev",
      "bare-plural",
      "slash-abbrev",
    ].map((id) => ({ id, split: "train" as const })),
    intents: WEEKDAY_INTENTS.map((weekdays) => weekly(weekdays)),
    render: (sentence, intent, templateId) => renderWeekdayList(sentence, intent!, templateId),
  },
  {
    id: "weekday-range",
    family: "weekday-range",
    templates: [
      ...WEEKDAY_RANGE_CONNECTORS.map((_, index) => ({
        id: `connector-${index}`,
        split: "train" as const,
      })),
      { id: "from-to", split: "train" },
      { id: "on-through", split: "train" },
      { id: "weekly-from-to", split: "train" },
      { id: "each-week-up-to-inclusive", split: "train" },
      { id: "dash", split: "heldout" },
      { id: "full-dash", split: "train" },
      { id: "dotted-dash", split: "train" },
    ],
    intents: WEEKDAY_RUNS.map((weekdays) => weekly(weekdays)),
    render: (sentence, intent, templateId) => renderWeekdayRange(sentence, intent!, templateId),
  },
  {
    id: "every-other",
    family: "weekday-list",
    templates: [
      { id: "every-other", split: "train" },
      { id: "every-two-weeks", split: "heldout" },
    ],
    intents: WEEKDAY_SEQUENCE.map((weekday) => ({
      frequency: { unit: "week" as const, interval: 2 },
      weekdays: [weekday],
    })),
    render: (sentence, intent, templateId) => renderEveryOther(sentence, intent!, templateId),
  },
  {
    id: "every-other-at-time",
    family: "weekday-list",
    templates: [
      { id: "every-other-at", split: "train" },
      { id: "time-every-other", split: "train" },
      { id: "alternate-at", split: "train" },
      { id: "two-weeks-on-at", split: "train" },
    ],
    intents: WEEKDAY_SEQUENCE.flatMap((weekday) =>
      COMPOUND_CLOCKS.map((time) => ({
        frequency: { unit: "week" as const, interval: 2 },
        weekdays: [weekday],
        time,
      })),
    ),
    render: (sentence, intent, templateId) => renderEveryOtherAtTime(sentence, intent!, templateId),
  },
  {
    id: "time-of-day",
    family: "time-of-day",
    templates: [
      "digits",
      "hour-meridiem",
      "spoken",
      "24h",
      "compact",
      "military",
      "named",
      "quarter-past",
      "quarter-after",
      "half-past",
      "half-after",
      "quarter-to",
      "quarter-before",
      "minute-past",
      "minute-after",
      "minute-to",
      "minute-before",
      "dotted-meridiem",
      "compact-dotted-meridiem",
      "spaced-dotted-meridiem",
      "oclock",
      "sharp",
      "on-the-dot",
      ...CLOCK_PREFIXES.slice(1).map((_, index) => `digits|${index + 1}`),
    ].map((id) => ({ id, split: "train" as const })),
    intents: CLOCKS.map((time) => daily(time)),
    render: (sentence, intent, templateId) => renderTimeOfDay(sentence, intent!, templateId),
  },
  {
    id: "day-at-time",
    family: "time-of-day",
    templates: [
      { id: "every-day-at", split: "train" },
      { id: "daily-at", split: "heldout" },
      { id: "each-day-at", split: "train" },
      { id: "once-a-day-at", split: "train" },
      { id: "time-every-day", split: "train" },
      { id: "time-daily", split: "train" },
      { id: "digits-daily", split: "train" },
      { id: "compact-daily", split: "train" },
      { id: "24h-daily", split: "train" },
      { id: "digits-every-day", split: "train" },
      { id: "digits-each-day", split: "train" },
      { id: "daily-digits", split: "train" },
      { id: "daily-basis-at", split: "train" },
      { id: "every-day-qualified", split: "train" },
      { id: "every-day-alternative-clock", split: "train" },
      { id: "every-day-dotted", split: "train" },
      { id: "each-day-compact-dotted", split: "train" },
      { id: "once-a-day-spaced-dotted", split: "train" },
    ],
    intents: COMPOUND_CLOCKS.map((time) => daily(time)),
    render: (sentence, intent, templateId) => renderDayAtTime(sentence, intent!, templateId),
  },
  {
    id: "interval-at-time",
    family: "interval",
    templates: [
      { id: "every-at", split: "train" },
      { id: "time-every", split: "train" },
      { id: "once-every-at", split: "train" },
      { id: "repeating-every-at", split: "train" },
      { id: "every-word-at", split: "train" },
      { id: "time-every-word", split: "train" },
    ],
    intents: N_INTERVALS.day.flatMap((interval) =>
      COMPOUND_CLOCKS.map((time) => ({
        frequency: { unit: "day" as const, interval },
        time,
      })),
    ),
    render: (sentence, intent, templateId) => renderIntervalAtTime(sentence, intent!, templateId),
  },
  {
    id: "daygroup-at-time",
    family: "weekday-list",
    templates: DAYGROUP_EXPRESSIONS.flatMap((_, index) => [
      { id: `expression-${index}`, split: "train" as const },
      { id: `time-expression-${index}`, split: "train" as const },
      { id: `qualified-expression-${index}`, split: "train" as const },
    ]),
    intents: DAYGROUPS.flatMap((group) =>
      COMPOUND_CLOCKS.map((time) => weekly(group.weekdays, time)),
    ),
    render: (sentence, intent, templateId) => renderDaygroupAtTime(sentence, intent!, templateId),
  },
  {
    id: "weekday-at-time",
    family: "weekday-list",
    templates: [
      { id: "every-day-at", split: "train" },
      { id: "on-day-at", split: "heldout" },
      { id: "time-every-day", split: "train" },
      { id: "time-on-days", split: "train" },
      { id: "each-week-on-at", split: "train" },
      { id: "days-at-each-week", split: "train" },
      { id: "weekdays-qualified", split: "train" },
      { id: "weekdays-alternative-clock", split: "heldout" },
      { id: "plural-at", split: "train" },
      { id: "on-plural-at", split: "train" },
      { id: "slash-at", split: "heldout" },
      { id: "plural-exact", split: "train" },
    ],
    intents: WEEKDAY_SETS.flatMap((weekdays) =>
      COMPOUND_CLOCKS.map((time) => weekly(weekdays, time)),
    ),
    render: (sentence, intent, templateId) => renderWeekdayAtTime(sentence, intent!, templateId),
  },
  {
    id: "day-of-month",
    family: "day-of-month",
    templates: [
      { id: "on-the", split: "train" },
      { id: "on-word", split: "train" },
      { id: "every-month-word", split: "train" },
      { id: "monthly-on", split: "heldout" },
      { id: "day-of-month", split: "train" },
      { id: "every-month-on-day", split: "train" },
      { id: "ordinal-each-month", split: "heldout" },
      { id: "on-day-each-month", split: "train" },
      { id: "once-a-month-on", split: "train" },
    ],
    intents: DOM_DAYS.map((dayOfMonth) => ({
      frequency: { unit: "month" as const, interval: 1 },
      dayOfMonth,
    })),
    render: (sentence, intent, templateId) => renderDayOfMonth(sentence, intent!, templateId),
  },
  {
    id: "day-of-month-at-time",
    family: "day-of-month",
    templates: [
      { id: "the-at", split: "train" },
      { id: "word-at", split: "train" },
      { id: "time-word-monthly", split: "train" },
      { id: "monthly-on-at", split: "heldout" },
      { id: "day-of-month-at", split: "train" },
      { id: "every-month-day-at", split: "train" },
      { id: "on-day-each-month-at", split: "train" },
      { id: "once-a-month-on-at", split: "train" },
    ],
    intents: DOM_DAYS.flatMap((dayOfMonth) =>
      COMPOUND_CLOCKS.map((time) => ({
        frequency: { unit: "month" as const, interval: 1 },
        dayOfMonth,
        time,
      })),
    ),
    render: (sentence, intent, templateId) => renderDayOfMonthAtTime(sentence, intent!, templateId),
  },
  {
    id: "ordinal-weekday",
    family: "ordinal-weekday",
    templates: [
      { id: "the-ordinal-of", split: "train" },
      { id: "of-each-month", split: "train" },
      { id: "monthly-on", split: "train" },
      { id: "every-months", split: "train" },
      { id: "on-every-month", split: "train" },
      { id: "every-ordinal", split: "heldout" },
      { id: "numeric-of-each", split: "train" },
    ],
    intents: (["first", "second", "third", "fourth", "last"] as const).flatMap((ordinal) =>
      WEEKDAY_SEQUENCE.map((weekday) => ({
        frequency: { unit: "month" as const, interval: 1 },
        ordinalWeekday: { ordinal, weekday },
      })),
    ),
    render: (sentence, intent, templateId) => renderOrdinalWeekday(sentence, intent!, templateId),
  },
  {
    id: "ordinal-weekday-at-time",
    family: "ordinal-weekday",
    templates: [
      { id: "ordinal-of-each-at", split: "train" },
      { id: "final-of-each-at", split: "train" },
      { id: "monthly-on-at", split: "train" },
      { id: "time-on-ordinal", split: "train" },
      { id: "numeric-of-each-at", split: "heldout" },
    ],
    intents: (["first", "second", "third", "fourth", "last"] as const).flatMap((ordinal) =>
      WEEKDAY_SEQUENCE.flatMap((weekday) =>
        COMPOUND_CLOCKS.map((time) => ({
          frequency: { unit: "month" as const, interval: 1 },
          ordinalWeekday: { ordinal, weekday },
          time,
        })),
      ),
    ),
    render: (sentence, intent, templateId) =>
      renderOrdinalWeekdayAtTime(sentence, intent!, templateId),
  },
  {
    id: "bounds",
    family: "bounds",
    templates: [
      { id: "from-to", split: "train" },
      { id: "until", split: "heldout" },
      ...START_BOUND_PREFIXES.map((_, index) => ({
        id: `start-${index}`,
        split: "train" as const,
      })),
      ...END_BOUND_PREFIXES.map((_, index) => ({
        id: `end-${index}`,
        split: "train" as const,
      })),
      ...START_BOUND_PREFIXES.flatMap((_, prefixIndex) =>
        DATE_STYLES.slice(1).map((_, styleIndex) => ({
          id: `start-natural-${prefixIndex}-${styleIndex + 1}`,
          split: "train" as const,
        })),
      ),
      ...END_BOUND_PREFIXES.flatMap((_, prefixIndex) =>
        DATE_STYLES.slice(1).map((_, styleIndex) => ({
          id: `end-natural-${prefixIndex}-${styleIndex + 1}`,
          split: "train" as const,
        })),
      ),
      ...DATE_STYLES.slice(1).map((_, styleIndex) => ({
        id: `between-natural-${styleIndex + 1}`,
        split: "heldout" as const,
      })),
      ...INCLUSIVE_END_BOUND_PREFIXES.flatMap((_, prefixIndex) =>
        DATE_STYLES.map((_, styleIndex) => ({
          id: `inclusive-end-${prefixIndex}-${styleIndex}`,
          split: styleIndex === 0 ? ("heldout" as const) : ("train" as const),
        })),
      ),
    ],
    intents: [
      ...CALENDAR_DATE_LATTICE.map((start) => ({
        frequency: { unit: "day" as const, interval: 1 },
        bounds: { start },
      })),
      ...CALENDAR_DATE_LATTICE.map((end) => ({
        frequency: { unit: "day" as const, interval: 1 },
        bounds: { end },
      })),
      ...MONTH_RANGES_2026.map((bounds) => ({
        frequency: { unit: "day" as const, interval: 1 },
        bounds,
      })),
    ],
    render: (sentence, intent, templateId) => renderBounds(sentence, intent!, templateId),
  },
  {
    id: "bounds-at-time",
    family: "bounds",
    templates: DATE_STYLES.flatMap((_, styleIndex) => [
      { id: `base-start-${styleIndex}`, split: "train" as const },
      { id: `start-base-${styleIndex}`, split: "heldout" as const },
      { id: `base-end-${styleIndex}`, split: "train" as const },
      { id: `base-between-${styleIndex}`, split: "train" as const },
      { id: `between-base-${styleIndex}`, split: "heldout" as const },
      ...INCLUSIVE_END_BOUND_PREFIXES.flatMap((_, prefixIndex) => [
        {
          id: `base-inclusive-${prefixIndex}-${styleIndex}`,
          split: "train" as const,
        },
        {
          id: `inclusive-base-${prefixIndex}-${styleIndex}`,
          split: "heldout" as const,
        },
      ]),
    ]),
    intents: boundedAtTimeIntents(),
    render: (sentence, intent, templateId) => renderBoundsAtTime(sentence, intent!, templateId),
  },
  {
    id: "exception-group",
    family: "exception",
    templates: [
      { id: "except-weekend", split: "train" },
      { id: "except-weekends", split: "heldout" },
      { id: "except-the-weekend", split: "train" },
      { id: "but-not-weekends", split: "train" },
      { id: "excluding-weekend-days", split: "heldout" },
    ],
    intents: [
      {
        frequency: { unit: "day", interval: 1 },
        exceptions: { weekdays: [...WEEKDAY_GROUPS.weekend!] },
      },
    ],
    render: (sentence, intent, templateId) => renderExceptionGroup(sentence, intent!, templateId),
  },
  {
    id: "exception-days",
    family: "exception",
    templates: [
      ...EXCEPTION_PREFIXES.map((_, index) => ({
        id: `prefix-${index}`,
        split: "train" as const,
      })),
      ...EXCEPTION_PREFIXES.flatMap((_, index) =>
        (["plural", "abbrev", "dotted-abbrev"] as const).map((mode) => ({
          id: `surface-${index}-${mode}`,
          split: mode === "dotted-abbrev" ? ("heldout" as const) : ("train" as const),
        })),
      ),
    ],
    intents: [
      {
        frequency: { unit: "day", interval: 1 },
        exceptions: { weekdays: ["friday"] },
      },
      {
        frequency: { unit: "day", interval: 1 },
        exceptions: { weekdays: ["saturday", "sunday"] },
      },
    ],
    render: (sentence, intent, templateId) => renderExceptionDays(sentence, intent!, templateId),
  },
  {
    id: "selector-exception",
    family: "exception",
    templates: [
      ...EXCEPTION_PREFIXES.map((_, index) => ({
        id: `prefix-${index}`,
        split: "train" as const,
      })),
      ...EXCEPTION_PREFIXES.flatMap((_, index) =>
        (["plural", "abbrev", "dotted-abbrev"] as const).map((mode) => ({
          id: `surface-${index}-${mode}`,
          split: mode === "dotted-abbrev" ? ("heldout" as const) : ("train" as const),
        })),
      ),
    ],
    intents: integers(1, 30).map((mask) => ({
      frequency: { unit: "week" as const, interval: 1 },
      weekdays: [...WEEKDAY_GROUPS.weekdays!],
      exceptions: {
        weekdays: WEEKDAY_GROUPS.weekdays!.filter((_, index) => (mask & (1 << index)) !== 0),
      },
    })),
    render: (sentence, intent, templateId) =>
      renderSelectorException(sentence, intent!, templateId),
  },
  {
    id: "exception-date",
    family: "exception",
    templates: EXCEPTION_PREFIXES.flatMap((_, prefixIndex) =>
      DATE_STYLES.map((_, styleIndex) => ({
        id: `prefix-${prefixIndex}-${styleIndex}`,
        split: styleIndex === DATE_STYLES.length - 1 ? ("heldout" as const) : ("train" as const),
      })),
    ),
    intents: [
      ...CALENDAR_DATE_LATTICE.map((date) => ({
        frequency: { unit: "day" as const, interval: 1 },
        exceptions: { dates: [date] },
      })),
      {
        frequency: { unit: "day" as const, interval: 1 },
        exceptions: { dates: ["2026-01-01", "2026-12-25"] },
      },
    ],
    render: (sentence, intent, templateId) => renderExceptionDate(sentence, intent!, templateId),
  },
  {
    id: "exception-date-at-time",
    family: "exception",
    templates: EXCEPTION_PREFIXES.flatMap((_, prefixIndex) =>
      DATE_STYLES.flatMap((_, styleIndex) => [
        {
          id: `after-${prefixIndex}-${styleIndex}`,
          split: "train" as const,
        },
        {
          id: `before-${prefixIndex}-${styleIndex}`,
          split: "heldout" as const,
        },
      ]),
    ),
    intents: exceptionDateAtTimeIntents(),
    render: (sentence, intent, templateId) =>
      renderExceptionDateAtTime(sentence, intent!, templateId),
  },
  {
    id: "exception-at-time",
    family: "exception",
    templates: EXCEPTION_PREFIXES.flatMap((_, index) => [
      { id: `before-${index}`, split: "train" as const },
      { id: `after-${index}`, split: "train" as const },
      { id: `tail-${index}`, split: "heldout" as const },
    ]),
    intents: WEEKDAY_SEQUENCE.flatMap((weekday) =>
      COMPOUND_CLOCKS.map((time) => ({
        frequency: { unit: "day" as const, interval: 1 },
        time,
        exceptions: { weekdays: [weekday] },
      })),
    ),
    render: (sentence, intent, templateId) => renderExceptionAtTime(sentence, intent!, templateId),
  },
  {
    id: "selector-exception-at-time",
    family: "exception",
    templates: EXCEPTION_PREFIXES.flatMap((_, index) => [
      { id: `before-${index}`, split: "train" as const },
      { id: `after-${index}`, split: "train" as const },
      { id: `tail-${index}`, split: "heldout" as const },
    ]),
    intents: WEEKDAY_GROUPS.weekdays!.flatMap((weekday) =>
      COMPOUND_CLOCKS.map((time) => ({
        frequency: { unit: "week" as const, interval: 1 },
        weekdays: [...WEEKDAY_GROUPS.weekdays!],
        time,
        exceptions: { weekdays: [weekday] },
      })),
    ),
    render: (sentence, intent, templateId) =>
      renderSelectorExceptionAtTime(sentence, intent!, templateId),
  },
  {
    id: "frequency-idiom",
    family: "interval",
    templates: Object.keys(FREQUENCY_IDIOMS).map((_, index) => ({
      id: `idiom-${index}`,
      split: "train" as const,
    })),
    intents: frequencyIdiomIntents(),
    render: (sentence, intent, templateId) => renderFrequencyIdiom(sentence, intent!, templateId),
  },
  {
    id: "day-rule",
    family: "day-of-month",
    templates: Object.keys(DAY_RULES).map((_, index) => ({
      id: `rule-${index}`,
      split: "train" as const,
    })),
    intents: ["last_day_of_month", "last_business_day"].map((type) => ({
      frequency: { unit: "month" as const, interval: 1 },
      day: { type: type as "last_day_of_month" | "last_business_day" },
    })),
    render: (sentence, intent, templateId) => renderDayRule(sentence, intent!, templateId),
  },
  {
    id: "day-rule-at-time",
    family: "day-of-month",
    templates: Object.keys(DAY_RULES).flatMap((_, index) => [
      { id: `rule-at-${index}`, split: "train" as const },
      { id: `time-rule-${index}`, split: "train" as const },
    ]),
    intents: ["last_day_of_month", "last_business_day"].flatMap((type) =>
      COMPOUND_CLOCKS.map((time) => ({
        frequency: { unit: "month" as const, interval: 1 },
        day: { type: type as "last_day_of_month" | "last_business_day" },
        time,
      })),
    ),
    render: (sentence, intent, templateId) => renderDayRuleAtTime(sentence, intent!, templateId),
  },
  {
    id: "negative",
    family: "negative",
    templates: Object.keys(NEGATIVE_GROUPS).map((id) => ({
      id,
      split: "train" as const,
    })),
    intents: NEGATIVES.map(() => null),
    render: () => true,
  },
];

/**
 * Expand the catalog: one record per (kind, matching template, intent).
 * Templates that cannot render a given intent (day-group surface mismatch)
 * are skipped, not padded.
 */
export function* iterateKinds(wanted?: ReadonlySet<KindId>): Generator<KindExample> {
  let negativeIndex = 0;

  for (const kind of KINDS) {
    if (wanted !== undefined && !wanted.has(kind.id)) continue;

    for (const intent of kind.intents) {
      if (intent !== null) assertValid(intent);

      for (const template of kind.templates) {
        const sentence = new Sentence();

        if (kind.id === "negative") {
          const negative = NEGATIVES[negativeIndex]!;

          if (template.id !== negative.category) continue;
          sentence.add(negative.text, "O");
          if (!kind.render(sentence, intent, template.id)) continue;
        } else {
          sentence.clause();
          if (!kind.render(sentence, intent, template.id)) continue;
        }

        const name =
          kind.id === "negative"
            ? `${kind.id}/${template.id}/${negativeIndex}`
            : `${kind.id}/${template.id}`;

        yield {
          kind: kind.id,
          family: kind.family,
          template: name,
          split: template.split,
          intent,
          example: finish(sentence, name),
        };

        if (receivesCarrier(kind.id, template.id)) {
          for (let carrierIndex = 0; carrierIndex < POSITIVE_CARRIERS.length; carrierIndex++) {
            const carried = new Sentence();

            carried.add(POSITIVE_CARRIERS[carrierIndex]!, "O");
            carried.clause();
            if (!kind.render(carried, intent, template.id)) continue;
            const carriedName = `${kind.id}/carrier-${carrierIndex}-${template.id}`;

            yield {
              kind: kind.id,
              family: kind.family,
              template: carriedName,
              split: template.split,
              intent,
              example: finish(carried, carriedName),
            };
          }
        }
      }

      if (kind.id === "negative") negativeIndex++;
    }
  }
}

/** Eager compatibility view for callers that intentionally need every row. */
export function expandKinds(wanted?: ReadonlySet<KindId>): KindExample[] {
  return [...iterateKinds(wanted)];
}

/** Kind ids in catalog order. */
export function kindIds(): KindId[] {
  return KINDS.map((kind) => kind.id);
}
