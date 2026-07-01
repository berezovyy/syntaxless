/**
 * Deterministic renderer: sampled {@link ScheduleIntent} -> surface text + spans.
 *
 * Every fragment is inserted through {@link Sentence.add}, which is the single
 * source of span labels. The renderer never parses text and never derives a
 * label from a rendered string: the label chosen at insertion is final. It
 * imports the intent types and the one label vocabulary, and must not import
 * the runtime parser.
 *
 * Alignment contract: every `add()` inserts one fragment covering whole
 * tokenizer tokens, so spans line up with `tokenize()` output (Phase 13).
 */
import type { FrequencyUnit, Ordinal, ScheduleIntent, Weekday } from "../../src/types/intent";
import {
  prefix as carrierPrefix,
  sentence as proseSentence,
  suffix as carrierSuffix,
} from "./background";
import type { Rng } from "./rng";
import type { Family } from "./sampler";
import { Sentence, type Span } from "./sentence";

export interface RenderedExample {
  text: string;
  spans: Span[];
  /** `${family}/${style}`; recorded as the example's template name. */
  template: string;
  style: string;
}

const ONES = [
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

const TENS: Readonly<Record<number, string>> = {
  20: "twenty",
  30: "thirty",
  40: "forty",
  50: "fifty",
};

function numberWord(value: number): string {
  if (value < 20) return ONES[value]!;
  const tens = Math.floor(value / 10) * 10;
  const ones = value % 10;

  return ones === 0 ? TENS[tens]! : `${TENS[tens]!} ${ONES[ones]!}`;
}

const WEEKDAY_NAME: Record<Weekday, string> = {
  monday: "Monday",
  tuesday: "Tuesday",
  wednesday: "Wednesday",
  thursday: "Thursday",
  friday: "Friday",
  saturday: "Saturday",
  sunday: "Sunday",
};

const WEEKDAY_PLURAL: Record<Weekday, string> = {
  monday: "Mondays",
  tuesday: "Tuesdays",
  wednesday: "Wednesdays",
  thursday: "Thursdays",
  friday: "Fridays",
  saturday: "Saturdays",
  sunday: "Sundays",
};

const WEEKDAY_ABBR: Record<Weekday, string> = {
  monday: "Mon",
  tuesday: "Tue",
  wednesday: "Wed",
  thursday: "Thu",
  friday: "Fri",
  saturday: "Sat",
  sunday: "Sun",
};

const ORDINAL_WORD: Record<Ordinal, string> = {
  first: "first",
  second: "second",
  third: "third",
  fourth: "fourth",
  last: "last",
};

const UNIT_WORD: Record<FrequencyUnit, string> = {
  minute: "minute",
  hour: "hour",
  day: "day",
  week: "week",
  month: "month",
};

const STYLES: Record<Family, readonly string[]> = {
  interval: ["every", "spoken", "each", "apart", "recurring"],
  "weekday-list": ["on-days", "every-days", "abbrev", "ampersand"],
  "weekday-range": ["through", "from-to", "dash", "every-through"],
  "ordinal-weekday": ["the-ordinal-of", "every-ordinal", "last-of", "ordinal-of-each"],
  "day-of-month": ["on-the", "every-month-on", "monthly-on", "the-of-every"],
  "time-of-day": ["digits", "spoken", "qualifier", "24h", "compact"],
  bounds: ["from-to", "starting", "until", "between"],
  exception: ["except", "excluding", "but-not", "not-on"],
  terse: ["bare", "every", "on"],
  negative: [],
};

function plural(word: string, count: number): string {
  return count === 1 ? word : `${word}s`;
}

type WeekdayMode = "full" | "plural" | "abbrev";

function weekdayText(weekday: Weekday, mode: WeekdayMode): string {
  switch (mode) {
    case "full":
      return WEEKDAY_NAME[weekday];
    case "plural":
      return WEEKDAY_PLURAL[weekday];
    case "abbrev":
      return WEEKDAY_ABBR[weekday];
  }
}

function addWeekdayList(
  sentence: Sentence,
  weekdays: readonly Weekday[],
  mode: WeekdayMode,
  connector: string,
): void {
  weekdays.forEach((weekday, index) => {
    if (index > 0) {
      sentence.add(connector, "GLUE", connector === "," ? "" : " ");
    }

    sentence.add(weekdayText(weekday, mode), "WEEKDAY");
  });
}

function ordinalSuffix(day: number): string {
  if (day % 100 >= 11 && day % 100 <= 13) return "th";

  switch (day % 10) {
    case 1:
      return "st";
    case 2:
      return "nd";
    case 3:
      return "rd";
    default:
      return "th";
  }
}

function addDayOfMonth(sentence: Sentence, day: number): void {
  sentence.add(String(day), "DOM");
  sentence.add(ordinalSuffix(day), "GLUE", "");
}

function qualifierFor(hour24: number): string {
  if (hour24 < 12) return "in the morning";
  if (hour24 < 17) return "in the afternoon";
  if (hour24 < 21) return "in the evening";

  return "at night";
}

function renderInterval(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const { unit, interval } = intent.frequency;
  const unitText = plural(UNIT_WORD[unit], interval);

  switch (style) {
    case "every":
      sentence.add("every", "RECUR");
      sentence.add(String(interval), "INTERVAL");
      sentence.add(unitText, "GLUE");
      break;
    case "spoken":
      sentence.add("every", "RECUR");
      sentence.add(numberWord(interval), "INTERVAL");
      sentence.add(unitText, "GLUE");
      break;
    case "each":
      sentence.add("each", "RECUR");
      sentence.add(String(interval), "INTERVAL");
      sentence.add(unitText, "GLUE");
      break;
    case "apart":
      sentence.add(String(interval), "INTERVAL");
      sentence.add(unitText, "GLUE");
      sentence.add("apart", "GLUE");
      break;
    case "recurring":
      sentence.add("recurring", "RECUR");
      sentence.add("every", "RECUR");
      sentence.add(String(interval), "INTERVAL");
      sentence.add(unitText, "GLUE");
      break;
    default:
      throw new RangeError(`interval style: ${style}`);
  }
}

function renderWeekdayList(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const weekdays = intent.weekdays!;

  switch (style) {
    case "on-days":
      sentence.add("on", "GLUE");
      addWeekdayList(sentence, weekdays, "full", "and");
      break;
    case "every-days":
      sentence.add("every", "RECUR");
      addWeekdayList(sentence, weekdays, "plural", "and");
      break;
    case "abbrev":
      sentence.add("on", "GLUE");
      addWeekdayList(sentence, weekdays, "abbrev", ",");
      break;
    case "ampersand":
      sentence.add("on", "GLUE");
      addWeekdayList(sentence, weekdays, "plural", "&");
      break;
    default:
      throw new RangeError(`weekday-list style: ${style}`);
  }
}

function renderWeekdayRange(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const weekdays = intent.weekdays!;
  const first = weekdays[0]!;
  const last = weekdays[weekdays.length - 1]!;

  switch (style) {
    case "through":
      sentence.add(WEEKDAY_NAME[first], "WEEKDAY");
      sentence.add("through", "GLUE");
      sentence.add(WEEKDAY_NAME[last], "WEEKDAY");
      break;
    case "from-to":
      sentence.add("from", "GLUE");
      sentence.add(WEEKDAY_NAME[first], "WEEKDAY");
      sentence.add("to", "GLUE");
      sentence.add(WEEKDAY_NAME[last], "WEEKDAY");
      break;
    case "dash":
      sentence.add(WEEKDAY_ABBR[first], "WEEKDAY");
      sentence.add("-", "GLUE", "");
      sentence.add(WEEKDAY_ABBR[last], "WEEKDAY");
      break;
    case "every-through":
      sentence.add("every", "RECUR");
      sentence.add(WEEKDAY_PLURAL[first], "WEEKDAY");
      sentence.add("through", "GLUE");
      sentence.add(WEEKDAY_PLURAL[last], "WEEKDAY");
      break;
    default:
      throw new RangeError(`weekday-range style: ${style}`);
  }
}

function renderOrdinalWeekday(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const { ordinal, weekday } = intent.ordinalWeekday!;

  switch (style) {
    case "the-ordinal-of":
      sentence.add("the", "GLUE");
      sentence.add(ORDINAL_WORD[ordinal], "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      sentence.add("of", "GLUE");
      sentence.add("every", "RECUR");
      sentence.add("month", "GLUE");
      break;
    case "every-ordinal":
      sentence.add("every", "RECUR");
      sentence.add(ORDINAL_WORD[ordinal], "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      sentence.add("of", "GLUE");
      sentence.add("the", "GLUE");
      sentence.add("month", "GLUE");
      break;
    case "last-of":
      sentence.add("the", "GLUE");
      sentence.add(ORDINAL_WORD[ordinal], "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      sentence.add("of", "GLUE");
      sentence.add("the", "GLUE");
      sentence.add("month", "GLUE");
      break;
    case "ordinal-of-each":
      sentence.add(ORDINAL_WORD[ordinal], "ORDINAL");
      sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
      sentence.add("of", "GLUE");
      sentence.add("each", "RECUR");
      sentence.add("month", "GLUE");
      break;
    default:
      throw new RangeError(`ordinal-weekday style: ${style}`);
  }
}

function renderDayOfMonth(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const day = intent.dayOfMonth!;

  switch (style) {
    case "on-the":
      sentence.add("on", "GLUE");
      sentence.add("the", "GLUE");
      addDayOfMonth(sentence, day);
      sentence.add("of", "GLUE");
      sentence.add("every", "RECUR");
      sentence.add("month", "GLUE");
      break;
    case "every-month-on":
      sentence.add("every", "RECUR");
      sentence.add("month", "GLUE");
      sentence.add("on", "GLUE");
      sentence.add("the", "GLUE");
      addDayOfMonth(sentence, day);
      break;
    case "monthly-on":
      sentence.add("monthly", "RECUR");
      sentence.add("on", "GLUE");
      sentence.add("the", "GLUE");
      addDayOfMonth(sentence, day);
      break;
    case "the-of-every":
      sentence.add("the", "GLUE");
      addDayOfMonth(sentence, day);
      sentence.add("of", "GLUE");
      sentence.add("every", "RECUR");
      sentence.add("month", "GLUE");
      break;
    default:
      throw new RangeError(`day-of-month style: ${style}`);
  }
}

function renderTimeOfDay(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const time = intent.time!;
  const hour24 = time.hour;
  const minute = time.minute;
  const hour12 = hour24 % 12 === 0 ? 12 : hour24 % 12;
  const meridiem = hour24 >= 12 ? "pm" : "am";
  const padded = String(minute).padStart(2, "0");

  switch (style) {
    case "digits":
      sentence.add("at", "GLUE");
      sentence.add(String(hour12), "HOUR");
      sentence.add(":", "GLUE", "");
      sentence.add(padded, "MINUTE", "");
      sentence.add(meridiem, "MERIDIEM");
      break;
    case "spoken":
      sentence.add("at", "GLUE");
      sentence.add(numberWord(hour12), "HOUR");
      if (minute !== 0) sentence.add(numberWord(minute), "MINUTE");
      sentence.add(meridiem, "MERIDIEM");
      break;
    case "qualifier":
      sentence.add("at", "GLUE");
      sentence.add(String(hour12), "HOUR");

      if (minute !== 0) {
        sentence.add(":", "GLUE", "");
        sentence.add(padded, "MINUTE", "");
      }

      sentence.add(qualifierFor(hour24), "MERIDIEM");
      break;
    case "24h":
      sentence.add("at", "GLUE");
      sentence.add(String(hour24), "HOUR");
      sentence.add(":", "GLUE", "");
      sentence.add(padded, "MINUTE", "");
      break;
    case "compact":
      sentence.add("at", "GLUE");
      sentence.add(String(hour12), "HOUR");
      sentence.add(":", "GLUE", "");
      sentence.add(padded, "MINUTE", "");
      sentence.add(meridiem, "MERIDIEM", "");
      break;
    default:
      throw new RangeError(`time-of-day style: ${style}`);
  }
}

function renderBounds(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const { start, end } = intent.bounds!;

  sentence.add("every", "RECUR");
  sentence.add("day", "GLUE");

  switch (style) {
    case "from-to":
      if (start === undefined && end === undefined) {
        throw new Error("bounds: no start or end");
      }

      if (start !== undefined) {
        sentence.add("from", "BOUND");
        sentence.add(start, "DOM");
      }

      if (end !== undefined) {
        sentence.add("to", "BOUND");
        sentence.add(end, "DOM");
      }

      break;
    case "starting":
      if (start !== undefined) {
        sentence.add("starting", "BOUND");
        sentence.add(start, "DOM");
      } else {
        sentence.add("until", "BOUND");
        sentence.add(end!, "DOM");
      }

      break;
    case "until":
      if (end !== undefined) {
        sentence.add("until", "BOUND");
        sentence.add(end, "DOM");
      } else {
        sentence.add("starting", "BOUND");
        sentence.add(start!, "DOM");
      }

      break;
    case "between":
      if (start !== undefined && end !== undefined) {
        sentence.add("between", "BOUND");
        sentence.add(start, "DOM");
        sentence.add("and", "GLUE");
        sentence.add(end, "DOM");
      } else if (start !== undefined) {
        sentence.add("from", "BOUND");
        sentence.add(start, "DOM");
      } else {
        sentence.add("until", "BOUND");
        sentence.add(end!, "DOM");
      }

      break;
    default:
      throw new RangeError(`bounds style: ${style}`);
  }
}

function renderBaseFrequency(sentence: Sentence, intent: ScheduleIntent): void {
  sentence.add("every", "RECUR");

  if (intent.weekdays !== undefined && intent.weekdays.length > 0) {
    addWeekdayList(sentence, intent.weekdays, "full", "and");
  } else {
    sentence.add("day", "GLUE");
  }
}

function renderException(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const weekdays = intent.exceptions!.weekdays!;

  renderBaseFrequency(sentence, intent);

  switch (style) {
    case "except":
      sentence.add("except", "EXCEPT");
      addWeekdayList(sentence, weekdays, "full", "and");
      break;
    case "excluding":
      sentence.add("excluding", "EXCEPT");
      addWeekdayList(sentence, weekdays, "full", "and");
      break;
    case "but-not":
      sentence.add("but", "EXCEPT");
      sentence.add("not", "GLUE");
      addWeekdayList(sentence, weekdays, "full", "and");
      break;
    case "not-on":
      sentence.add("not", "EXCEPT");
      sentence.add("on", "GLUE");
      addWeekdayList(sentence, weekdays, "full", "and");
      break;
    default:
      throw new RangeError(`exception style: ${style}`);
  }
}

function renderTerse(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const weekdays = intent.weekdays!;

  switch (style) {
    case "bare":
      addWeekdayList(sentence, weekdays, "full", "and");
      break;
    case "every":
      sentence.add("every", "RECUR");
      addWeekdayList(sentence, weekdays, "full", "and");
      break;
    case "on":
      sentence.add("on", "GLUE");
      addWeekdayList(sentence, weekdays, "full", "and");
      break;
    default:
      throw new RangeError(`terse style: ${style}`);
  }
}

function renderFamily(
  sentence: Sentence,
  intent: ScheduleIntent,
  family: Family,
  style: string,
): void {
  switch (family) {
    case "interval":
      renderInterval(sentence, intent, style);
      break;
    case "weekday-list":
      renderWeekdayList(sentence, intent, style);
      break;
    case "weekday-range":
      renderWeekdayRange(sentence, intent, style);
      break;
    case "ordinal-weekday":
      renderOrdinalWeekday(sentence, intent, style);
      break;
    case "day-of-month":
      renderDayOfMonth(sentence, intent, style);
      break;
    case "time-of-day":
      renderTimeOfDay(sentence, intent, style);
      break;
    case "bounds":
      renderBounds(sentence, intent, style);
      break;
    case "exception":
      renderException(sentence, intent, style);
      break;
    case "terse":
      renderTerse(sentence, intent, style);
      break;
    case "negative":
      throw new Error("renderIntent does not render the negative family");
  }
}

/**
 * Render one sampled intent for `family`. When `style` is omitted the renderer
 * draws one with `rng`; either way the chosen style is returned so the caller
 * can record it as the template name.
 */
export function renderIntent(
  intent: ScheduleIntent,
  family: Family,
  rng: Rng,
  style?: string,
): RenderedExample {
  const styles = STYLES[family];

  if (styles.length === 0) {
    throw new Error(`renderIntent: no styles for family ${family}`);
  }

  const chosen = style ?? rng.pick(styles);

  if (!styles.includes(chosen)) {
    throw new RangeError(`renderIntent: unknown style ${chosen} (${family})`);
  }

  const sentence = new Sentence();

  if (rng.bool(0.5)) sentence.add(carrierPrefix(rng), "O");
  sentence.clause();
  renderFamily(sentence, intent, family, chosen);
  if (rng.bool(0.2)) sentence.add(carrierSuffix(rng), "O");

  return {
    text: sentence.text,
    spans: sentence.spans,
    template: `${family}/${chosen}`,
    style: chosen,
  };
}

/**
 * Render an all-`O` negative from ordinary prose. It carries no intent and never
 * calls the parser.
 */
export function renderNegative(rng: Rng): RenderedExample {
  const sentence = new Sentence();

  sentence.add(proseSentence(rng), "O");

  return {
    text: sentence.text,
    spans: sentence.spans,
    template: "negative/prose",
    style: "prose",
  };
}
