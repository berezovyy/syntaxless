import {
  boundLabel,
  hourLabel,
  intervalLabel,
  meridiemLabel,
  minuteLabel,
  ordinalLabel,
  unitLabel,
  weekdayLabel,
} from "../../src/model/labels.js";
import type { SpeechAct } from "../../src/model/speech-acts";
/**
 * Deterministic renderer: sampled {@link ScheduleIntent} -> surface text + spans.
 *
 * Every fragment is inserted through {@link Sentence.add}, which is the single
 * source of span labels. The renderer never parses text and never derives a
 * label from a rendered string: the label chosen at insertion is final. It
 * imports the intent types and the one label vocabulary, and must not import
 * the runtime parser.
 *
 * All schedule surface words come from the runtime lexicon
 * (`../../src/parser/lexicon.js`); the renderer declares no literal schedule
 * vocabulary of its own, so there is exactly one copy to keep in sync.
 *
 * Alignment contract: every `add()` inserts one fragment covering whole
 * tokenizer tokens, so spans line up with `tokenize()` output (Phase 13).
 */
import {
  numberWord,
  WEEKDAY_ABBR,
  WEEKDAY_NAME,
  WEEKDAY_PLURAL,
  WORDS,
} from "../../src/parser/lexicon.js";
import type { ScheduleIntent, Weekday } from "../../src/types/intent";
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
  speechAct: SpeechAct;
}

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
      sentence.add(connector, "GLUE", connector === WORDS.glue.comma ? "" : " ");
    }

    sentence.add(weekdayText(weekday, mode), weekdayLabel(weekday));
  });
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

function addDayOfMonth(sentence: Sentence, day: number): void {
  sentence.add(String(day), "DOM_LIT");
  sentence.add(ordinalSuffix(day), "GLUE", "");
}

function qualifierFor(hour24: number): string {
  if (hour24 < 12) return WORDS.qualifier.morning;
  if (hour24 < 17) return WORDS.qualifier.afternoon;
  if (hour24 < 21) return WORDS.qualifier.evening;

  return WORDS.qualifier.night;
}

function renderInterval(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const { unit, interval } = intent.frequency;
  const unitText = plural(unit, interval);

  switch (style) {
    case "every":
      sentence.add(WORDS.recur.every, "RECUR");
      if (interval !== 1) sentence.add(String(interval), intervalLabel(interval));
      sentence.add(unitText, unitLabel(unit));
      break;
    case "spoken":
      sentence.add(WORDS.recur.every, "RECUR");

      if (interval !== 1) {
        sentence.add(
          interval >= 2 && interval <= 12 ? numberWord(interval) : String(interval),
          intervalLabel(interval),
        );
      }

      sentence.add(unitText, unitLabel(unit));
      break;
    case "each":
      sentence.add(WORDS.recur.each, "RECUR");
      if (interval !== 1) sentence.add(String(interval), intervalLabel(interval));
      sentence.add(unitText, unitLabel(unit));
      break;
    case "apart":
      if (interval !== 1) sentence.add(String(interval), intervalLabel(interval));
      sentence.add(unitText, unitLabel(unit));
      sentence.add(WORDS.glue.apart, "GLUE");
      break;
    case "recurring":
      sentence.add(WORDS.recur.recurring, "RECUR");
      sentence.add(WORDS.recur.every, "RECUR");
      if (interval !== 1) sentence.add(String(interval), intervalLabel(interval));
      sentence.add(unitText, unitLabel(unit));
      break;
    default:
      throw new RangeError(`interval style: ${style}`);
  }
}

function renderWeekdayList(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const weekdays = intent.weekdays!;

  switch (style) {
    case "on-days":
      sentence.add(WORDS.glue.on, "GLUE");
      addWeekdayList(sentence, weekdays, "full", WORDS.glue.and);
      break;
    case "every-days":
      sentence.add(WORDS.recur.every, "RECUR");
      addWeekdayList(sentence, weekdays, "plural", WORDS.glue.and);
      break;
    case "abbrev":
      sentence.add(WORDS.glue.on, "GLUE");
      addWeekdayList(sentence, weekdays, "abbrev", WORDS.glue.comma);
      break;
    case "ampersand":
      sentence.add(WORDS.glue.on, "GLUE");
      addWeekdayList(sentence, weekdays, "plural", WORDS.glue.ampersand);
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
      sentence.add(WEEKDAY_NAME[first], weekdayLabel(first));
      sentence.add(WORDS.range.through, boundLabel("between"));
      sentence.add(WEEKDAY_NAME[last], weekdayLabel(last));
      break;
    case "from-to":
      sentence.add(WORDS.range.from, boundLabel("start"));
      sentence.add(WEEKDAY_NAME[first], weekdayLabel(first));
      sentence.add(WORDS.range.to, boundLabel("end-excl"));
      sentence.add(WEEKDAY_NAME[last], weekdayLabel(last));
      break;
    case "dash":
      sentence.add(WEEKDAY_ABBR[first], weekdayLabel(first));
      sentence.add(WORDS.glue.dash, boundLabel("between"), "");
      sentence.add(WEEKDAY_ABBR[last], weekdayLabel(last));
      break;
    case "every-through":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WEEKDAY_PLURAL[first], weekdayLabel(first));
      sentence.add(WORDS.range.through, boundLabel("between"));
      sentence.add(WEEKDAY_PLURAL[last], weekdayLabel(last));
      break;
    default:
      throw new RangeError(`weekday-range style: ${style}`);
  }
}

function renderOrdinalWeekday(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const { ordinal, weekday } = intent.ordinalWeekday!;

  switch (style) {
    case "the-ordinal-of":
      sentence.add(WORDS.glue.the, "GLUE");
      sentence.add(ordinal, ordinalLabel(ordinal));
      sentence.add(WEEKDAY_NAME[weekday], weekdayLabel(weekday));
      sentence.add(WORDS.glue.of, "GLUE");
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.unit.month, unitLabel("month"));
      break;
    case "every-ordinal":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(ordinal, ordinalLabel(ordinal));
      sentence.add(WEEKDAY_NAME[weekday], weekdayLabel(weekday));
      sentence.add(WORDS.glue.of, "GLUE");
      sentence.add(WORDS.glue.the, "GLUE");
      sentence.add(WORDS.unit.month, unitLabel("month"));
      break;
    case "last-of":
      sentence.add(WORDS.glue.the, "GLUE");
      sentence.add(ordinal, ordinalLabel(ordinal));
      sentence.add(WEEKDAY_NAME[weekday], weekdayLabel(weekday));
      sentence.add(WORDS.glue.of, "GLUE");
      sentence.add(WORDS.glue.the, "GLUE");
      sentence.add(WORDS.unit.month, unitLabel("month"));
      break;
    case "ordinal-of-each":
      sentence.add(ordinal, ordinalLabel(ordinal));
      sentence.add(WEEKDAY_NAME[weekday], weekdayLabel(weekday));
      sentence.add(WORDS.glue.of, "GLUE");
      sentence.add(WORDS.recur.each, "RECUR");
      sentence.add(WORDS.unit.month, unitLabel("month"));
      break;
    default:
      throw new RangeError(`ordinal-weekday style: ${style}`);
  }
}

function renderDayOfMonth(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const day = intent.dayOfMonth!;

  switch (style) {
    case "on-the":
      sentence.add(WORDS.glue.on, "GLUE");
      sentence.add(WORDS.glue.the, "GLUE");
      addDayOfMonth(sentence, day);
      sentence.add(WORDS.glue.of, "GLUE");
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.unit.month, unitLabel("month"));
      break;
    case "every-month-on":
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.unit.month, unitLabel("month"));
      sentence.add(WORDS.glue.on, "GLUE");
      sentence.add(WORDS.glue.the, "GLUE");
      addDayOfMonth(sentence, day);
      break;
    case "monthly-on":
      sentence.add(WORDS.recur.monthly, unitLabel("month"));
      sentence.add(WORDS.glue.on, "GLUE");
      sentence.add(WORDS.glue.the, "GLUE");
      addDayOfMonth(sentence, day);
      break;
    case "the-of-every":
      sentence.add(WORDS.glue.the, "GLUE");
      addDayOfMonth(sentence, day);
      sentence.add(WORDS.glue.of, "GLUE");
      sentence.add(WORDS.recur.every, "RECUR");
      sentence.add(WORDS.unit.month, unitLabel("month"));
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
  const meridiem = hour24 >= 12 ? WORDS.meridiem.pm : WORDS.meridiem.am;
  const padded = String(minute).padStart(2, "0");

  switch (style) {
    case "digits":
      sentence.add(WORDS.glue.at, "GLUE");
      sentence.add(String(hour12), hourLabel(hour24));
      sentence.add(WORDS.glue.colon, "GLUE", "");
      sentence.add(padded, minuteLabel(minute), "");
      sentence.add(meridiem, meridiemLabel(hour24 >= 12 ? "pm" : "am"));
      break;

    case "spoken": {
      sentence.add(WORDS.glue.at, "GLUE");
      sentence.add(numberWord(hour12), hourLabel(hour24));

      if (minute !== 0) {
        const minuteText =
          minute === 0 || minute === 15 || minute === 30 || minute === 45
            ? numberWord(minute)
            : padded;

        sentence.add(minuteText, minuteLabel(minute));
      }

      sentence.add(meridiem, meridiemLabel(hour24 >= 12 ? "pm" : "am"));
      break;
    }

    case "qualifier":
      sentence.add(WORDS.glue.at, "GLUE");
      sentence.add(String(hour12), hourLabel(hour24));

      if (minute !== 0) {
        sentence.add(WORDS.glue.colon, "GLUE", "");
        sentence.add(padded, minuteLabel(minute), "");
      }

      sentence.add(qualifierFor(hour24), meridiemLabel(hour24 >= 12 ? "pm" : "am"));
      break;
    case "24h":
      sentence.add(WORDS.glue.at, "GLUE");
      sentence.add(String(hour24), hourLabel(hour24));
      sentence.add(WORDS.glue.colon, "GLUE", "");
      sentence.add(padded, minuteLabel(minute), "");
      break;
    case "compact":
      sentence.add(WORDS.glue.at, "GLUE");
      sentence.add(String(hour12), hourLabel(hour24));
      sentence.add(WORDS.glue.colon, "GLUE", "");
      sentence.add(padded, minuteLabel(minute), "");
      sentence.add(meridiem, meridiemLabel(hour24 >= 12 ? "pm" : "am"), "");
      break;
    default:
      throw new RangeError(`time-of-day style: ${style}`);
  }
}

function renderBounds(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const { start, end } = intent.bounds!;

  sentence.add(WORDS.recur.every, "RECUR");
  sentence.add(WORDS.unit.day, unitLabel("day"));

  switch (style) {
    case "from-to":
      if (start === undefined && end === undefined) {
        throw new Error("bounds: no start or end");
      }

      if (start !== undefined) {
        sentence.add(WORDS.bounds.from, boundLabel("start"));
        sentence.add(start, "DOM_LIT");
      }

      if (end !== undefined) {
        sentence.add(WORDS.bounds.to, boundLabel("end-excl"));
        sentence.add(end, "DOM_LIT");
      }

      break;
    case "starting":
      if (start !== undefined) {
        sentence.add(WORDS.bounds.starting, boundLabel("start"));
        sentence.add(start, "DOM_LIT");
      } else {
        sentence.add(WORDS.bounds.until, boundLabel("end-excl"));
        sentence.add(end!, "DOM_LIT");
      }

      break;
    case "until":
      if (end !== undefined) {
        sentence.add(WORDS.bounds.until, boundLabel("end-excl"));
        sentence.add(end, "DOM_LIT");
      } else {
        sentence.add(WORDS.bounds.starting, boundLabel("start"));
        sentence.add(start!, "DOM_LIT");
      }

      break;
    case "between":
      if (start !== undefined && end !== undefined) {
        sentence.add(WORDS.bounds.between, boundLabel("between"));
        sentence.add(start, "DOM_LIT");
        sentence.add(WORDS.glue.and, "GLUE");
        sentence.add(end, "DOM_LIT");
      } else if (start !== undefined) {
        sentence.add(WORDS.bounds.from, boundLabel("start"));
        sentence.add(start, "DOM_LIT");
      } else {
        sentence.add(WORDS.bounds.until, boundLabel("end-excl"));
        sentence.add(end!, "DOM_LIT");
      }

      break;
    default:
      throw new RangeError(`bounds style: ${style}`);
  }
}

function renderBaseFrequency(sentence: Sentence, intent: ScheduleIntent): void {
  sentence.add(WORDS.recur.every, "RECUR");

  if (intent.weekdays !== undefined && intent.weekdays.length > 0) {
    addWeekdayList(sentence, intent.weekdays, "full", WORDS.glue.and);
  } else {
    sentence.add(WORDS.unit.day, unitLabel("day"));
  }
}

function renderException(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const weekdays = intent.exceptions!.weekdays!;

  renderBaseFrequency(sentence, intent);

  switch (style) {
    case "except":
      sentence.add(WORDS.exception.except, "EXCEPT");
      addWeekdayList(sentence, weekdays, "full", WORDS.glue.and);
      break;
    case "excluding":
      sentence.add(WORDS.exception.excluding, "EXCEPT");
      addWeekdayList(sentence, weekdays, "full", WORDS.glue.and);
      break;
    case "but-not":
      sentence.add(WORDS.exception.but, "EXCEPT");
      sentence.add(WORDS.exception.not, "GLUE");
      addWeekdayList(sentence, weekdays, "full", WORDS.glue.and);
      break;
    case "not-on":
      sentence.add(WORDS.exception.not, "EXCEPT");
      sentence.add(WORDS.glue.on, "GLUE");
      addWeekdayList(sentence, weekdays, "full", WORDS.glue.and);
      break;
    default:
      throw new RangeError(`exception style: ${style}`);
  }
}

function renderTerse(sentence: Sentence, intent: ScheduleIntent, style: string): void {
  const weekdays = intent.weekdays!;

  switch (style) {
    case "bare":
      addWeekdayList(sentence, weekdays, "full", WORDS.glue.and);
      break;
    case "every":
      sentence.add(WORDS.recur.every, "RECUR");
      addWeekdayList(sentence, weekdays, "full", WORDS.glue.and);
      break;
    case "on":
      sentence.add(WORDS.glue.on, "GLUE");
      addWeekdayList(sentence, weekdays, "full", WORDS.glue.and);
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
    speechAct: "request",
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
    speechAct: "none",
  };
}
