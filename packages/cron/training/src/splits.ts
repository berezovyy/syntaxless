/**
 * Held-out renderer: reordered frames plus the reserved carriers.
 *
 * Train and validation use {@link renderIntent} with ordinary background prose.
 * The held-out split is rendered here instead, with a reordered frame per
 * family and carriers drawn from the single {@link RESERVED_PHRASES} list.
 * Because both the literal order and the carrier vocabulary differ from
 * anything the train renderer can produce, held-out frames cannot collide with
 * a training frame even after surface paraphrasing.
 */
import type { ScheduleIntent, Weekday } from "../../src/types/intent";
import { RESERVED_PHRASES } from "./background";
import type { RenderedExample } from "./renderer";
import type { Rng } from "./rng";
import type { Family } from "./sampler";
import { sampleIntent } from "./sampler";
import { Sentence } from "./sentence";

const WEEKDAY_NAME: Record<Weekday, string> = {
  monday: "Monday",
  tuesday: "Tuesday",
  wednesday: "Wednesday",
  thursday: "Thursday",
  friday: "Friday",
  saturday: "Saturday",
  sunday: "Sunday",
};

const UNIT_WORD: Record<string, string> = {
  minute: "minute",
  hour: "hour",
  day: "day",
  week: "week",
  month: "month",
};

const RESERVED: readonly string[] = [...RESERVED_PHRASES];

function reservedCarrier(rng: Rng): string {
  return RESERVED[rng.int(0, RESERVED.length - 1)]!;
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

function pluralUnit(unit: string, interval: number): string {
  const word = UNIT_WORD[unit] ?? unit;

  return interval === 1 ? word : `${word}s`;
}

function addWeekdayList(sentence: Sentence, weekdays: readonly Weekday[], connector: string): void {
  weekdays.forEach((weekday, index) => {
    if (index > 0) {
      sentence.add(connector, "GLUE", connector === "," ? "" : " ");
    }

    sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
  });
}

function heldoutInterval(sentence: Sentence, intent: ScheduleIntent): void {
  const { unit, interval } = intent.frequency;

  sentence.add("the", "GLUE");
  sentence.add("cycle", "GLUE");
  sentence.add("is", "GLUE");
  sentence.add(String(interval), "INTERVAL");
  sentence.add(pluralUnit(unit, interval), "GLUE");
}

function heldoutWeekdayList(sentence: Sentence, intent: ScheduleIntent): void {
  sentence.add("the", "GLUE");
  sentence.add("days", "GLUE");
  sentence.add("are", "GLUE");
  addWeekdayList(sentence, intent.weekdays!, "and");
}

function heldoutWeekdayRange(sentence: Sentence, intent: ScheduleIntent): void {
  const weekdays = intent.weekdays!;

  sentence.add("the", "GLUE");
  sentence.add("span", "GLUE");
  sentence.add("runs", "GLUE");
  sentence.add(WEEKDAY_NAME[weekdays[0]!], "WEEKDAY");
  sentence.add("through", "GLUE");
  sentence.add(WEEKDAY_NAME[weekdays[weekdays.length - 1]!], "WEEKDAY");
}

function heldoutOrdinalWeekday(sentence: Sentence, intent: ScheduleIntent): void {
  const { ordinal, weekday } = intent.ordinalWeekday!;

  sentence.add("the", "GLUE");
  sentence.add(ordinal, "ORDINAL");
  sentence.add(WEEKDAY_NAME[weekday], "WEEKDAY");
  sentence.add(",", "GLUE", "");
  sentence.add("monthly", "RECUR");
}

function heldoutDayOfMonth(sentence: Sentence, intent: ScheduleIntent): void {
  const day = intent.dayOfMonth!;

  sentence.add("the", "GLUE");
  sentence.add(String(day), "DOM");
  sentence.add(ordinalSuffix(day), "GLUE", "");
  sentence.add("day", "GLUE");
  sentence.add(",", "GLUE", "");
  sentence.add("monthly", "RECUR");
}

function heldoutTimeOfDay(sentence: Sentence, intent: ScheduleIntent): void {
  const time = intent.time!;
  const hour12 = time.hour % 12 === 0 ? 12 : time.hour % 12;

  sentence.add(String(hour12), "HOUR");
  sentence.add(":", "GLUE", "");
  sentence.add(String(time.minute).padStart(2, "0"), "MINUTE", "");
  sentence.add("daily", "RECUR");
}

function heldoutBounds(sentence: Sentence, intent: ScheduleIntent): void {
  const { start, end } = intent.bounds!;

  if (start !== undefined) {
    sentence.add(start, "DOM");

    if (end !== undefined) {
      sentence.add("through", "BOUND");
      sentence.add(end, "DOM");
    }
  } else {
    sentence.add("ending", "BOUND");
    sentence.add(end!, "DOM");
  }

  sentence.add(",", "GLUE", "");
  sentence.add("every", "RECUR");
  sentence.add("day", "GLUE");
}

function heldoutException(sentence: Sentence, intent: ScheduleIntent): void {
  sentence.add("excluding", "EXCEPT");
  addWeekdayList(sentence, intent.exceptions!.weekdays!, "and");
  sentence.add(",", "GLUE", "");
  sentence.add("every", "RECUR");

  if (intent.weekdays !== undefined && intent.weekdays.length > 0) {
    addWeekdayList(sentence, intent.weekdays, "and");
  } else {
    sentence.add("day", "GLUE");
  }
}

function heldoutTerse(sentence: Sentence, intent: ScheduleIntent): void {
  sentence.add("days", "GLUE");
  sentence.add("are", "GLUE");
  addWeekdayList(sentence, intent.weekdays!, "and");
}

function renderHeldoutFamily(sentence: Sentence, intent: ScheduleIntent, family: Family): void {
  switch (family) {
    case "interval":
      heldoutInterval(sentence, intent);
      break;
    case "weekday-list":
      heldoutWeekdayList(sentence, intent);
      break;
    case "weekday-range":
      heldoutWeekdayRange(sentence, intent);
      break;
    case "ordinal-weekday":
      heldoutOrdinalWeekday(sentence, intent);
      break;
    case "day-of-month":
      heldoutDayOfMonth(sentence, intent);
      break;
    case "time-of-day":
      heldoutTimeOfDay(sentence, intent);
      break;
    case "bounds":
      heldoutBounds(sentence, intent);
      break;
    case "exception":
      heldoutException(sentence, intent);
      break;
    case "terse":
      heldoutTerse(sentence, intent);
      break;
    case "negative":
      throw new Error("renderHeldoutFamily does not take the negative family");
  }
}

export interface HeldoutExample extends RenderedExample {
  intent: ScheduleIntent | null;
}

/**
 * Render one held-out example for `family`. Draws the intent with `sampleIntent`
 * (so the expected value is ground truth, as in training), then records a
 * reordered frame with a reserved carrier. The reserved carrier is prefixed
 * always and suffixed sometimes, so every held-out frame carries phrase
 * vocabulary the train renderer is forbidden to emit.
 */
export function renderHeldout(family: Family, rng: Rng): HeldoutExample {
  const sample = sampleIntent(rng, family);
  const sentence = new Sentence();

  sentence.add(reservedCarrier(rng), "O");

  if (sample.intent !== null) {
    sentence.clause();
    renderHeldoutFamily(sentence, sample.intent, family);
  }

  if (rng.bool(0.2)) sentence.add(reservedCarrier(rng), "O");

  return {
    text: sentence.text,
    spans: sentence.spans,
    template: `${family}/heldout`,
    style: "heldout",
    intent: sample.intent,
  };
}
