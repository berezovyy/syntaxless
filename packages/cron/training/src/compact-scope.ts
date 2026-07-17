/** Compact-only additions and coverage of the public intent fields. */
import type { ScheduleIntent } from "../../src/types/intent.js";
import { type KindDef, KINDS } from "./kinds.js";

export type CompactKind = Omit<KindDef, "id"> & { id: string };

const unit = KINDS.find((kind) => kind.id === "unit-interval")!;
const counted = KINDS.find((kind) => kind.id === "n-interval")!;

export const HOURLY_MINUTE: CompactKind = {
  id: "hour-at-minute",
  family: "time-of-day",
  templates: [
    "at-minute",
    "past-hour",
    "minute-first",
    "hourly-minute",
    "each-minute",
    "minute-of-hour",
  ].map((id) => ({ id, split: "train" })),
  intents: [1, 2, 3, 6, 12, 24, 25, 48].flatMap((interval) =>
    [0, 15, 37, 59].map((minute) => ({
      frequency: { unit: "hour" as const, interval },
      time: { minute },
    })),
  ),
  render(sentence, intent, template) {
    if (!intent?.time || intent.time.hour !== undefined || intent.frequency.unit !== "hour") {
      return false;
    }

    const { interval } = intent.frequency;
    const minute = () => sentence.add(String(intent.time!.minute), "MINUTE_LIT");

    const frequency = (each = false) =>
      interval === 1
        ? unit.render(sentence, intent, each ? "each-unit" : "every-unit")
        : counted.render(sentence, intent, each ? "each-n" : "every-n");

    switch (template) {
      case "at-minute":
      case "each-minute":
        frequency(template === "each-minute");
        sentence.add("at minute", "GLUE");
        minute();
        break;
      case "past-hour":
        frequency();
        sentence.add("at", "GLUE");
        minute();
        sentence.add("minutes past the hour", "GLUE");
        break;
      case "minute-first":
        sentence.add("at minute", "GLUE");
        minute();
        sentence.add("of", "GLUE");
        frequency();
        break;
      case "hourly-minute":
        if (interval !== 1) return false;
        sentence.add("hourly", "UNIT_HOUR");
        sentence.add("at minute", "GLUE");
        minute();
        break;
      case "minute-of-hour":
        sentence.add("at", "GLUE");
        minute();
        sentence.add("minutes past", "GLUE");
        frequency();
        break;
      default:
        return false;
    }

    return true;
  },
};

export const REQUIRED_SCOPE = [
  ...["minute", "hour", "day", "week", "month"].flatMap((unit) => [
    `unit/${unit}`,
    `unit-step/${unit}`,
    `interval/${unit}`,
  ]),
  "large/minute",
  "large/hour",
  "time/clock",
  "time/hour-minute",
  "selector/weekdays",
  "selector/dayOfMonth",
  ...["first", "second", "third", "fourth", "last"].map((ordinal) => `ordinal/${ordinal}`),
  "day/last_day_of_month",
  "day/last_business_day",
  "bounds/start",
  "bounds/end",
  "exceptions/dates",
  "exceptions/weekdays",
] as const;

export function intentScope(intent: ScheduleIntent): string[] {
  const { unit, interval } = intent.frequency;
  const tags = [`unit/${unit}`];

  if (interval === 1) tags.push(`unit-step/${unit}`);
  if (interval > 1) tags.push(`interval/${unit}`);

  if ((unit === "minute" && interval >= 60) || (unit === "hour" && interval >= 24)) {
    tags.push(`large/${unit}`);
  }

  if (intent.time) tags.push(intent.time.hour === undefined ? "time/hour-minute" : "time/clock");
  if (intent.weekdays) tags.push("selector/weekdays");
  if (intent.dayOfMonth) tags.push("selector/dayOfMonth");
  if (intent.ordinalWeekday) tags.push(`ordinal/${intent.ordinalWeekday.ordinal}`);
  if (intent.day) tags.push(`day/${intent.day.type}`);
  if (intent.bounds?.start) tags.push("bounds/start");
  if (intent.bounds?.end) tags.push("bounds/end");
  if (intent.exceptions?.dates) tags.push("exceptions/dates");
  if (intent.exceptions?.weekdays) tags.push("exceptions/weekdays");

  return tags;
}
