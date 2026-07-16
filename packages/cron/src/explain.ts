import { compile } from "./compiler/cron.js";
import type { ScheduleIntent } from "./types/intent.js";
import { validate } from "./validate.js";

const padded = (value: number): string => String(value).padStart(2, "0");

/** Describe every supplied fact and compiler default, without parsing or loading a model. */
export function explain(intent: ScheduleIntent): string {
  validate(intent);
  const { unit, interval } = intent.frequency;
  const parts = [`The recurrence interval is ${interval} ${unit}${interval === 1 ? "" : "s"}.`];

  const time = intent.time;
  const timeDefault = time === undefined ? " by default" : "";

  if (unit === "hour") {
    parts.push(`The selected minute within each hour is ${time?.minute ?? 0}${timeDefault}.`);
  } else if (unit !== "minute") {
    const clock = `${padded(time?.hour ?? 0)}:${padded(time?.minute ?? 0)}`;

    parts.push(`The local time is ${clock}${timeDefault}.`);
  }

  if (intent.weekdays !== undefined) {
    parts.push(`The selected weekdays are ${intent.weekdays.join(", ")}.`);
  } else if (unit === "week") {
    parts.push(
      "No weekday is supplied. Cron compilation needs selected weekdays, while recurrence " +
        "previews use the anchor's weekday and default the anchor to the reference instant.",
    );
  }

  if (intent.dayOfMonth !== undefined) {
    parts.push(`The selected day of the month is ${intent.dayOfMonth}.`);
  } else if (intent.ordinalWeekday !== undefined) {
    const { ordinal, weekday } = intent.ordinalWeekday;

    parts.push(`The selected day is the ${ordinal} ${weekday} of the month.`);
  } else if (intent.day !== undefined) {
    parts.push(
      intent.day.type === "last_business_day"
        ? "The selected day is the last Monday to Friday day of the month, " +
            "without holiday exclusions."
        : "The selected day is the last day of the month.",
    );
  } else if (unit === "month") {
    parts.push(
      "No day of the month is supplied. Cron compilation needs a selected day, " +
        "while recurrence previews use the anchor's day of the month " +
        "and default the anchor to the reference instant.",
    );
  }

  if (intent.bounds?.start !== undefined) {
    parts.push(`The inclusive local start date is ${intent.bounds.start}.`);
  }

  if (intent.bounds?.end !== undefined) {
    parts.push(`The exclusive local end date is ${intent.bounds.end}.`);
  }

  if (intent.exceptions?.dates !== undefined && intent.exceptions.dates.length > 0) {
    parts.push(`The excluded local dates are ${intent.exceptions.dates.join(", ")}.`);
  }

  if (intent.exceptions?.weekdays !== undefined) {
    parts.push(`The excluded weekdays are ${intent.exceptions.weekdays.join(", ")}.`);
  }

  parts.push("The caller supplies the timezone and any recurrence reference instant.");

  parts.push(
    "Recurrence intervals follow local calendar fields, " +
      "so elapsed gaps can change across timezone transitions.",
  );

  const result = compile(intent);

  parts.push(
    result.supported
      ? `The cron expression is ${result.expression}.`
      : `Cron compilation is unsupported because ${result.reason}.`,
  );

  return parts.join(" ");
}
