/**
 * Weighted sampler for structurally valid {@link ScheduleIntent} trees.
 *
 * The sampler is the ground truth for the corpus: the expected intent is the
 * value drawn here, never the runtime parser's interpretation. Each call draws
 * a whole intent from one named family using a deterministic {@link Rng}.
 *
 * Only fields the Phase 02 `ScheduleIntent` union declares are produced. The
 * compiler remains the authority on representability; {@link assertValid}
 * guards shape alone.
 */

import type {
  FrequencyUnit,
  Ordinal,
  OrdinalWeekday,
  RecurrenceBounds,
  ScheduleExceptions,
  ScheduleIntent,
  TimeOfDay,
  Weekday,
} from "../../src/index";
import type { Rng } from "./rng";

export type Family =
  | "interval"
  | "weekday-list"
  | "weekday-range"
  | "ordinal-weekday"
  | "day-of-month"
  | "time-of-day"
  | "bounds"
  | "exception"
  | "terse"
  | "negative";

/**
 * Hand-tuned family weights. Negatives carry the largest single weight so the
 * model sees enough ordinary prose; the manifest records a hash of the
 * generator, not this table (Phase 08).
 */
export const FAMILY_WEIGHTS = {
  "weekday-list": 20,
  "time-of-day": 14,
  interval: 12,
  "weekday-range": 10,
  "ordinal-weekday": 9,
  "day-of-month": 9,
  terse: 8,
  bounds: 6,
  exception: 6,
  negative: 28,
} as const satisfies Record<Family, number>;

export interface IntentSample {
  family: Family;
  /** Structural sub-variant within the family (e.g. interval unit). */
  variant: string;
  intent: ScheduleIntent | null;
}

const WEEKDAYS: readonly Weekday[] = [
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
];

const ORDINALS: readonly Ordinal[] = ["first", "second", "third", "fourth", "last"];

const INTERVAL_UNITS: readonly FrequencyUnit[] = ["minute", "hour", "day", "week", "month"];

const MINUTE_SLOTS: readonly number[] = [0, 15, 30, 45];

/**
 * Draw one intent from `family`. Returns `{ family, variant, intent }` where
 * `negative` yields `intent: null` (ordinary prose, rendered in Phase 07).
 * Throws if the drawn shape violates {@link assertValid}.
 */
export function sampleIntent(rng: Rng, family: Family): IntentSample {
  const draft = sampleFamily(rng, family);

  if (draft === null) return { family, variant: "prose", intent: null };
  assertValid(draft.intent);

  return { family, variant: draft.variant, intent: draft.intent };
}

/**
 * Shape guard, not a second compiler. Checks only invariants independent of
 * cron expressibility: weekday lists non-empty/distinct/known, clock ranges,
 * and the ordinal/weekday enums. Representability bounds (interval, day of
 * month) are deliberately left to the compiler.
 */
export function assertValid(intent: ScheduleIntent): void {
  if (intent.weekdays !== undefined) {
    if (intent.weekdays.length === 0) {
      throw new Error("assertValid: weekdays must be non-empty");
    }

    const seen = new Set<Weekday>();

    for (const weekday of intent.weekdays) {
      if (!WEEKDAYS.includes(weekday)) {
        throw new Error(`assertValid: unknown weekday ${String(weekday)}`);
      }

      if (seen.has(weekday)) {
        throw new Error(`assertValid: duplicate weekday ${weekday}`);
      }

      seen.add(weekday);
    }
  }

  if (intent.time !== undefined) {
    const { hour, minute } = intent.time;

    if (!Number.isInteger(hour) || hour < 0 || hour > 23) {
      throw new Error(`assertValid: hour out of range: ${String(hour)}`);
    }

    if (!Number.isInteger(minute) || minute < 0 || minute > 59) {
      throw new Error(`assertValid: minute out of range: ${String(minute)}`);
    }
  }

  if (intent.ordinalWeekday !== undefined) {
    const { ordinal, weekday } = intent.ordinalWeekday;

    if (!ORDINALS.includes(ordinal)) {
      throw new Error(`assertValid: unknown ordinal ${String(ordinal)}`);
    }

    if (!WEEKDAYS.includes(weekday)) {
      throw new Error(`assertValid: unknown ordinal weekday ${String(weekday)}`);
    }
  }
}

interface Draft {
  variant: string;
  intent: ScheduleIntent;
}

function sampleFamily(rng: Rng, family: Family): Draft | null {
  switch (family) {
    case "interval":
      return sampleInterval(rng);
    case "weekday-list":
      return sampleWeekdayList(rng);
    case "weekday-range":
      return sampleWeekdayRange(rng);
    case "ordinal-weekday":
      return sampleOrdinalWeekday(rng);
    case "day-of-month":
      return sampleDayOfMonth(rng);
    case "time-of-day":
      return sampleTimeOfDay(rng);
    case "bounds":
      return sampleBounds(rng);
    case "exception":
      return sampleException(rng);
    case "terse":
      return sampleTerse(rng);
    case "negative":
      return null;
  }
}

function sampleInterval(rng: Rng): Draft {
  const unit = rng.pick(INTERVAL_UNITS);
  const maxInterval = unit === "minute" ? 59 : unit === "hour" ? 23 : 6;
  const interval = rng.int(2, maxInterval);

  return { variant: unit, intent: { frequency: { unit, interval } } };
}

function sampleWeekdayList(rng: Rng): Draft {
  const weekdays = pickDistinctWeekdays(rng, rng.int(2, 4));

  return {
    variant: "list",
    intent: { frequency: { unit: "week", interval: 1 }, weekdays },
  };
}

function sampleWeekdayRange(rng: Rng): Draft {
  const weekdays = weekdayRun(rng, 2, 5);

  return {
    variant: "range",
    intent: { frequency: { unit: "week", interval: 1 }, weekdays },
  };
}

function sampleOrdinalWeekday(rng: Rng): Draft {
  const ordinalWeekday: OrdinalWeekday = {
    ordinal: rng.pick(ORDINALS),
    weekday: rng.pick(WEEKDAYS),
  };

  return {
    variant: ordinalWeekday.ordinal,
    intent: { frequency: { unit: "month", interval: 1 }, ordinalWeekday },
  };
}

function sampleDayOfMonth(rng: Rng): Draft {
  const dayOfMonth = rng.int(1, 28);

  return {
    variant: "day-of-month",
    intent: { frequency: { unit: "month", interval: 1 }, dayOfMonth },
  };
}

function sampleTimeOfDay(rng: Rng): Draft {
  const time: TimeOfDay = { hour: rng.int(0, 23), minute: rng.pick(MINUTE_SLOTS) };

  return {
    variant: "time",
    intent: { frequency: { unit: "day", interval: 1 }, time },
  };
}

type BoundsVariant = "start" | "end" | "start-end";

const BOUNDS_VARIANTS: readonly BoundsVariant[] = ["start", "end", "start-end"];

function sampleBounds(rng: Rng): Draft {
  const variant = rng.pick(BOUNDS_VARIANTS);
  const bounds: RecurrenceBounds = {};

  if (variant === "start" || variant === "start-end") bounds.start = dateInYear(rng, 2026);
  if (variant === "end" || variant === "start-end") bounds.end = dateInYear(rng, 2027);

  return {
    variant,
    intent: { frequency: { unit: "day", interval: 1 }, bounds },
  };
}

function sampleException(rng: Rng): Draft {
  const group = rng.bool();

  const exceptions: ScheduleExceptions = {
    weekdays: pickDistinctWeekdays(rng, group ? rng.int(2, 3) : 1),
  };

  const base: ScheduleIntent = rng.bool()
    ? { frequency: { unit: "day", interval: 1 } }
    : {
        frequency: { unit: "week", interval: 1 },
        weekdays: pickDistinctWeekdays(rng, rng.int(1, 2)),
      };

  return {
    variant: group ? "weekday-group" : "weekday",
    intent: { ...base, exceptions },
  };
}

function sampleTerse(rng: Rng): Draft {
  if (rng.bool()) {
    return {
      variant: "weekday",
      intent: {
        frequency: { unit: "week", interval: 1 },
        weekdays: [rng.pick(WEEKDAYS)],
      },
    };
  }

  return {
    variant: "range",
    intent: {
      frequency: { unit: "week", interval: 1 },
      weekdays: weekdayRun(rng, 2, 3),
    },
  };
}

function pickDistinctWeekdays(rng: Rng, count: number): Weekday[] {
  const pool = [...WEEKDAYS];
  const picked: Weekday[] = [];

  for (let i = 0; i < count; i++) {
    const index = rng.int(0, pool.length - 1);

    picked.push(pool[index]!);
    pool.splice(index, 1);
  }

  return picked;
}

function weekdayRun(rng: Rng, minLength: number, maxLength: number): Weekday[] {
  const length = rng.int(minLength, maxLength);
  const start = rng.int(0, WEEKDAYS.length - length);

  return WEEKDAYS.slice(start, start + length);
}

function dateInYear(rng: Rng, year: number): string {
  const month = rng.int(1, 12);
  const day = rng.int(1, 28);

  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}
