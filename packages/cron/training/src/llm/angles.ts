/**
 * Angle registry: the curated ways the LLM is asked to produce schedule text.
 *
 * An angle describes how to ask, never what to believe. It fixes the Phase 06
 * family, the output kind (`schedule` or `negative`), and seven surface
 * dimensions. The registry is deliberately a curated subset of the dimension
 * product: not every tuple is legal, and the exclusions are documented here
 * rather than encoded as a predicate the caller can silently widen.
 *
 * Legality rules:
 * - `output: "negative"` requires `family: "negative"` and vice versa. A
 *   negative angle has no intent; every other family produces one.
 * - `difficulty: "bounds"` requires `family: "bounds"` and `difficulty:
 *   "exception"` requires `family: "exception"`. The other difficulties cover
 *   the remaining families: `single` for atom families, `compound` for the
 *   multi-selector families.
 * - No two angles share the same dimension tuple. `coverage.ts` expands the
 *   product and keeps exactly the tuples registered here, so a duplicate would
 *   collapse two cells into one.
 */
import type { Family } from "../sampler";

export const ANGLE_DIMENSIONS = {
  output: ["schedule", "negative"],
  register: ["terse", "spoken", "polite", "formal", "chat"],
  domain: ["reminders", "standup", "ops", "calendar", "travel", "chores"],
  numberStyle: ["digits", "words", "mixed"],
  timeframe: ["absolute", "relative"],
  noise: ["none", "casing", "typos", "punctuation", "filler"],
  difficulty: ["single", "compound", "bounds", "exception"],
} as const;

export type AngleOutput = (typeof ANGLE_DIMENSIONS.output)[number];
export type Register = (typeof ANGLE_DIMENSIONS.register)[number];
export type Domain = (typeof ANGLE_DIMENSIONS.domain)[number];
export type NumberStyle = (typeof ANGLE_DIMENSIONS.numberStyle)[number];
export type Timeframe = (typeof ANGLE_DIMENSIONS.timeframe)[number];
export type Noise = (typeof ANGLE_DIMENSIONS.noise)[number];
export type Difficulty = (typeof ANGLE_DIMENSIONS.difficulty)[number];

/** The seven coverage dimensions, without the family or provenance fields. */
export interface AngleDimensions {
  output: AngleOutput;
  register: Register;
  domain: Domain;
  numberStyle: NumberStyle;
  timeframe: Timeframe;
  noise: Noise;
  difficulty: Difficulty;
}

export type Angle = AngleDimensions & {
  /** Stable id, e.g. `"ops/weekday-list/terse"`. Never rename once used. */
  id: string;
  /** Phase 06 corpus family the sampler draws from. */
  family: Family;
  promptId: string;
  promptVersion: number;
};

/** Curated legal combinations. One entry per coverage cell. */
export const ANGLES: readonly Angle[] = [
  {
    id: "ops/weekday-list/terse",
    family: "weekday-list",
    output: "schedule",
    register: "terse",
    domain: "ops",
    numberStyle: "digits",
    timeframe: "absolute",
    noise: "none",
    difficulty: "compound",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "ops/weekday-list/chat",
    family: "weekday-list",
    output: "schedule",
    register: "chat",
    domain: "ops",
    numberStyle: "mixed",
    timeframe: "relative",
    noise: "casing",
    difficulty: "compound",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "standup/weekday-list/polite",
    family: "weekday-list",
    output: "schedule",
    register: "polite",
    domain: "standup",
    numberStyle: "words",
    timeframe: "absolute",
    noise: "none",
    difficulty: "compound",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "reminders/time-of-day/spoken",
    family: "time-of-day",
    output: "schedule",
    register: "spoken",
    domain: "reminders",
    numberStyle: "words",
    timeframe: "relative",
    noise: "none",
    difficulty: "single",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "reminders/time-of-day/chat",
    family: "time-of-day",
    output: "schedule",
    register: "chat",
    domain: "reminders",
    numberStyle: "digits",
    timeframe: "relative",
    noise: "typos",
    difficulty: "single",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "calendar/time-of-day/formal",
    family: "time-of-day",
    output: "schedule",
    register: "formal",
    domain: "calendar",
    numberStyle: "digits",
    timeframe: "absolute",
    noise: "punctuation",
    difficulty: "single",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "chores/interval/spoken",
    family: "interval",
    output: "schedule",
    register: "spoken",
    domain: "chores",
    numberStyle: "words",
    timeframe: "relative",
    noise: "filler",
    difficulty: "single",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "travel/interval/polite",
    family: "interval",
    output: "schedule",
    register: "polite",
    domain: "travel",
    numberStyle: "digits",
    timeframe: "absolute",
    noise: "none",
    difficulty: "single",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "ops/interval/terse",
    family: "interval",
    output: "schedule",
    register: "terse",
    domain: "ops",
    numberStyle: "digits",
    timeframe: "absolute",
    noise: "none",
    difficulty: "single",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "calendar/day-of-month/formal",
    family: "day-of-month",
    output: "schedule",
    register: "formal",
    domain: "calendar",
    numberStyle: "digits",
    timeframe: "absolute",
    noise: "none",
    difficulty: "compound",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "reminders/day-of-month/polite",
    family: "day-of-month",
    output: "schedule",
    register: "polite",
    domain: "reminders",
    numberStyle: "mixed",
    timeframe: "relative",
    noise: "punctuation",
    difficulty: "compound",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "ops/ordinal-weekday/terse",
    family: "ordinal-weekday",
    output: "schedule",
    register: "terse",
    domain: "ops",
    numberStyle: "digits",
    timeframe: "relative",
    noise: "none",
    difficulty: "compound",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "standup/ordinal-weekday/chat",
    family: "ordinal-weekday",
    output: "schedule",
    register: "chat",
    domain: "standup",
    numberStyle: "words",
    timeframe: "relative",
    noise: "typos",
    difficulty: "compound",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "calendar/weekday-range/formal",
    family: "weekday-range",
    output: "schedule",
    register: "formal",
    domain: "calendar",
    numberStyle: "digits",
    timeframe: "absolute",
    noise: "casing",
    difficulty: "compound",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "chores/weekday-range/polite",
    family: "weekday-range",
    output: "schedule",
    register: "polite",
    domain: "chores",
    numberStyle: "mixed",
    timeframe: "relative",
    noise: "filler",
    difficulty: "compound",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "ops/bounds/terse",
    family: "bounds",
    output: "schedule",
    register: "terse",
    domain: "ops",
    numberStyle: "digits",
    timeframe: "absolute",
    noise: "none",
    difficulty: "bounds",
    promptId: "schedule-complex",
    promptVersion: 1,
  },
  {
    id: "travel/bounds/formal",
    family: "bounds",
    output: "schedule",
    register: "formal",
    domain: "travel",
    numberStyle: "digits",
    timeframe: "absolute",
    noise: "punctuation",
    difficulty: "bounds",
    promptId: "schedule-complex",
    promptVersion: 1,
  },
  {
    id: "reminders/exception/polite",
    family: "exception",
    output: "schedule",
    register: "polite",
    domain: "reminders",
    numberStyle: "mixed",
    timeframe: "relative",
    noise: "none",
    difficulty: "exception",
    promptId: "schedule-complex",
    promptVersion: 1,
  },
  {
    id: "standup/exception/chat",
    family: "exception",
    output: "schedule",
    register: "chat",
    domain: "standup",
    numberStyle: "words",
    timeframe: "relative",
    noise: "typos",
    difficulty: "exception",
    promptId: "schedule-complex",
    promptVersion: 1,
  },
  {
    id: "chores/terse/spoken",
    family: "terse",
    output: "schedule",
    register: "spoken",
    domain: "chores",
    numberStyle: "words",
    timeframe: "relative",
    noise: "none",
    difficulty: "single",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "calendar/terse/polite",
    family: "terse",
    output: "schedule",
    register: "polite",
    domain: "calendar",
    numberStyle: "digits",
    timeframe: "absolute",
    noise: "none",
    difficulty: "compound",
    promptId: "schedule-basic",
    promptVersion: 1,
  },
  {
    id: "chores/negative/chat",
    family: "negative",
    output: "negative",
    register: "chat",
    domain: "chores",
    numberStyle: "mixed",
    timeframe: "relative",
    noise: "filler",
    difficulty: "single",
    promptId: "negative",
    promptVersion: 1,
  },
  {
    id: "travel/negative/polite",
    family: "negative",
    output: "negative",
    register: "polite",
    domain: "travel",
    numberStyle: "words",
    timeframe: "absolute",
    noise: "none",
    difficulty: "single",
    promptId: "negative",
    promptVersion: 1,
  },
  {
    id: "ops/negative/terse",
    family: "negative",
    output: "negative",
    register: "terse",
    domain: "ops",
    numberStyle: "digits",
    timeframe: "absolute",
    noise: "punctuation",
    difficulty: "single",
    promptId: "negative",
    promptVersion: 1,
  },
  {
    id: "reminders/negative/formal",
    family: "negative",
    output: "negative",
    register: "formal",
    domain: "reminders",
    numberStyle: "digits",
    timeframe: "absolute",
    noise: "casing",
    difficulty: "single",
    promptId: "negative",
    promptVersion: 1,
  },
];
