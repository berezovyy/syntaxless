/**
 * The LLM contract: what the model is allowed to say, and what the verifier
 * writes once it has checked it.
 *
 * A candidate carries only prose and an intent. There is deliberately no
 * `slots` field: the model proposes text, and the deterministic aligner owns
 * every token label. `strict()` makes that structural — a candidate that
 * carries a label hint fails `schema-invalid` instead of being silently
 * ignored. Nothing here imports the model, the provider, or the network.
 */
import { z } from "zod";

import { LABELS } from "../../../src/model/labels.js";

export const WEEKDAYS = [
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
] as const;

export const FREQUENCY_UNITS = ["minute", "hour", "day", "week", "month"] as const;

export const ORDINALS = ["first", "second", "third", "fourth", "last"] as const;

export const weekdaySchema = z.enum(WEEKDAYS);
export const frequencyUnitSchema = z.enum(FREQUENCY_UNITS);
export const ordinalSchema = z.enum(ORDINALS);

export const timeOfDaySchema = z
  .object({ hour: z.number().int(), minute: z.number().int() })
  .strict();

export const ordinalWeekdaySchema = z
  .object({ ordinal: ordinalSchema, weekday: weekdaySchema })
  .strict();

export const daySpecSchema = z
  .object({ type: z.enum(["last_business_day", "last_day_of_month"]) })
  .strict();

export const boundsSchema = z
  .object({ start: z.string().optional(), end: z.string().optional() })
  .strict();

export const exceptionsSchema = z
  .object({
    dates: z.array(z.string()).optional(),
    weekdays: z.array(weekdaySchema).optional(),
  })
  .strict();

export const frequencySchema = z
  .object({ unit: frequencyUnitSchema, interval: z.number().int() })
  .strict();

/** The schedule AST as the model must emit it. Unknown fields are rejected. */
export const scheduleIntentSchema = z
  .object({
    frequency: frequencySchema,
    time: timeOfDaySchema.optional(),
    weekdays: z.array(weekdaySchema).optional(),
    dayOfMonth: z.number().int().optional(),
    ordinalWeekday: ordinalWeekdaySchema.optional(),
    day: daySpecSchema.optional(),
    bounds: boundsSchema.optional(),
    exceptions: exceptionsSchema.optional(),
  })
  .strict();

/**
 * One untrusted model proposal. `intent` is null for negatives; `output`
 * mirrors the angle. No `slots`, no labels, no scores.
 */
export const candidateSchema = z
  .object({
    text: z.string(),
    intent: scheduleIntentSchema.nullable(),
    output: z.enum(["schedule", "negative"]),
  })
  .strict();

export type Candidate = z.infer<typeof candidateSchema>;

/** One deterministic span recovered by the aligner. */
export const alignedSpanSchema = z
  .object({
    start: z.number().int(),
    end: z.number().int(),
    label: z.enum(LABELS),
    clauseStart: z.boolean(),
    value: z.union([z.string(), z.number()]).optional(),
  })
  .strict();

export type AlignedSpanRecord = z.infer<typeof alignedSpanSchema>;

/**
 * A verified candidate, ready for the merge phase. Provenance is additive:
 * `source` plus the four runner fields the model never controls.
 */
export const acceptedRecordSchema = z
  .object({
    text: z.string(),
    intent: scheduleIntentSchema.nullable(),
    output: z.enum(["schedule", "negative"]),
    spans: z.array(alignedSpanSchema),
    fingerprint: z.string(),
    source: z.literal("llm"),
    angleId: z.string(),
    model: z.string(),
    promptHash: z.string(),
    runId: z.string(),
  })
  .strict();

export type AcceptedRecord = z.infer<typeof acceptedRecordSchema>;
