/**
 * Strict contracts for labeled corpus spans, schedule intents, and the
 * existing verified LLM source. Source ingestion performs no generation.
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

const minuteOfHourSchema = z.object({ minute: z.number().int() }).strict();

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

/** The schedule AST accepted by source ingestion. Unknown fields are rejected. */
export const scheduleIntentSchema = z
  .object({
    frequency: frequencySchema,
    time: z.union([timeOfDaySchema, minuteOfHourSchema]).optional(),
    weekdays: z.array(weekdaySchema).optional(),
    dayOfMonth: z.number().int().optional(),
    ordinalWeekday: ordinalWeekdaySchema.optional(),
    day: daySpecSchema.optional(),
    bounds: boundsSchema.optional(),
    exceptions: exceptionsSchema.optional(),
  })
  .strict();

/** One labeled span supplied by a corpus source. */
export const alignedSpanSchema = z
  .object({
    start: z.number().int(),
    end: z.number().int(),
    label: z.enum(LABELS),
    clauseStart: z.boolean(),
    value: z.union([z.string(), z.number()]).optional(),
    actionable: z.boolean().optional(),
  })
  .strict();

export type AlignedSpanRecord = z.infer<typeof alignedSpanSchema>;

/**
 * An existing verified source record with its original generation provenance.
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
