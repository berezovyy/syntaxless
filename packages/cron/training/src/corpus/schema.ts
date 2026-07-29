/** Strict contracts for labeled corpus spans and schedule intents. */
import { z } from "zod";

import { LABELS } from "../../../src/model/labels.js";

const WEEKDAYS = [
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
] as const;

const FREQUENCY_UNITS = ["minute", "hour", "day", "week", "month"] as const;

const ORDINALS = ["first", "second", "third", "fourth", "last"] as const;

const weekdaySchema = z.enum(WEEKDAYS);
const frequencyUnitSchema = z.enum(FREQUENCY_UNITS);
const ordinalSchema = z.enum(ORDINALS);

const timeOfDaySchema = z.object({ hour: z.number().int(), minute: z.number().int() }).strict();

const minuteOfHourSchema = z.object({ minute: z.number().int() }).strict();

const ordinalWeekdaySchema = z.object({ ordinal: ordinalSchema, weekday: weekdaySchema }).strict();

const daySpecSchema = z
  .object({ type: z.enum(["last_business_day", "last_day_of_month"]) })
  .strict();

const boundsSchema = z
  .object({ start: z.string().optional(), end: z.string().optional() })
  .strict();

const exceptionsSchema = z
  .object({
    dates: z.array(z.string()).optional(),
    weekdays: z.array(weekdaySchema).optional(),
  })
  .strict();

const frequencySchema = z
  .object({ unit: frequencyUnitSchema, interval: z.number().int() })
  .strict();

/** The schedule AST of a corpus record. Unknown fields are rejected. */
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

/** One labeled span of a corpus record. */
export const alignedSpanSchema = z
  .object({
    start: z.number().int(),
    end: z.number().int(),
    label: z.enum(LABELS),
    clauseStart: z.boolean(),
    actionable: z.boolean().optional(),
  })
  .strict();
