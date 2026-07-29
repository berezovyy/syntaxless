/** The corpus record contract shared by the generator, featurizer and scorers. */
import { z } from "zod";

import type { Label } from "../../../src/model/labels.js";
import { SPEECH_ACTS } from "../../../src/model/speech-acts.js";
import type { SpeechAct } from "../../../src/model/speech-acts.js";
import type { ScheduleIntent } from "../../../src/types/intent.js";
import type { Family } from "../family.js";
import { type CorpusProvenance, provenanceSchema } from "./provenance.js";
import { alignedSpanSchema, scheduleIntentSchema } from "./schema.js";

export const SPLITS = ["train", "validation", "heldout"] as const;
export type Split = (typeof SPLITS)[number];

/** The family vocabulary; must list every {@link Family}. */
export const FAMILIES = [
  "interval",
  "weekday-list",
  "weekday-range",
  "ordinal-weekday",
  "day-of-month",
  "time-of-day",
  "bounds",
  "exception",
  "terse",
  "negative",
] as const satisfies readonly Family[];

/** A labeled span. */
export interface CorpusSpan {
  start: number;
  end: number;
  label: Label;
  clauseStart: boolean;
  actionable?: boolean;
}

/** The merged record contract. Every record traces back to one producer. */
export interface CorpusRecord {
  id: string;
  split: Split;
  family: Family;
  fingerprint: string;
  text: string;
  intent: ScheduleIntent | null;
  speechAct?: SpeechAct;
  spans: CorpusSpan[];
  template?: string;
  source: "rule";
  provenance?: CorpusProvenance;
}

const splitSchema = z.enum(SPLITS);
const familySchema = z.enum(FAMILIES);

/** The strict corpus contract. Unknown fields are rejected, not ignored. */
export const corpusRecordSchema = z
  .object({
    id: z.string(),
    split: splitSchema,
    family: familySchema,
    fingerprint: z.string(),
    text: z.string(),
    intent: scheduleIntentSchema.nullable(),
    speechAct: z.enum(SPEECH_ACTS).optional(),
    spans: z.array(alignedSpanSchema),
    template: z.string().optional(),
    source: z.literal("rule"),
    provenance: provenanceSchema.optional(),
  })
  .strict();
