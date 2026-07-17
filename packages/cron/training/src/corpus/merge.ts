/**
 * Shared corpus contract and verified-source loading.
 * Production publication is owned by the single training workflow.
 */
import { readFileSync } from "node:fs";

import { z } from "zod";

import type { Label } from "../../../src/model/labels.js";
import { isScheduleLabel, isStructural } from "../../../src/model/labels.js";
import { SPEECH_ACTS } from "../../../src/model/speech-acts.js";
import type { SpeechAct } from "../../../src/model/speech-acts.js";
import type { ScheduleIntent } from "../../../src/types/intent.js";
import { sha256Hex } from "../fingerprint.js";
import { LLM_FAMILIES } from "../llm/families.js";
import { acceptedRecordSchema, alignedSpanSchema, scheduleIntentSchema } from "../llm/schema.js";
import type { AcceptedRecord } from "../llm/schema.js";
import type { Family } from "../sampler.js";
import { readLinesSync } from "./jsonl-io.js";
import { type CorpusProvenance, provenanceSchema } from "./provenance.js";

export const SPLITS = ["train", "validation", "heldout"] as const;
export type Split = (typeof SPLITS)[number];

/** The one family vocabulary, derived from the sampler's own weight table. */
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

export type CorpusSource = "rule" | "llm";

/** A labeled span; `value` is present only on aligned LLM content spans. */
export interface CorpusSpan {
  start: number;
  end: number;
  label: Label;
  clauseStart: boolean;
  value?: string | number;
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
  source: CorpusSource;
  angleId?: string;
  model?: string;
  promptHash?: string;
  runId?: string;
  provenance?: CorpusProvenance;
}

const splitSchema = z.enum(SPLITS);
const familySchema = z.enum(FAMILIES);
const sourceSchema = z.enum(["rule", "llm"]);

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
    source: sourceSchema,
    angleId: z.string().optional(),
    model: z.string().optional(),
    promptHash: z.string().optional(),
    runId: z.string().optional(),
    provenance: provenanceSchema.optional(),
  })
  .strict();

function describeIssues(error: z.ZodError): string {
  return error.issues
    .map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`)
    .join("; ");
}

export function readJsonl(path: string): unknown[] {
  const raw = readFileSync(path, "utf8");
  const values: unknown[] = [];

  for (const line of raw.split("\n")) {
    const trimmed = line.trim();

    if (trimmed.length === 0) continue;

    try {
      values.push(JSON.parse(trimmed));
    } catch {
      throw new Error(`${path}: unparseable JSON line: ${trimmed.slice(0, 80)}`);
    }
  }

  return values;
}

function resolvedSpeechAct(intent: ScheduleIntent | null, speechAct?: SpeechAct): SpeechAct {
  return speechAct ?? (intent !== null ? "request" : "none");
}

function resolvedActionable(label: Label, speechAct: SpeechAct, actionable?: boolean): boolean {
  if (actionable !== undefined) return actionable;
  if (speechAct !== "request") return false;

  return isScheduleLabel(label) && !isStructural(label);
}

function hydrateDiscourse(
  intent: ScheduleIntent | null,
  spans: readonly CorpusSpan[],
  speechAct?: SpeechAct,
): { speechAct: SpeechAct; spans: CorpusSpan[] } {
  const resolved = resolvedSpeechAct(intent, speechAct);

  return {
    speechAct: resolved,
    spans: spans.map((span) => ({
      ...span,
      actionable: resolvedActionable(span.label, resolved, span.actionable),
    })),
  };
}

/** Stable content id for an LLM record, independent of input order. */
function llmId(record: AcceptedRecord): string {
  return `llm-${sha256Hex(
    [record.runId, record.angleId, record.fingerprint, record.text].join("\u0000"),
  ).slice(0, 16)}`;
}

function llmCorpusRecord(path: string, value: unknown, index: number): CorpusRecord {
  const parsed = acceptedRecordSchema.safeParse(value);

  if (!parsed.success) {
    throw new Error(
      `${path} record ${index + 1}: invalid llm record: ${describeIssues(parsed.error)}`,
    );
  }

  const record = parsed.data;
  const family = LLM_FAMILIES.get(record.angleId);

  if (family === undefined) {
    throw new Error(`${path} record ${index + 1}: unknown angleId "${record.angleId}"`);
  }

  const intent = record.intent as ScheduleIntent | null;
  const discourse = hydrateDiscourse(intent, record.spans as CorpusSpan[]);

  return {
    id: llmId(record),
    split: "train" as Split,
    family,
    fingerprint: record.fingerprint,
    text: record.text,
    intent,
    speechAct: discourse.speechAct,
    spans: discourse.spans,
    source: "llm" as const,
    angleId: record.angleId,
    model: record.model,
    promptHash: record.promptHash,
    runId: record.runId,
  };
}

/**
 * LLM records arrive as verified accepted records, which carry provenance but
 * no `id`, `split`, or `family`. The family comes from the fixed source mapping; the
 * split defaults to `train`; production applies its global isolation checks.
 * Streams one line at a time: the corpus path uses this, never an array.
 */
export function* iterateLlmRecords(path: string): Generator<CorpusRecord> {
  let index = 0;

  for (const line of readLinesSync(path)) {
    const trimmed = line.trim();

    if (trimmed.length === 0) continue;
    let value: unknown;

    try {
      value = JSON.parse(trimmed);
    } catch {
      throw new Error(`${path}: unparseable JSON line: ${trimmed.slice(0, 80)}`);
    }

    yield llmCorpusRecord(path, value, index++);
  }
}

/** Array form for small fixtures and tools; corpus preparation streams instead. */
export function readLlmRecords(path: string): CorpusRecord[] {
  return Array.from(iterateLlmRecords(path));
}
