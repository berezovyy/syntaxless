/**
 * Oracle decode: gold spans → LabeledToken[] → shipped `decodeIntent`.
 *
 * This module does not reimplement the decoder. It only builds the labeled
 * tokens the runtime already consumes (roles, clauseStart → boundary,
 * actionable, score 1) and asks `decodeIntent` whether they reconstruct gold.
 */
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";

import { isScheduleLabel, labelId } from "../../src/model/labels.js";
import type { Label } from "../../src/model/labels.js";
import type { SpeechAct } from "../../src/model/speech-acts.js";
import { tokenize } from "../../src/model/tokenizer.js";
import { decodeIntent } from "../../src/parser/decode.js";
import type { LabeledToken } from "../../src/parser/decode.js";
import type { ScheduleIntent } from "../../src/types/intent";
import { canonicalizeIntent } from "./canonical-intent";
import { iterateKindOracleSeeds } from "./kinds";
import type { KindExample } from "./kinds";
import type { RenderedExample } from "./renderer";
import { defaultActionable, Sentence } from "./sentence";
import { wrapsForKernel } from "./slots/sample";

export interface OracleSpan {
  start: number;
  end: number;
  label: Label;
  clauseStart: boolean;
  actionable?: boolean;
}

export interface OracleRecord {
  id: string;
  text: string;
  spans: readonly OracleSpan[];
  intent: ScheduleIntent | null;
  speechAct?: SpeechAct;
  kind?: string;
  template?: string;
  family?: string;
  source: "gold" | "kinds-sample" | "kind-seed";
}

export type OracleOutcome = { ok: true; intent: ScheduleIntent } | { ok: false; error: string };

export interface OracleFailure {
  id: string;
  kind: string;
  template: string;
  family?: string;
  source: OracleRecord["source"];
  text: string;
  expected: ScheduleIntent | null;
  actual: ScheduleIntent | { ok: false; error: string };
}

export interface OracleReport {
  total: number;
  correct: number;
  /** Complete miss count; `failures` is a per-group sample. */
  failureCount: number;
  failures: OracleFailure[];
  byKindTemplate: Record<string, { total: number; failures: number; sampled: number }>;
  sources: Record<OracleRecord["source"], { total: number; failures: number }>;
}

const KIND_SEED_COUNT = 50;

/** Map gold spans onto tokenizer tokens the way featurize does, then label. */
export function labeledTokensFromGold(
  text: string,
  spans: readonly OracleSpan[],
  score = 1,
): LabeledToken[] {
  const tokens = tokenize(text);
  const labeled: LabeledToken[] = [];
  let spanIndex = 0;

  for (const token of tokens) {
    if (token.kind === 3) continue;

    while (spanIndex < spans.length && spans[spanIndex]!.end <= token.start) {
      spanIndex += 1;
    }

    const span = spans[spanIndex];

    if (span === undefined || token.start < span.start || token.end > span.end) {
      throw new Error(`unaligned token: ${JSON.stringify(token.text)}`);
    }

    labeled.push({
      token,
      role: labelId[span.label],
      boundary: span.clauseStart && token.start === span.start,
      score,
      boundaryScore: score,
      actionable: span.actionable ?? defaultActionable(span.label),
    });
  }

  return labeled;
}

function speechActFor(record: OracleRecord): SpeechAct {
  if (record.speechAct !== undefined) return record.speechAct;

  return record.intent === null ? "none" : "request";
}

/** Run shipped `decodeIntent` on gold-span labels. */
export function oracleDecode(record: OracleRecord): OracleOutcome {
  try {
    const decoded = decodeIntent(
      labeledTokensFromGold(record.text, record.spans),
      undefined,
      record.text,
      speechActFor(record),
    );

    return { ok: true, intent: decoded.intent };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}

const SPEECH_ACT_REFUSE = new Set<SpeechAct>([
  "descriptive",
  "question",
  "quoted",
  "one-off",
  "cancel",
  "incomplete",
  "none",
]);

export function oracleAgrees(
  outcome: OracleOutcome,
  expected: ScheduleIntent | null,
  speechAct?: SpeechAct,
): boolean {
  if (speechAct !== undefined && SPEECH_ACT_REFUSE.has(speechAct)) return !outcome.ok;
  if (expected === null) return !outcome.ok;

  return (
    outcome.ok &&
    isDeepStrictEqual(canonicalizeIntent(outcome.intent), canonicalizeIntent(expected))
  );
}

export function kindTemplateKey(
  record: Pick<OracleRecord, "kind" | "template" | "id" | "family" | "source">,
): {
  kind: string;
  template: string;
} {
  if (record.source === "gold" && record.kind === undefined) {
    const family = record.family ?? "gold";

    return { kind: family, template: record.template ?? family };
  }

  const template = record.template ?? record.id;

  const kind =
    record.kind ??
    (typeof template === "string" && template.includes("/")
      ? template.slice(0, template.indexOf("/"))
      : (record.family ?? "unknown"));

  return { kind, template };
}

export function emptyOracleReport(): OracleReport {
  return {
    total: 0,
    correct: 0,
    failureCount: 0,
    failures: [],
    byKindTemplate: {},
    sources: {
      gold: { total: 0, failures: 0 },
      "kinds-sample": { total: 0, failures: 0 },
      "kind-seed": { total: 0, failures: 0 },
    },
  };
}

const MAX_FAILURES_PER_GROUP = 3;

export function recordOracleOutcome(report: OracleReport, record: OracleRecord): boolean {
  const outcome = oracleDecode(record);
  const ok = oracleAgrees(outcome, record.intent, speechActFor(record));
  const { kind, template } = kindTemplateKey(record);
  const groupKey = `${kind}\t${template}`;

  const group = (report.byKindTemplate[groupKey] ??= {
    total: 0,
    failures: 0,
    sampled: 0,
  });

  group.total += 1;
  report.total += 1;
  report.sources[record.source].total += 1;

  if (ok) {
    report.correct += 1;

    return true;
  }

  group.failures += 1;
  report.failureCount += 1;
  report.sources[record.source].failures += 1;

  if (group.sampled < MAX_FAILURES_PER_GROUP) {
    group.sampled += 1;

    report.failures.push({
      id: record.id,
      kind,
      template,
      ...(record.family === undefined ? {} : { family: record.family }),
      source: record.source,
      text: record.text,
      expected: record.intent,
      actual: outcome.ok ? outcome.intent : { ok: false, error: outcome.error },
    });
  }

  return false;
}

function asOracleRecord(
  value: {
    id: string;
    text: string;
    spans: readonly OracleSpan[];
    intent: ScheduleIntent | null;
    speechAct?: SpeechAct;
    template?: string;
    family?: string;
    kind?: string;
  },
  source: OracleRecord["source"],
): OracleRecord {
  return {
    id: value.id,
    text: value.text,
    spans: value.spans,
    intent: value.intent,
    ...(value.speechAct === undefined ? {} : { speechAct: value.speechAct }),
    ...(value.kind === undefined ? {} : { kind: value.kind }),
    ...(value.template === undefined ? {} : { template: value.template }),
    ...(value.family === undefined ? {} : { family: value.family }),
    source,
  };
}

function kindExampleToRecord(example: KindExample, index: number): OracleRecord {
  return {
    id: `${example.kind}/${example.template}/${index}`,
    text: example.example.text,
    spans: example.example.spans,
    intent: example.intent,
    speechAct: example.example.speechAct,
    kind: example.kind,
    template: example.template,
    family: example.family,
    source: "kind-seed",
  };
}

export async function loadGoldOracleRecords(): Promise<OracleRecord[]> {
  const dir = fileURLToPath(new URL("../data/gold/", import.meta.url));
  const names = (await readdir(dir)).filter((name) => name.endsWith(".jsonl")).sort();
  const records: OracleRecord[] = [];

  for (const name of names) {
    const contents = await readFile(join(dir, name), "utf8");

    for (const line of contents.split("\n")) {
      if (line.trim().length === 0) continue;

      const value = JSON.parse(line) as {
        id: string;
        text: string;
        spans: OracleSpan[];
        intent: ScheduleIntent | null;
        speechAct?: SpeechAct;
        family?: string;
        template?: string;
      };

      records.push(asOracleRecord(value, "gold"));
    }
  }

  return records;
}

function sentenceFromRendered(example: RenderedExample): Sentence {
  const out = new Sentence();
  const text = example.text;

  example.spans.forEach((span, index) => {
    const previousEnd = index === 0 ? 0 : example.spans[index - 1]!.end;
    const gap = text.slice(previousEnd, span.start);
    const fragment = text.slice(span.start, span.end);

    if (span.clauseStart) out.clause();
    out.add(fragment, span.label, index === 0 ? "" : gap, span.actionable);
  });

  return out;
}

/** Composed slot wraps over kind-seed holes; request/correction must oracle-match. */
export function loadSlottedOracleRecords(limit = 400): OracleRecord[] {
  const records: OracleRecord[] = [];

  for (const example of iterateKindOracleSeeds(2)) {
    if (example.intent === null) continue;
    if (!example.example.spans.some((span) => isScheduleLabel(span.label))) continue;
    const hole = sentenceFromRendered(example.example);
    const wraps = wrapsForKernel(hole, ["oracle", example.kind, example.template], 2);

    for (const wrap of wraps) {
      records.push({
        id: `slot/${example.kind}/${records.length}`,
        text: wrap.sentence.text,
        spans: wrap.sentence.spans,
        intent: example.intent,
        speechAct: wrap.speechAct,
        kind: example.kind,
        template: `slot/${wrap.draw.speech.id}`,
        family: example.family,
        source: "kind-seed",
      });

      if (records.length >= limit) return records;
    }
  }

  return records;
}

/**
 * Score every oracle record into counters and a capped failure sample. The
 * inputs are fixed by checked-in gold files and the kind catalog, not by the
 * training corpus; kind seeds are scored as they are generated.
 */
export async function runOracleSuite(): Promise<OracleReport> {
  const report = emptyOracleReport();

  for (const record of await loadGoldOracleRecords()) recordOracleOutcome(report, record);
  let index = 0;

  for (const example of iterateKindOracleSeeds(KIND_SEED_COUNT)) {
    recordOracleOutcome(report, kindExampleToRecord(example, index++));
  }

  for (const record of loadSlottedOracleRecords()) recordOracleOutcome(report, record);

  return report;
}

export interface OracleRedSummary {
  total: number;
  correct: number;
  failureCount: number;
  sources: OracleReport["sources"];
  byKind: Record<string, { total: number; failures: number }>;
  byKindTemplate: Record<string, { kind: string; template: string; count: number }>;
  examples: OracleFailure[];
}

export function summarizeOracleReport(report: OracleReport): OracleRedSummary {
  const byKind: Record<string, { total: number; failures: number }> = {};

  for (const [key, stats] of Object.entries(report.byKindTemplate)) {
    const tab = key.indexOf("\t");
    const kind = tab === -1 ? key : key.slice(0, tab);
    const bucket = (byKind[kind] ??= { total: 0, failures: 0 });

    bucket.total += stats.total;
    bucket.failures += stats.failures;
  }

  const grouped = groupedOracleFailures(report);

  return {
    total: report.total,
    correct: report.correct,
    failureCount: report.failureCount,
    sources: report.sources,
    byKind,
    byKindTemplate: Object.fromEntries(
      Object.entries(grouped).map(([key, value]) => [
        key,
        { kind: value.kind, template: value.template, count: value.count },
      ]),
    ),
    examples: Object.values(grouped).flatMap((value) => value.examples),
  };
}

export function groupedOracleFailures(
  report: OracleReport,
): Record<string, { kind: string; template: string; count: number; examples: OracleFailure[] }> {
  const grouped: Record<
    string,
    { kind: string; template: string; count: number; examples: OracleFailure[] }
  > = {};

  for (const [key, stats] of Object.entries(report.byKindTemplate)) {
    if (stats.failures === 0) continue;
    const tab = key.indexOf("\t");
    const kind = tab === -1 ? key : key.slice(0, tab);
    const template = tab === -1 ? key : key.slice(tab + 1);

    grouped[key] = {
      kind,
      template,
      count: stats.failures,
      examples: [],
    };
  }

  for (const failure of report.failures) {
    const key = `${failure.kind}\t${failure.template}`;
    const entry = grouped[key];

    if (entry !== undefined && entry.examples.length < 3) entry.examples.push(failure);
  }

  return grouped;
}
