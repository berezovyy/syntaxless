/**
 * Oracle decode: gold spans → LabeledToken[] → shipped `decodeIntent`.
 *
 * This module does not reimplement the decoder. It only builds the labeled
 * tokens the runtime already consumes (roles, clauseStart → boundary,
 * actionable, score 1) and asks `decodeIntent` whether they reconstruct gold.
 */
import { createReadStream } from "node:fs";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";

import { labelId } from "../../src/model/labels.js";
import type { Label } from "../../src/model/labels.js";
import type { SpeechAct } from "../../src/model/speech-acts.js";
import { tokenize } from "../../src/model/tokenizer.js";
import { decodeIntent } from "../../src/parser/decode.js";
import type { LabeledToken } from "../../src/parser/decode.js";
import type { ScheduleIntent } from "../../src/types/intent";
import { canonicalizeIntent } from "./canonical-intent";
import { iterateKindOracleSeeds } from "./kinds";
import type { KindExample } from "./kinds";
import { createRng } from "./rng";
import { defaultActionable } from "./sentence";

function repoRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
}

function resolveRepo(path: string): string {
  return isAbsolute(path) ? path : resolve(repoRoot(), path);
}

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
const KINDS_SAMPLE = 200_000;
const KINDS_SAMPLE_SEED = 20260915;

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
      "off",
      speechActFor(record),
    );

    return { ok: true, intent: decoded.intent };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}

export function oracleAgrees(outcome: OracleOutcome, expected: ScheduleIntent | null): boolean {
  if (expected === null) return !outcome.ok;

  return (
    outcome.ok &&
    isDeepStrictEqual(canonicalizeIntent(outcome.intent), canonicalizeIntent(expected))
  );
}

export function oracleMatches(record: OracleRecord): boolean {
  return oracleAgrees(oracleDecode(record), record.intent);
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
  const ok = oracleAgrees(outcome, record.intent);
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

/** Uniform reservoir sample of non-empty JSONL lines. Does not slurp the file. */
export async function reservoirSampleJsonl(
  path: string,
  k: number,
  seed = KINDS_SAMPLE_SEED,
): Promise<string[]> {
  const chosen = await reservoirSampleLineNumbers(path, k, seed);
  const sample: string[] = new Array(chosen.size);
  const wanted = new Map<number, number>();
  let slot = 0;

  for (const lineNumber of chosen) {
    wanted.set(lineNumber, slot);
    slot += 1;
  }

  const stream = createReadStream(path, { encoding: "utf8" });
  const rl = createInterface({ input: stream, crlfDelay: Infinity });
  let seen = 0;

  try {
    for await (const line of rl) {
      if (line.trim().length === 0) continue;
      seen += 1;
      const at = wanted.get(seen);

      if (at !== undefined) sample[at] = line;
    }
  } finally {
    rl.close();
    stream.destroy();
  }

  return sample.filter((line) => line !== undefined);
}

/** Reservoir of line numbers only so a 200k sample does not retain 200k JSON strings. */
export async function reservoirSampleLineNumbers(
  path: string,
  k: number,
  seed = KINDS_SAMPLE_SEED,
): Promise<Set<number>> {
  const rng = createRng(seed);
  const sample: number[] = [];
  const stream = createReadStream(path, { encoding: "utf8" });
  const rl = createInterface({ input: stream, crlfDelay: Infinity });
  let seen = 0;

  try {
    for await (const line of rl) {
      if (line.trim().length === 0) continue;
      seen += 1;

      if (sample.length < k) {
        sample.push(seen);
      } else {
        const index = rng.int(0, seen - 1);

        if (index < k) sample[index] = seen;
      }
    }
  } finally {
    rl.close();
    stream.destroy();
  }

  return new Set(sample);
}

export async function loadGoldOracleRecords(): Promise<OracleRecord[]> {
  const dir = resolveRepo("packages/cron/training/data/gold");
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

export async function loadKindsSampleOracleRecords(count = KINDS_SAMPLE): Promise<OracleRecord[]> {
  const path = resolveRepo("packages/cron/training/data/synth/kinds.jsonl");
  const lines = await reservoirSampleJsonl(path, count);

  return lines.map((line) => {
    const value = JSON.parse(line) as {
      id: string;
      text: string;
      spans: OracleSpan[];
      intent: ScheduleIntent | null;
      speechAct?: SpeechAct;
      family?: string;
      template?: string;
    };

    return asOracleRecord(value, "kinds-sample");
  });
}

/** Stream-sample kinds.jsonl and score in place so 200k rows are not retained. */
export async function scoreKindsJsonlSample(
  report: OracleReport,
  count = KINDS_SAMPLE,
  kindsJsonl = "packages/cron/training/data/synth/kinds.jsonl",
): Promise<void> {
  const path = resolveRepo(kindsJsonl);
  const chosen = await reservoirSampleLineNumbers(path, count);
  const stream = createReadStream(path, { encoding: "utf8" });
  const rl = createInterface({ input: stream, crlfDelay: Infinity });
  let seen = 0;

  try {
    for await (const line of rl) {
      if (line.trim().length === 0) continue;
      seen += 1;
      if (!chosen.has(seen)) continue;

      const value = JSON.parse(line) as {
        id: string;
        text: string;
        spans: OracleSpan[];
        intent: ScheduleIntent | null;
        speechAct?: SpeechAct;
        family?: string;
        template?: string;
      };

      recordOracleOutcome(report, asOracleRecord(value, "kinds-sample"));
    }
  } finally {
    rl.close();
    stream.destroy();
  }
}

export function loadKindSeedOracleRecords(seedsPerTemplate = KIND_SEED_COUNT): OracleRecord[] {
  const records: OracleRecord[] = [];
  let index = 0;

  for (const example of iterateKindOracleSeeds(seedsPerTemplate)) {
    records.push(kindExampleToRecord(example, index));
    index += 1;
  }

  return records;
}

export async function runOracleSuite(options?: {
  kindsSample?: number;
  seedsPerTemplate?: number;
  kindsJsonl?: string;
}): Promise<OracleReport> {
  const report = emptyOracleReport();

  for (const record of await loadGoldOracleRecords()) recordOracleOutcome(report, record);

  for (const record of loadKindSeedOracleRecords(options?.seedsPerTemplate ?? KIND_SEED_COUNT)) {
    recordOracleOutcome(report, record);
  }

  const kindsSample = options?.kindsSample ?? 0;

  if (kindsSample > 0) {
    await scoreKindsJsonlSample(
      report,
      kindsSample,
      options?.kindsJsonl ?? "packages/cron/training/data/synth/kinds.jsonl",
    );
  }

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

export interface OracleCliOptions {
  seeds: number;
  kindsSample: number;
  out: string;
  gate: boolean;
  kindsJsonl?: string;
}

export function parseOracleArgs(argv: readonly string[]): OracleCliOptions {
  const options: OracleCliOptions = {
    seeds: KIND_SEED_COUNT,
    kindsSample: 0,
    out: "packages/cron/training/results/oracle-red.json",
    gate: false,
  };

  let kindsSampleExplicit = false;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]!;

    if (arg === "--seeds") {
      const value = argv[++index];

      if (value === undefined) throw new RangeError("missing value for --seeds");
      options.seeds = Number(value);
    } else if (arg === "--kinds-sample") {
      const value = argv[++index];

      if (value === undefined) throw new RangeError("missing value for --kinds-sample");
      options.kindsSample = Number(value);
      kindsSampleExplicit = true;
    } else if (arg === "--out") {
      const value = argv[++index];

      if (value === undefined) throw new RangeError("missing value for --out");
      options.out = value;
    } else if (arg === "--gate") {
      options.gate = true;
    } else if (arg === "--kinds-jsonl") {
      const value = argv[++index];

      if (value === undefined) throw new RangeError("missing value for --kinds-jsonl");
      options.kindsJsonl = value;
    } else {
      throw new RangeError(`unknown flag: ${arg}`);
    }
  }

  if (options.gate && !kindsSampleExplicit) options.kindsSample = KINDS_SAMPLE;

  return options;
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

if (import.meta.main) {
  const args = parseOracleArgs(process.argv.slice(2));

  const report = await runOracleSuite({
    seedsPerTemplate: args.seeds,
    kindsSample: args.kindsSample,
    ...(args.kindsJsonl === undefined ? {} : { kindsJsonl: args.kindsJsonl }),
  });

  const summary = summarizeOracleReport(report);
  const out = resolveRepo(args.out);

  await mkdir(dirname(out), { recursive: true });
  await writeFile(out, `${JSON.stringify(summary, null, 2)}\n`);

  process.stdout.write(
    `oracle: ${report.correct}/${report.total} (failures ${report.failureCount})\n`,
  );

  for (const [kind, stats] of Object.entries(summary.byKind).sort()) {
    if (stats.failures === 0) continue;
    process.stdout.write(`  ${kind}: ${stats.failures}/${stats.total}\n`);
  }

  process.stdout.write(`report -> ${out}\n`);
  if (args.gate && report.failureCount > 0) process.exit(1);
}
