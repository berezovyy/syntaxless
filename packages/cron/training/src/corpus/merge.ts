/**
 * Corpus merge: one provenance-tagged corpus from the rule and LLM sources.
 *
 * The two producers are merged last, right before featurization, so each stays
 * independently testable. Merge does four things in a fixed order:
 *
 *   1. read and validate each source's JSONL against its strict schema;
 *   2. prove the held-out split is disjoint from train/validation — the leak
 *      gate, which runs before dedup so a planted duplicate cannot hide a leak;
 *   3. dedup in two passes (normalized text, then `(fingerprint, family)`),
 *      keeping the first record in `rule`, `llm`, ascending-id order;
 *   4. assign each record a split, a stable order, and report balance.
 *
 * Provenance is additive: an `angleId` or `runId` already on a record is never
 * overwritten. Nothing here inspects the network or a model; every LLM record
 * has already passed the Phase 11 verifier.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";

import { z } from "zod";

import type { Label } from "../../../src/model/labels.js";
import { isStructural } from "../../../src/model/labels.js";
import { SPEECH_ACTS } from "../../../src/model/speech-acts.js";
import type { SpeechAct } from "../../../src/model/speech-acts.js";
import { tokenize } from "../../../src/model/tokenizer.js";
import type { ScheduleIntent } from "../../../src/types/intent.js";
import { sha256Hex } from "../fingerprint.js";
import type { Split } from "../generate.js";
import { ANGLES } from "../llm/angles.js";
import { applyCoverage, gaps, planCoverage } from "../llm/coverage.js";
import type { CoverageReport } from "../llm/coverage.js";
import { acceptedRecordSchema, alignedSpanSchema, scheduleIntentSchema } from "../llm/schema.js";
import type { AcceptedRecord } from "../llm/schema.js";
import type { Family } from "../sampler.js";
import { normalizeForKey } from "./normalize.js";

export const SPLITS = ["train", "validation", "heldout"] as const;

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
  })
  .strict();

interface CandidateInput {
  source: CorpusSource;
  records: readonly CorpusRecord[];
}

export interface DedupSample {
  keptId: string;
  droppedId: string;
  key: string;
}

export interface DedupReport {
  exact: { dropped: number; samples: DedupSample[] };
  structural: { dropped: number; samples: DedupSample[] };
  totalDropped: number;
}

export interface BalanceReport {
  totals: { records: number; negatives: number; schedules: number };
  bySource: Record<CorpusSource, number>;
  bySplit: Record<Split, number>;
  byFamily: Record<string, number>;
  byAngleId: Record<string, number>;
  byLabel: Record<string, number>;
  negativeRatio: number;
  tokenLengthHistogram: Record<string, number>;
  coverage: CoverageReport;
  warnings: string[];
  rejectionActions: Readonly<Record<string, string>>;
}

export interface MergeResult {
  records: CorpusRecord[];
  dedup: DedupReport;
  balance: BalanceReport;
  heldoutFingerprints: string[];
}

/**
 * Rejection reasons the coverage report turns into concrete follow-up actions.
 * A rejection is data, not an error: the action names where the fix belongs.
 */
export const REJECTION_ACTIONS: Readonly<Record<string, string>> = {
  "unknown-lexeme":
    "add the surface to the runtime lexicon with a parser test before it is generated",
  "intent-text-mismatch": "a harder prompt variant or a negative twin",
  "negative-has-schedule": "an adversarial negative variant",
  unsupported: "a Phase 03/04 compiler gap, not a corpus proposal",
};

const SAMPLE_LIMIT = 20;

const SOURCE_RANK: Record<CorpusSource, number> = { rule: 0, llm: 1 };

const ANGLE_FAMILY = new Map(ANGLES.map((angle) => [angle.id, angle.family]));

const TOKEN_BUCKETS: readonly (readonly [string, number])[] = [
  ["0", 0],
  ["1-8", 8],
  ["9-16", 16],
  ["17-32", 32],
  ["33-64", 64],
  ["65-128", 128],
  ["129+", Number.POSITIVE_INFINITY],
];

function tokenBucket(count: number): string {
  for (const [label, max] of TOKEN_BUCKETS) {
    if (count <= max) return label;
  }

  return "129+";
}

function compare(left: string, right: string): number {
  if (left < right) return -1;
  if (left > right) return 1;

  return 0;
}

/**
 * An all-`O` example (every negative) fingerprints as the single `NEGATIVE`
 * sentinel, so it carries no structural frame. Such records are excluded from
 * the held-out fingerprint set: there is no frame to leak, and otherwise every
 * negative would be treated as a held-out collision.
 */
function isAllO(record: CorpusRecord): boolean {
  return !record.spans.some((span) => span.label !== "O");
}

/** The held-out fingerprints across every source. This is the split authority. */
export function heldoutFingerprintsOf(records: readonly CorpusRecord[]): string[] {
  return [
    ...new Set(
      records
        .filter((record) => record.split === "heldout" && !isAllO(record))
        .map((record) => record.fingerprint),
    ),
  ].sort();
}

/**
 * The leak gate. A train or validation record may never carry a held-out
 * fingerprint; if one does, the merge stops rather than emit a corpus that
 * silently inflates validation scores.
 */
export function assertDisjoint(
  records: readonly CorpusRecord[],
  heldout: ReadonlySet<string>,
): void {
  const offenders = records.filter(
    (record) =>
      (record.split === "train" || record.split === "validation") &&
      heldout.has(record.fingerprint),
  );

  if (offenders.length > 0) {
    const names = offenders.slice(0, 5).map((record) => record.id);

    throw new Error(
      `merge: ${offenders.length} train/validation record(s) carry a held-out ` +
        `fingerprint (${names.join(", ")}); split is a leak boundary`,
    );
  }
}

/** Deterministic input order for dedup: rule before llm, then ascending id. */
function dedupOrder(records: readonly CorpusRecord[]): CorpusRecord[] {
  return [...records].sort((left, right) => {
    const bySource = SOURCE_RANK[left.source] - SOURCE_RANK[right.source];

    if (bySource !== 0) return bySource;

    return compare(left.id, right.id);
  });
}

/** Stable output order so review diffs line up: `(split, fingerprint, id)`. */
function outputOrder(records: readonly CorpusRecord[]): CorpusRecord[] {
  return [...records].sort((left, right) => {
    const bySplit = compare(left.split, right.split);

    if (bySplit !== 0) return bySplit;
    const byFingerprint = compare(left.fingerprint, right.fingerprint);

    if (byFingerprint !== 0) return byFingerprint;

    return compare(left.id, right.id);
  });
}

/**
 * The structural dedup key.
 *
 * A fingerprint erases content values, so `(fingerprint, family)` alone would
 * collapse two genuinely different schedules that share a frame (`every Monday`
 * and `every Tuesday`), and every all-`O` negative shares the `NEGATIVE`
 * sentinel. The value signature keeps those apart: an intent-bearing record is
 * keyed by its intent, and a negative (no intent) by its normalized text.
 */
function structuralKey(record: CorpusRecord): string {
  const values =
    record.intent === null
      ? `text:${normalizeForKey(record.text)}`
      : `intent:${JSON.stringify(record.intent)}`;

  return `${record.fingerprint}\u0000${record.family}\u0000${values}`;
}

/**
 * Two-pass dedup. Exact duplicates collapse by {@link normalizeForKey}; the
 * survivors then collapse by frame, family, and value signature. Every drop is
 * counted and sampled, so no record is ever silently discarded.
 */
export function dedup(records: readonly CorpusRecord[]): {
  kept: CorpusRecord[];
  report: DedupReport;
} {
  const ordered = dedupOrder(records);
  const exactSeen = new Map<string, string>();
  const exactSamples: DedupSample[] = [];
  let exactDropped = 0;
  const exactSurvivors: CorpusRecord[] = [];

  for (const record of ordered) {
    const key = normalizeForKey(record.text);
    const keptId = exactSeen.get(key);

    if (keptId !== undefined) {
      exactDropped++;

      if (exactSamples.length < SAMPLE_LIMIT) {
        exactSamples.push({ keptId, droppedId: record.id, key });
      }

      continue;
    }

    exactSeen.set(key, record.id);
    exactSurvivors.push(record);
  }

  const structuralSeen = new Map<string, string>();
  const structuralSamples: DedupSample[] = [];
  let structuralDropped = 0;
  const kept: CorpusRecord[] = [];

  for (const record of exactSurvivors) {
    const key = structuralKey(record);
    const keptId = structuralSeen.get(key);

    if (keptId !== undefined) {
      structuralDropped++;

      if (structuralSamples.length < SAMPLE_LIMIT) {
        structuralSamples.push({ keptId, droppedId: record.id, key });
      }

      continue;
    }

    structuralSeen.set(key, record.id);
    kept.push(record);
  }

  return {
    kept,
    report: {
      exact: { dropped: exactDropped, samples: exactSamples },
      structural: { dropped: structuralDropped, samples: structuralSamples },
      totalDropped: exactDropped + structuralDropped,
    },
  };
}

/** Counts by `source`, split, family, angle, and span label, plus warnings. */
export function balance(records: readonly CorpusRecord[]): BalanceReport {
  const bySource: Record<CorpusSource, number> = { rule: 0, llm: 0 };
  const bySplit: Record<Split, number> = { train: 0, validation: 0, heldout: 0 };
  const byFamily: Record<string, number> = {};
  const byAngleId: Record<string, number> = {};
  const byLabel: Record<string, number> = {};
  const histogram: Record<string, number> = {};
  let negatives = 0;

  for (const record of records) {
    bySource[record.source]++;
    bySplit[record.split]++;
    byFamily[record.family] = (byFamily[record.family] ?? 0) + 1;

    if (record.angleId !== undefined) {
      byAngleId[record.angleId] = (byAngleId[record.angleId] ?? 0) + 1;
    }

    if (record.intent === null) negatives++;

    for (const span of record.spans) {
      byLabel[span.label] = (byLabel[span.label] ?? 0) + 1;
    }

    const bucket = tokenBucket(tokenize(record.text).length);

    histogram[bucket] = (histogram[bucket] ?? 0) + 1;
  }

  const plan = planCoverage();
  const coverage = applyCoverage(plan, byAngleId);
  const warnings: string[] = [];

  for (const cell of gaps(coverage)) {
    warnings.push(`cell ${cell.key} below target: ${cell.produced}/${cell.target}`);
  }

  const familyTargets = new Map<string, number>();

  for (const cell of plan.cells) {
    familyTargets.set(cell.family, (familyTargets.get(cell.family) ?? 0) + cell.target);
  }

  for (const [family, target] of [...familyTargets].sort(([a], [b]) => compare(a, b))) {
    const produced = byFamily[family] ?? 0;

    if (produced < target) {
      warnings.push(`family ${family} below target: ${produced}/${target}`);
    }
  }

  return {
    totals: {
      records: records.length,
      negatives,
      schedules: records.length - negatives,
    },
    bySource,
    bySplit,
    byFamily,
    byAngleId,
    byLabel,
    negativeRatio: records.length === 0 ? 0 : negatives / records.length,
    tokenLengthHistogram: histogram,
    coverage,
    warnings,
    rejectionActions: REJECTION_ACTIONS,
  };
}

/**
 * Merge sources into one deduped, split-checked corpus. Pure and deterministic:
 * the result depends on the set of input records, never on their order.
 */
export function mergeCorpus(sources: readonly CandidateInput[]): MergeResult {
  const all = sources.flatMap((source) => [...source.records]);
  const heldout = new Set(heldoutFingerprintsOf(all));

  assertDisjoint(all, heldout);
  const { kept, report } = dedup(all);
  const records = outputOrder(kept);

  return {
    records,
    dedup: report,
    balance: balance(records),
    heldoutFingerprints: [...heldout].sort(),
  };
}

// ---------------------------------------------------------------------------
// File I/O and normalization into the record contract
// ---------------------------------------------------------------------------

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

  return label !== "O" && !isStructural(label);
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

function fromRule(record: z.infer<typeof corpusRecordSchema>): CorpusRecord {
  const discourse = hydrateDiscourse(
    record.intent as ScheduleIntent | null,
    record.spans as CorpusSpan[],
    record.speechAct,
  );

  return {
    id: record.id,
    split: record.split,
    family: record.family,
    fingerprint: record.fingerprint,
    text: record.text,
    intent: record.intent as ScheduleIntent | null,
    speechAct: discourse.speechAct,
    spans: discourse.spans,
    source: "rule",
    ...(record.template === undefined ? {} : { template: record.template }),
    ...(record.angleId === undefined ? {} : { angleId: record.angleId }),
    ...(record.model === undefined ? {} : { model: record.model }),
    ...(record.promptHash === undefined ? {} : { promptHash: record.promptHash }),
    ...(record.runId === undefined ? {} : { runId: record.runId }),
  };
}

/** Stable content id for an LLM record, independent of input order. */
function llmId(record: AcceptedRecord): string {
  return `llm-${sha256Hex(
    [record.runId, record.angleId, record.fingerprint, record.text].join("\u0000"),
  ).slice(0, 16)}`;
}

export function readRuleRecords(path: string): CorpusRecord[] {
  return readJsonl(path).map((value, index) => {
    const parsed = corpusRecordSchema.safeParse(value);

    if (!parsed.success) {
      throw new Error(
        `${path} record ${index + 1}: invalid rule record: ${describeIssues(parsed.error)}`,
      );
    }

    if (parsed.data.source !== "rule") {
      throw new Error(
        `${path} record ${index + 1}: expected source "rule", got "${parsed.data.source}"`,
      );
    }

    return fromRule(parsed.data);
  });
}

/**
 * LLM records arrive as verified accepted records, which carry provenance but
 * no `id`, `split`, or `family`. The family comes from the angle registry; the
 * split defaults to `train` and the leak gate rejects any held-out collision.
 */
export function readLlmRecords(path: string): CorpusRecord[] {
  return readJsonl(path).map((value, index) => {
    const parsed = acceptedRecordSchema.safeParse(value);

    if (!parsed.success) {
      throw new Error(
        `${path} record ${index + 1}: invalid llm record: ${describeIssues(parsed.error)}`,
      );
    }

    const record = parsed.data;
    const family = ANGLE_FAMILY.get(record.angleId);

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
  });
}

export interface MergeInputInfo {
  role: CorpusSource;
  path: string;
  sha256: string;
  records: number;
}

export interface MergeManifest {
  inputs: MergeInputInfo[];
  inputRecords: number;
  mergedRecords: number;
  recordCounts: Record<CorpusSource, number>;
  dedup: DedupReport;
  splitHashes: Record<Split, string>;
  balance: {
    bySource: Record<CorpusSource, number>;
    bySplit: Record<Split, number>;
    negativeRatio: number;
    warnings: string[];
  };
  mergerSha256: string;
}

export interface MergeOptions {
  rule: string[];
  llm: string[];
  out: string;
}

export interface MergeSummary {
  out: string;
  records: number;
  dropped: number;
  warnings: number;
  files: {
    corpus: string;
    manifest: string;
    dedup: string;
    balance: string;
  };
  manifest: MergeManifest;
}

function fileSha256(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

/** Merger sources hashed into the manifest so drift is detectable. */
const MERGER_SOURCES: readonly string[] = ["normalize.ts", "merge.ts"];

export function mergerSha256(): string {
  const base = dirname(fileURLToPath(import.meta.url));
  const digest = createHash("sha256");

  for (const name of [...MERGER_SOURCES].sort()) {
    digest.update(name, "utf8");
    digest.update("\0", "utf8");
    digest.update(readFileSync(join(base, name)));
    digest.update("\0", "utf8");
  }

  return digest.digest("hex");
}

function splitHash(records: readonly CorpusRecord[], split: Split): string {
  const fingerprints = records
    .filter((record) => record.split === split)
    .map((record) => record.fingerprint)
    .sort();

  return sha256Hex(JSON.stringify(fingerprints));
}

function writeJson(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`);
}

/** Walk up to the workspace root (the package.json that lists `workspaces`). */
function repoRoot(): string {
  let dir = dirname(fileURLToPath(import.meta.url));

  for (;;) {
    const manifest = join(dir, "package.json");

    if (existsSync(manifest)) {
      try {
        const pkg = JSON.parse(readFileSync(manifest, "utf8")) as {
          workspaces?: unknown;
        };

        if (Array.isArray(pkg.workspaces)) return dir;
      } catch {
        // Not a readable manifest; keep walking up.
      }
    }

    const parent = dirname(dir);

    if (parent === dir) return process.cwd();
    dir = parent;
  }
}

/** Resolve a CLI path against the repo root so `--filter` cwd does not matter. */
function resolvePath(value: string): string {
  return isAbsolute(value) ? value : join(repoRoot(), value);
}

/** Read, merge, and write the corpus and its sidecar reports. */
export function mergeFiles(options: MergeOptions): MergeSummary {
  const rulePaths = options.rule.map(resolvePath);
  const llmPaths = options.llm.map(resolvePath);
  const out = resolvePath(options.out);

  const ruleRecords: CorpusRecord[] = [];
  const llmRecords: CorpusRecord[] = [];
  const inputs: MergeInputInfo[] = [];

  for (const path of rulePaths) {
    const records = readRuleRecords(path);

    ruleRecords.push(...records);

    inputs.push({
      role: "rule",
      path,
      sha256: fileSha256(path),
      records: records.length,
    });
  }

  for (const path of llmPaths) {
    const records = readLlmRecords(path);

    llmRecords.push(...records);

    inputs.push({
      role: "llm",
      path,
      sha256: fileSha256(path),
      records: records.length,
    });
  }

  const result = mergeCorpus([
    { source: "rule", records: ruleRecords },
    { source: "llm", records: llmRecords },
  ]);

  const corpusPath = `${out}.jsonl`;
  const manifestPath = `${out}.manifest.json`;
  const dedupPath = `${out}.dedup.json`;
  const balancePath = `${out}.balance.json`;

  mkdirSync(dirname(corpusPath), { recursive: true });
  const body = result.records.map((record) => JSON.stringify(record)).join("\n");

  writeFileSync(corpusPath, body.length > 0 ? `${body}\n` : "");
  writeJson(dedupPath, result.dedup);
  writeJson(balancePath, result.balance);

  const splitHashes: Record<Split, string> = {
    train: splitHash(result.records, "train"),
    validation: splitHash(result.records, "validation"),
    heldout: splitHash(result.records, "heldout"),
  };

  const manifest: MergeManifest = {
    inputs,
    inputRecords: ruleRecords.length + llmRecords.length,
    mergedRecords: result.records.length,
    recordCounts: result.balance.bySource,
    dedup: result.dedup,
    splitHashes,
    balance: {
      bySource: result.balance.bySource,
      bySplit: result.balance.bySplit,
      negativeRatio: result.balance.negativeRatio,
      warnings: result.balance.warnings,
    },
    mergerSha256: mergerSha256(),
  };

  writeJson(manifestPath, manifest);

  return {
    out,
    records: result.records.length,
    dropped: result.dedup.totalDropped,
    warnings: result.balance.warnings.length,
    files: {
      corpus: corpusPath,
      manifest: manifestPath,
      dedup: dedupPath,
      balance: balancePath,
    },
    manifest,
  };
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

function readOption(argv: readonly string[], index: number): string {
  const value = argv[index + 1];

  if (value === undefined) throw new RangeError(`missing value for ${argv[index]}`);

  return value;
}

/** Parse repeated `--rule`/`--llm` inputs and one `--out` prefix. */
export function parseArgs(argv: readonly string[]): MergeOptions {
  const rule: string[] = [];
  const llm: string[] = [];
  let out: string | undefined;

  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index]!;

    switch (arg) {
      case "--rule":
        rule.push(readOption(argv, index));
        index++;
        break;
      case "--llm":
        llm.push(readOption(argv, index));
        index++;
        break;
      case "--out":
        out = readOption(argv, index);
        index++;
        break;
      default:
        throw new RangeError(`unknown flag: ${arg}`);
    }
  }

  if (out === undefined || out.length === 0) {
    throw new RangeError("--out is required");
  }

  if (rule.length === 0 && llm.length === 0) {
    throw new RangeError("at least one --rule or --llm input is required");
  }

  return { rule, llm, out };
}

if (import.meta.main) {
  const options = parseArgs(process.argv.slice(2));
  const summary = mergeFiles(options);

  process.stdout.write(
    `merged: ${summary.records} records, ${summary.dropped} dropped, ` +
      `${summary.warnings} warnings\n` +
      `corpus   -> ${summary.files.corpus}\n` +
      `manifest -> ${summary.files.manifest}\n` +
      `dedup    -> ${summary.files.dedup}\n` +
      `balance  -> ${summary.files.balance}\n`,
  );
}
