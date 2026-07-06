/**
 * Candidate/baseline evaluation over a gold corpus.
 *
 * Every number this module reports comes from the *built* `packages/cron/dist/`
 * runtime, never from `src/`. The bundled `dist` is scored directly. When a
 * weight module path is supplied (a candidate artifact), the built tree is
 * copied to a scratch directory and only its `model/weights.gen.js` is
 * replaced, so the exact shipped engine runs on the candidate parameters in
 * this same process. The crate is what will ship, not a re-implementation.
 *
 * The corpus is gold data with rendered spans, an expected AST, and an expected
 * compiled expression. `tokenRoles` scores the built internal modules
 * (tokenizer + inference); `exactAst` and `compiledExpression` score the built
 * public entry (`parse` + `compile`). The three contracts are reported side by
 * side and never summed.
 *
 * Reports are write-only evidence: `results/<set>.json` is never read as an
 * input to another run. The calibration and held-out frame splits are
 * development data and may not gate promotion; every report says so in `scope`.
 */
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { isDeepStrictEqual } from "node:util";

import type { CompileResult } from "../../src/compiler/cron";
import type { RawToken } from "../../src/model/tokenizer";
import type { ParseResult } from "../../src/parser/parse";
import type { ScheduleIntent } from "../../src/types/intent";

/** Default corpus/report locations, relative to the repository root. */
export const DEFAULTS = {
  dist: "packages/cron/dist",
  corpus: "packages/cron/training/data/gold/reserved-carriers.jsonl",
  out: "packages/cron/training/results/reserved-carriers.json",
} as const;

/** Keep a few examples for every complete failure-count bucket. */
export const MAX_FAILURES_PER_CATEGORY = 3;

/** Repository root: this file lives at `packages/cron/training/src/`. */
export function repoRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
}

/** Resolve a CLI path against the repository root unless already absolute. */
export function resolveRepo(path: string): string {
  return isAbsolute(path) ? path : resolve(repoRoot(), path);
}

/** SHA-256 as lowercase hex. */
export function sha256Hex(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/** One gold record. `spans` are renderer-supplied labels, never parser output. */
export interface GoldRecord {
  id: string;
  family: string;
  text: string;
  spans: readonly {
    start: number;
    end: number;
    label: string;
    clauseStart: boolean;
  }[];
  intent: ScheduleIntent | null;
  expected: { supported: boolean; expression: string | null };
}

/** Running correct/total counts; the guard's `Counts` shape. */
export interface Counts {
  total: number;
  correct: number;
}

export type Metric = "tokenRoles" | "exactAst" | "compiledExpression";

export interface ContractFailure {
  id: string;
  family: string;
  detail: string;
  expected: unknown;
  actual: unknown;
}

/** One contract's report. Every report carries its own provenance fields. */
export interface ContractReport {
  metric: Metric;
  artifactSha256: string;
  corpus: string;
  corpusSha256: string;
  familyDistribution: Record<string, number>;
  scope: string;
  total: number;
  correct: number;
  accuracy: number;
  byFamily: Record<string, Counts>;
  /** Complete counts; unlike `failures`, these are never sample-capped. */
  failureCounts: Record<string, number>;
  failures: ContractFailure[];
}

export interface EvaluationResult {
  artifactSha256: string;
  corpus: string;
  corpusSha256: string;
  familyDistribution: Record<string, number>;
  scope: string;
  contracts: ContractReport[];
  selectivePrediction: RiskCoveragePoint[];
}

export interface SelectiveOutcome {
  accepted: boolean;
  correct: boolean;
  confidence: number;
  negative: boolean;
}

export interface RiskCoveragePoint {
  threshold: number;
  total: number;
  accepted: number;
  acceptedWrong: number;
  coverage: number;
  risk: number;
  silentMeaningChangeRate: number;
  negativeFalseAcceptanceRate: number;
}

/** Summarize confidence as operational risk rather than token probability. */
export function riskCoverage(
  outcomes: readonly SelectiveOutcome[],
  thresholds: readonly number[] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
): RiskCoveragePoint[] {
  const negativeTotal = outcomes.filter((outcome) => outcome.negative).length;

  return thresholds.map((threshold) => {
    const accepted = outcomes.filter(
      (outcome) => outcome.accepted && outcome.confidence >= threshold,
    );

    const acceptedWrong = accepted.filter((outcome) => !outcome.correct).length;
    const negativeFalseAccepts = accepted.filter((outcome) => outcome.negative).length;

    return {
      threshold,
      total: outcomes.length,
      accepted: accepted.length,
      acceptedWrong,
      coverage: accepted.length / Math.max(outcomes.length, 1),
      risk: acceptedWrong / Math.max(accepted.length, 1),
      silentMeaningChangeRate: acceptedWrong / Math.max(outcomes.length, 1),
      negativeFalseAcceptanceRate: negativeFalseAccepts / Math.max(negativeTotal, 1),
    };
  });
}

/**
 * Throw when reports disagree about the artifact that produced them. A report
 * set is only comparable when every member came from one decoded artifact.
 */
export function assertSameArtifact(reports: readonly { artifactSha256: string }[]): string {
  if (reports.length === 0) {
    throw new Error("assertSameArtifact: at least one report is required");
  }

  const expected = reports[0]!.artifactSha256;

  for (const report of reports) {
    if (report.artifactSha256 !== expected) {
      throw new Error(`artifact mismatch: ${report.artifactSha256} !== ${expected}`);
    }
  }

  return expected;
}

interface WeightsArtifact {
  q: string;
}

interface DistIndex {
  parse: (text: string) => Promise<ParseResult>;
  compile: (intent: ScheduleIntent) => CompileResult;
}

interface DistTokenizer {
  tokenize: (text: string) => RawToken[];
}

interface DistInference {
  inferTokens: (tokens: readonly RawToken[]) => { labels: Uint8Array };
}

interface DistLabels {
  LABELS: readonly string[];
}

interface DistEvaluation extends DistTokenizer, DistInference, DistLabels {
  weights: WeightsArtifact;
}

/** A decoded runtime plus the provenance of the artifact it runs. */
export interface Engine {
  artifactSha256: string;
  parse: (text: string) => Promise<ParseResult>;
  compile: (intent: ScheduleIntent) => CompileResult;
  tokenize: (text: string) => RawToken[];
  inferTokens: (tokens: readonly RawToken[]) => { labels: Uint8Array };
  labelName: (id: number) => string;
  dispose: () => Promise<void>;
}

async function importModule<T>(path: string): Promise<T> {
  return (await import(pathToFileURL(path).href)) as T;
}

/**
 * Load the built runtime from `distDir`. With no `weightsModulePath` the built
 * module runs as shipped. With one, a scratch copy of `dist/` is written with
 * the supplied artifact so the built engine can be scored on it.
 */
export async function loadEngine(distDir: string, weightsModulePath?: string): Promise<Engine> {
  const built = resolveRepo(distDir);

  if (!existsSync(join(built, "index.js"))) {
    throw new Error(`built runtime missing: ${built}/index.js (run the build)`);
  }

  let root = built;
  let artifact: WeightsArtifact;
  let temp: string | null = null;

  if (weightsModulePath === undefined) {
    const evaluation = await importModule<DistEvaluation>(join(built, "evaluation.js"));

    artifact = evaluation.weights;
  } else {
    const modulePath = resolveRepo(weightsModulePath);

    if (!existsSync(modulePath)) {
      throw new Error(`weights module not found: ${modulePath}`);
    }

    const supplied = await importModule<{ weights: WeightsArtifact }>(modulePath);

    artifact = supplied.weights;
    temp = await mkdtemp(join(tmpdir(), "syntaxless-evaluate-"));
    root = join(temp, "dist");
    await cp(built, root, { recursive: true });

    await writeFile(
      join(root, "model", "weights.gen.js"),
      `export const weights = ${JSON.stringify(artifact)};\n`,
    );
  }

  const index = await importModule<DistIndex>(join(root, "index.js"));
  const evaluation = await importModule<DistEvaluation>(join(root, "evaluation.js"));

  const artifactSha256 = sha256Hex(artifact.q);
  const scratch = temp;

  return {
    artifactSha256,
    parse: index.parse.bind(index),
    compile: index.compile.bind(index),
    tokenize: evaluation.tokenize.bind(evaluation),
    inferTokens: evaluation.inferTokens.bind(evaluation),
    labelName: (id: number) => {
      const label = evaluation.LABELS[id];

      if (label === undefined) throw new RangeError(`label id out of range: ${id}`);

      return label;
    },
    dispose: async () => {
      if (scratch !== null) await rm(scratch, { recursive: true, force: true });
    },
  };
}

interface Accumulator {
  total: number;
  correct: number;
  byFamily: Record<string, Counts>;
  failureCounts: Record<string, number>;
  sampledFailures: Record<string, number>;
  failures: ContractFailure[];
}

function emptyAccumulator(): Accumulator {
  return {
    total: 0,
    correct: 0,
    byFamily: {},
    failureCounts: {},
    sampledFailures: {},
    failures: [],
  };
}

function record(
  accumulator: Accumulator,
  family: string,
  correct: boolean,
  category: string,
  failure?: ContractFailure,
): void {
  const counts = (accumulator.byFamily[family] ??= { total: 0, correct: 0 });

  counts.total += 1;
  accumulator.total += 1;

  if (correct) {
    counts.correct += 1;
    accumulator.correct += 1;
  } else {
    const bucket = `${family}:${category}`;

    accumulator.failureCounts[bucket] = (accumulator.failureCounts[bucket] ?? 0) + 1;
    const sampled = accumulator.sampledFailures[bucket] ?? 0;

    if (failure !== undefined && sampled < MAX_FAILURES_PER_CATEGORY) {
      accumulator.failures.push(failure);
      accumulator.sampledFailures[bucket] = sampled + 1;
    }
  }
}

function report(
  metric: Metric,
  accumulator: Accumulator,
  provenance: {
    artifactSha256: string;
    corpus: string;
    corpusSha256: string;
    familyDistribution: Record<string, number>;
    scope: string;
  },
): ContractReport {
  return {
    metric,
    ...provenance,
    total: accumulator.total,
    correct: accumulator.correct,
    accuracy: accumulator.total === 0 ? 0 : accumulator.correct / accumulator.total,
    byFamily: accumulator.byFamily,
    failureCounts: accumulator.failureCounts,
    failures: accumulator.failures,
  };
}

const WEEKDAY_ORDER = [
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
] as const;

function canonicalStrings<T extends string>(values: readonly T[] | undefined): T[] | undefined {
  return values === undefined ? undefined : [...new Set(values)].sort();
}

/** Canonical semantic form used for evaluation; input arrays are never mutated. */
export function canonicalizeIntent(intent: ScheduleIntent): ScheduleIntent {
  const weekdayRank = new Map(WEEKDAY_ORDER.map((day, index) => [day, index]));

  const weekdays =
    intent.weekdays === undefined
      ? undefined
      : [...new Set(intent.weekdays)].sort(
          (left, right) => (weekdayRank.get(left) ?? 99) - (weekdayRank.get(right) ?? 99),
        );

  const exceptionWeekdays =
    intent.exceptions?.weekdays === undefined
      ? undefined
      : [...new Set(intent.exceptions.weekdays)].sort(
          (left, right) => (weekdayRank.get(left) ?? 99) - (weekdayRank.get(right) ?? 99),
        );

  const exceptionDates = canonicalStrings(intent.exceptions?.dates);

  return {
    ...intent,
    ...(weekdays === undefined ? {} : { weekdays }),
    ...(intent.exceptions === undefined
      ? {}
      : {
          exceptions: {
            ...(exceptionDates === undefined ? {} : { dates: exceptionDates }),
            ...(exceptionWeekdays === undefined ? {} : { weekdays: exceptionWeekdays }),
          },
        }),
  };
}

function intentDifference(expected: ScheduleIntent, actual: ScheduleIntent): string {
  const fields = new Set([...Object.keys(expected), ...Object.keys(actual)]);

  const different = [...fields].filter(
    (field) =>
      !isDeepStrictEqual(
        canonicalizeIntent(expected)[field as keyof ScheduleIntent],
        canonicalizeIntent(actual)[field as keyof ScheduleIntent],
      ),
  );

  return different.length === 0 ? "canonical-order" : different.sort().join("+");
}

/** Map each non-whitespace token onto the gold span that covers it. */
function goldLabelFor(
  token: RawToken,
  spans: GoldRecord["spans"],
  cursor: { index: number },
): string | null {
  while (cursor.index < spans.length && spans[cursor.index]!.end <= token.start) {
    cursor.index += 1;
  }

  const span = spans[cursor.index];

  if (span === undefined || token.start < span.start || token.end > span.end) {
    throw new Error(`unaligned token span`);
  }

  return span.label;
}

export function parseCorpus(contents: string): GoldRecord[] {
  const records = contents
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => repairObservableGold(JSON.parse(line) as GoldRecord))
    .map(repairInclusiveBoundGold)
    .map((item) =>
      item.intent === null ? item : { ...item, intent: canonicalizeIntent(item.intent) },
    );

  lintGoldRecords(records);

  return records;
}

/** Repair legacy held-out rows that rendered an exclusive end date after the
 * inclusive word "through". Newly generated rows already display end - 1. */
export function repairInclusiveBoundGold(item: GoldRecord): GoldRecord {
  const end = item.intent?.bounds?.end;

  if (item.family !== "bounds" || end === undefined || !/\bthrough\b/i.test(item.text)) return item;
  const shown = /\bthrough\s+(\d{4}-\d{2}-\d{2})\b/i.exec(item.text)?.[1];

  if (shown !== end) return item;
  const date = new Date(`${end}T00:00:00Z`);

  if (Number.isNaN(date.getTime())) return item;
  date.setUTCDate(date.getUTCDate() + 1);

  return {
    ...item,
    intent: {
      ...item.intent!,
      bounds: { ...item.intent!.bounds, end: date.toISOString().slice(0, 10) },
    },
  };
}

/** Fail before scoring when the gold contract is malformed or unobservable. */
export function lintGoldRecords(records: readonly GoldRecord[]): void {
  const ids = new Set<string>();

  for (const item of records) {
    if (!item.id || ids.has(item.id)) throw new Error(`duplicate or empty gold id: ${item.id}`);
    ids.add(item.id);
    if (!item.family || !item.text.trim()) throw new Error(`invalid gold text: ${item.id}`);

    if (item.intent === null && item.expected.supported) {
      throw new Error(`negative gold cannot expect compilation: ${item.id}`);
    }

    let end = 0;

    for (const span of item.spans) {
      if (
        !span.label ||
        !Number.isInteger(span.start) ||
        !Number.isInteger(span.end) ||
        span.start < end ||
        span.start < 0 ||
        span.end <= span.start ||
        span.end > item.text.length
      ) {
        throw new Error(`invalid gold span: ${item.id}`);
      }

      end = span.end;
    }
  }
}

/** Older reserved clock rows rendered a twelve-hour clock while keeping the
 * sampled 24-hour intent, so `1:30 daily` could secretly mean either 01:30 or
 * 13:30. Make the latent choice observable at evaluation time without writing
 * another generated JSONL artifact into Git. */
export function repairObservableGold(item: GoldRecord): GoldRecord {
  if (
    item.family !== "time-of-day" ||
    item.intent?.time === undefined ||
    /\b(?:am|pm)\b/i.test(item.text)
  ) {
    return item;
  }

  const minute = [...item.spans].reverse().find((span) => span.label === "MINUTE");

  if (minute === undefined) return item;
  const meridiem = item.intent.time.hour >= 12 ? "pm" : "am";
  const insertion = minute.end;
  const added = ` ${meridiem}`;

  const spans = item.spans.map((span) =>
    span.start >= insertion
      ? { ...span, start: span.start + added.length, end: span.end + added.length }
      : span,
  );

  spans.push({
    start: insertion + 1,
    end: insertion + added.length,
    label: "MERIDIEM",
    clauseStart: false,
  });

  spans.sort((left, right) => left.start - right.start);

  return {
    ...item,
    text: `${item.text.slice(0, insertion)}${added}${item.text.slice(insertion)}`,
    spans,
  };
}

/**
 * Score one decoded engine against a corpus. All three contracts share the
 * engine's `artifactSha256`; the caller may assert that with
 * {@link assertSameArtifact}.
 */
export async function evaluateCorpus(
  engine: Engine,
  corpusPath: string,
  scope: string,
): Promise<EvaluationResult> {
  const source = resolveRepo(corpusPath);
  const contents = await readFile(source, "utf8");
  const records = parseCorpus(contents);
  const effectiveContents = records.map((item) => JSON.stringify(item)).join("\n");
  const corpusSha256 = sha256Hex(effectiveContents.length === 0 ? "" : `${effectiveContents}\n`);

  const familyDistribution: Record<string, number> = {};

  for (const item of records) {
    familyDistribution[item.family] = (familyDistribution[item.family] ?? 0) + 1;
  }

  const tokenRoles = emptyAccumulator();
  const exactAst = emptyAccumulator();
  const compiledExpression = emptyAccumulator();
  const selectiveOutcomes: SelectiveOutcome[] = [];

  for (const item of records) {
    const tokens = engine.tokenize(item.text);
    const predictions = engine.inferTokens(tokens);
    const cursor = { index: 0 };

    for (let index = 0; index < tokens.length; index += 1) {
      const token = tokens[index]!;

      if (token.kind === 3) continue;
      const gold = goldLabelFor(token, item.spans, cursor);
      const predicted = engine.labelName(predictions.labels[index]!);
      const correct = gold === predicted;

      record(
        tokenRoles,
        item.family,
        correct,
        `role:${String(gold)}->${predicted}`,
        correct
          ? undefined
          : {
              id: item.id,
              family: item.family,
              detail: `token ${JSON.stringify(token.text)}`,
              expected: gold,
              actual: predicted,
            },
      );
    }

    const parsed = await engine.parse(item.text);

    if (item.intent === null) {
      const rejected = !parsed.ok;

      selectiveOutcomes.push({
        accepted: parsed.ok,
        correct: rejected,
        confidence: parsed.confidence,
        negative: true,
      });

      record(exactAst, item.family, rejected, "negative-false-accept", {
        id: item.id,
        family: item.family,
        detail: "expected no schedule",
        expected: null,
        actual: parsed.ok ? parsed.intent : { ok: false, code: parsed.code },
      });

      record(compiledExpression, item.family, rejected, "negative-false-accept", {
        id: item.id,
        family: item.family,
        detail: "expected no expression",
        expected: item.expected,
        actual: parsed.ok ? "parsed" : { ok: false, code: parsed.code },
      });

      continue;
    }

    const canonicalExpected = canonicalizeIntent(item.intent);
    const canonicalActual = parsed.ok ? canonicalizeIntent(parsed.intent) : null;
    const astCorrect = parsed.ok && isDeepStrictEqual(canonicalActual, canonicalExpected);

    selectiveOutcomes.push({
      accepted: parsed.ok,
      correct: astCorrect,
      confidence: parsed.confidence,
      negative: false,
    });

    const astCategory = parsed.ok
      ? `intent:${intentDifference(canonicalExpected, parsed.intent)}`
      : `parse:${parsed.code}`;

    record(exactAst, item.family, astCorrect, astCategory, {
      id: item.id,
      family: item.family,
      detail: "intent mismatch",
      expected: canonicalExpected,
      actual: parsed.ok ? canonicalActual : { ok: false, code: parsed.code },
    });

    const goldCompiled = engine.compile(canonicalExpected);

    if (
      goldCompiled.supported !== item.expected.supported ||
      (goldCompiled.supported && goldCompiled.expression !== item.expected.expression) ||
      (!goldCompiled.supported && item.expected.expression !== null)
    ) {
      throw new Error(`invalid gold compiler expectation: ${item.id}`);
    }

    const compiled = parsed.ok ? engine.compile(parsed.intent) : null;

    const expressionCorrect = item.expected.supported
      ? parsed.ok &&
        compiled !== null &&
        compiled.supported &&
        compiled.expression === item.expected.expression
      : parsed.ok &&
        compiled !== null &&
        !compiled.supported &&
        !goldCompiled.supported &&
        compiled.code === goldCompiled.code;

    const compileCategory = !parsed.ok
      ? `parse:${parsed.code}`
      : compiled?.supported
        ? "wrong-expression"
        : `unsupported:${compiled?.code ?? "missing"}`;

    record(compiledExpression, item.family, expressionCorrect, compileCategory, {
      id: item.id,
      family: item.family,
      detail: "compiled expression mismatch",
      expected: goldCompiled,
      actual: compiled,
    });
  }

  const provenance = {
    artifactSha256: engine.artifactSha256,
    corpus: source,
    corpusSha256,
    familyDistribution,
    scope,
  };

  return {
    artifactSha256: engine.artifactSha256,
    corpus: source,
    corpusSha256,
    familyDistribution,
    scope,
    contracts: [
      report("tokenRoles", tokenRoles, provenance),
      report("exactAst", exactAst, provenance),
      report("compiledExpression", compiledExpression, provenance),
    ],
    selectivePrediction: riskCoverage(selectiveOutcomes),
  };
}

export interface EvaluateOptions {
  dist: string;
  corpus: string;
  out: string;
  weights?: string;
}

function readOption(argv: readonly string[], index: number): string {
  const value = argv[index + 1];

  if (value === undefined) throw new RangeError(`missing value for ${argv[index]}`);

  return value;
}

export function parseArgs(argv: readonly string[]): EvaluateOptions {
  let dist: string = DEFAULTS.dist;
  let corpus: string = DEFAULTS.corpus;
  let out: string = DEFAULTS.out;
  let weights: string | undefined;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]!;

    switch (arg) {
      case "--dist":
        dist = readOption(argv, index);
        index += 1;
        break;
      case "--corpus":
        corpus = readOption(argv, index);
        index += 1;
        break;
      case "--out":
        out = readOption(argv, index);
        index += 1;
        break;
      case "--weights":
        weights = readOption(argv, index);
        index += 1;
        break;
      default:
        throw new RangeError(`unknown flag: ${arg}`);
    }
  }

  return { dist, corpus, out, ...(weights === undefined ? {} : { weights }) };
}

export const DEFAULT_SCOPE =
  "Held-out reserved-carrier phrases. Calibration and held-out frame splits " +
  "are development data and may not gate promotion.";

if (import.meta.main) {
  const args = parseArgs(process.argv.slice(2));
  const engine = await loadEngine(args.dist, args.weights);

  try {
    const result = await evaluateCorpus(engine, args.corpus, DEFAULT_SCOPE);

    assertSameArtifact(result.contracts);
    const out = resolveRepo(args.out);

    await mkdir(dirname(out), { recursive: true });
    await writeFile(out, `${JSON.stringify(result, null, 2)}\n`);

    for (const contract of result.contracts) {
      process.stdout.write(
        `${contract.metric}: ${contract.correct}/${contract.total} ` +
          `(${contract.accuracy.toFixed(3)})\n`,
      );
    }

    process.stdout.write(`artifact sha256: ${result.artifactSha256}\n`);
    process.stdout.write(`report -> ${out}\n`);
  } finally {
    await engine.dispose();
  }
}
