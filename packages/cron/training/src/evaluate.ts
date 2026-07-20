/**
 * Candidate/baseline evaluation over a gold corpus.
 *
 * Baselines run the built runtime. Typed candidates run current source with
 * their own decoded artifact; legacy candidates use the built runtime.
 * No source copies, generated-module replacement, or scratch trees are needed.
 *
 * The corpus is gold data with rendered spans, an expected AST, and an expected
 * compiled expression. The four contracts (`tokenRoles`, `rawDecode`,
 * `exactAst`, `compiledExpression`) are reported side by side and never summed.
 * Speech-act and actionable accuracy are extra columns, not a fifth contract.
 *
 * Scorers return evidence; callers own output files. The calibration and
 * held-out frame splits are development data and may not gate promotion.
 * Callers provide the report's evidence scope explicitly.
 */
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { isDeepStrictEqual } from "node:util";

import type { CompileResult } from "../../src/compiler/cron";
import type { WeightsArtifact } from "../../src/model/decode";
import { createEvaluation } from "../../src/model/evaluation";
import { isScheduleLabel, isStructural, LABEL_COUNT } from "../../src/model/labels";
import type { Label } from "../../src/model/labels";
import { SPEECH_ACTS } from "../../src/model/speech-acts";
import type { SpeechAct } from "../../src/model/speech-acts";
import type { RawToken } from "../../src/model/tokenizer";
import type { ParseResult } from "../../src/parser/parse";
import type { ScheduleIntent } from "../../src/types/intent";
import { addAttribution, attributeMiss, emptyAttributionReport } from "./attribution";
import type { AttributionReport } from "./attribution";
import { canonicalizeIntent } from "./canonical-intent";
import { execEqual } from "./exec-score";
import { faAtCoverage } from "./fa-at-cov";
import { spellingVariant } from "./noise";
import type { SpellingNoise } from "./noise";
import { oracleDecode } from "./oracle";
import type { OracleRecord } from "./oracle";
import { artifactSha256 } from "./weights-hash";

export { canonicalizeIntent } from "./canonical-intent";

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
    label: Label;
    clauseStart: boolean;
    actionable?: boolean;
  }[];
  intent: ScheduleIntent | null;
  expected: { supported: boolean; expression: string | null };
  speechAct?: SpeechAct;
}

/** Running correct/total counts; the guard's `Counts` shape. */
export interface Counts {
  total: number;
  correct: number;
}

export type Metric = "tokenRoles" | "rawDecode" | "exactAst" | "compiledExpression";

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

export interface OverlayOffEvaluation {
  contracts: ContractReport[];
  selectivePrediction: RiskCoveragePoint[];
  rawSelectivePrediction: RiskCoveragePoint[];
  typoInvariance: TypoInvarianceReport;
}

export interface EvaluationResult {
  artifactSha256: string;
  corpus: string;
  corpusSha256: string;
  familyDistribution: Record<string, number>;
  scope: string;
  contracts: ContractReport[];
  /** Public parser acceptance after its current confidence policy. */
  selectivePrediction: RiskCoveragePoint[];
  /** Counterfactual raw-decode curve used to select a better policy. */
  rawSelectivePrediction: RiskCoveragePoint[];
  typoInvariance: TypoInvarianceReport;
  /**
   * Overlay-off scores on the same artifact. Promotion uses this block;
   * overlay-on remains diagnostic. Missing overlay-off is a failed run.
   */
  overlayOff: OverlayOffEvaluation;
  /** Extra column; not a contract and not averaged into exactAst. */
  speechActAccuracy?: number;
  /** Extra column; not a contract and not averaged into exactAst. */
  actionableAccuracy?: number;
  /** Per-miss ROLE/ACT/DECODE/CONF/GOLD histogram (overlay-off exactAst). */
  attribution: AttributionReport;
  /** Overlay-off EXEC: execEqual(prediction, gold), not oracle correct/total. */
  exec: { correct: number; total: number };
  /** Overlay-off FA@cov from parse accepts after the F abstention threshold. */
  faAtCov: string;
}

export interface TypoInvarianceReport {
  total: number;
  cleanCorrect: number;
  noisyCorrect: number;
  pairedCorrect: number;
  noisySilentWrong: number;
  noisyAccuracy: number;
  pairedAccuracy: number;
  byFamily: Record<string, Counts>;
  byNoise: Partial<Record<SpellingNoise, Counts>>;
  /** Complete counts, grouped by family/noise/outcome. */
  failureCounts: Record<string, number>;
  /** Bounded examples; `failureCounts` remains complete. */
  failures: TypoFailure[];
}

export interface TypoFailure {
  id: string;
  family: string;
  noise: SpellingNoise;
  text: string;
  expected: ScheduleIntent;
  actual: ScheduleIntent | { ok: false; code: string };
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
  thresholds: readonly number[] = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
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

type EvaluationRuntime = ReturnType<typeof createEvaluation>;
type BuiltEvaluation = typeof import("../../src/model/evaluation");

/** A decoded runtime plus the provenance of the artifact it runs. No resources to close. */
export interface Engine {
  artifactSha256: string;
  predict: EvaluationRuntime["predict"];
  compile: (intent: ScheduleIntent) => CompileResult;
  tokenize: (text: string) => RawToken[];
  inferTokens: EvaluationRuntime["inferTokens"];
  labelName: (id: number) => string;
}

export type EvaluationPrediction = ReturnType<Engine["predict"]>;

async function importModule<T>(path: string): Promise<T> {
  return (await import(pathToFileURL(path).href)) as T;
}

async function builtEvaluation(distDir: string): Promise<BuiltEvaluation> {
  const built = resolveRepo(distDir);

  if (!existsSync(join(built, "index.js"))) {
    throw new Error(`built runtime missing: ${built}/index.js (run the build)`);
  }

  return importModule<BuiltEvaluation>(join(built, "evaluation.js"));
}

/** Baseline: built code/artifact. Typed candidate: current code/supplied artifact. */
export async function loadEngine(distDir: string, weightsModulePath?: string): Promise<Engine> {
  let evaluation: EvaluationRuntime;

  if (weightsModulePath === undefined) {
    const built = await builtEvaluation(distDir);

    evaluation = built.createEvaluation(built.weights);
  } else {
    const modulePath = resolveRepo(weightsModulePath);

    if (!existsSync(modulePath)) throw new Error(`weights module not found: ${modulePath}`);
    const supplied = await importModule<{ weights: WeightsArtifact }>(modulePath);

    evaluation =
      supplied.weights.roleClasses === LABEL_COUNT || supplied.weights.roleClasses === 77
        ? createEvaluation(supplied.weights)
        : (await builtEvaluation(distDir)).createEvaluation(supplied.weights);
  }

  return {
    artifactSha256: artifactSha256(evaluation.weights),
    predict: evaluation.predict,
    compile: evaluation.compile,
    tokenize: evaluation.tokenize,
    inferTokens: evaluation.inferTokens,
    labelName: (id: number) => {
      const label = evaluation.LABELS[id];

      if (label === undefined) throw new RangeError(`label id out of range: ${id}`);

      return label;
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
): Label {
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
    .map((line) => JSON.parse(line) as GoldRecord)
    .map((item) =>
      item.intent === null ? item : { ...item, intent: canonicalizeIntent(item.intent) },
    );

  lintGoldRecords(records);

  return records;
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

interface ParseScoreBuckets {
  rawDecode: Accumulator;
  exactAst: Accumulator;
  compiledExpression: Accumulator;
  selectiveOutcomes: SelectiveOutcome[];
  rawSelectiveOutcomes: SelectiveOutcome[];
}

function emptyParseScoreBuckets(): ParseScoreBuckets {
  return {
    rawDecode: emptyAccumulator(),
    exactAst: emptyAccumulator(),
    compiledExpression: emptyAccumulator(),
    selectiveOutcomes: [],
    rawSelectiveOutcomes: [],
  };
}

function emptyTypoReport(): TypoInvarianceReport {
  return {
    total: 0,
    cleanCorrect: 0,
    noisyCorrect: 0,
    pairedCorrect: 0,
    noisySilentWrong: 0,
    noisyAccuracy: 0,
    pairedAccuracy: 0,
    byFamily: {},
    byNoise: {},
    failureCounts: {},
    failures: [],
  };
}

function finalizeTypo(report: TypoInvarianceReport): void {
  report.noisyAccuracy = report.noisyCorrect / Math.max(report.total, 1);
  report.pairedAccuracy = report.pairedCorrect / Math.max(report.total, 1);
}

function recordTypo(
  report: TypoInvarianceReport,
  item: GoldRecord,
  canonicalExpected: ScheduleIntent,
  astCorrect: boolean,
  noisy: { noise: SpellingNoise; example: { text: string } },
  noisyParsed: ParseResult,
): void {
  const noisyCorrect =
    noisyParsed.ok && isDeepStrictEqual(canonicalizeIntent(noisyParsed.intent), canonicalExpected);

  report.total += 1;
  report.cleanCorrect += Number(astCorrect);
  report.noisyCorrect += Number(noisyCorrect);
  report.pairedCorrect += Number(astCorrect && noisyCorrect);
  report.noisySilentWrong += Number(noisyParsed.ok && !noisyCorrect);
  const family = (report.byFamily[item.family] ??= { total: 0, correct: 0 });

  family.total += 1;
  family.correct += Number(noisyCorrect);
  const noise = (report.byNoise[noisy.noise] ??= { total: 0, correct: 0 });

  noise.total += 1;
  noise.correct += Number(noisyCorrect);

  if (!noisyCorrect) {
    const outcome = noisyParsed.ok
      ? `intent:${intentDifference(canonicalExpected, noisyParsed.intent)}`
      : `parse:${noisyParsed.code}`;

    const category = `${item.family}/${noisy.noise}/${outcome}`;
    const count = (report.failureCounts[category] ?? 0) + 1;

    report.failureCounts[category] = count;

    if (count <= MAX_FAILURES_PER_CATEGORY) {
      report.failures.push({
        id: item.id,
        family: item.family,
        noise: noisy.noise,
        text: noisy.example.text,
        expected: canonicalExpected,
        actual: noisyParsed.ok
          ? canonicalizeIntent(noisyParsed.intent)
          : { ok: false, code: noisyParsed.code },
      });
    }
  }
}

function scoreParsedGold(
  engine: Engine,
  item: GoldRecord,
  parsed: ParseResult,
  raw: ReturnType<EvaluationPrediction["decodeRaw"]> | null,
  rawError: unknown,
  buckets: ParseScoreBuckets,
): { astCorrect: boolean } {
  if (item.intent === null) {
    const rejected = !parsed.ok;

    buckets.rawSelectiveOutcomes.push({
      accepted: raw !== null,
      correct: raw === null,
      confidence: raw?.confidence ?? 0,
      negative: true,
    });

    record(buckets.rawDecode, item.family, raw === null, "negative-false-accept", {
      id: item.id,
      family: item.family,
      detail: "expected no raw schedule",
      expected: null,
      actual: raw?.intent ?? String(rawError),
    });

    buckets.selectiveOutcomes.push({
      accepted: parsed.ok,
      correct: rejected,
      confidence: parsed.confidence,
      negative: true,
    });

    record(buckets.exactAst, item.family, rejected, "negative-false-accept", {
      id: item.id,
      family: item.family,
      detail: "expected no schedule",
      expected: null,
      actual: parsed.ok ? parsed.intent : { ok: false, code: parsed.code },
    });

    record(buckets.compiledExpression, item.family, rejected, "negative-false-accept", {
      id: item.id,
      family: item.family,
      detail: "expected no expression",
      expected: item.expected,
      actual: parsed.ok ? "parsed" : { ok: false, code: parsed.code },
    });

    return { astCorrect: false };
  }

  const canonicalExpected = canonicalizeIntent(item.intent);
  const canonicalRaw = raw === null ? null : canonicalizeIntent(raw.intent);
  const rawCorrect = raw !== null && isDeepStrictEqual(canonicalRaw, canonicalExpected);

  buckets.rawSelectiveOutcomes.push({
    accepted: raw !== null,
    correct: rawCorrect,
    confidence: raw?.confidence ?? 0,
    negative: false,
  });

  record(
    buckets.rawDecode,
    item.family,
    rawCorrect,
    raw === null ? "decode-error" : `intent:${intentDifference(canonicalExpected, raw.intent)}`,
    {
      id: item.id,
      family: item.family,
      detail: "pre-confidence intent mismatch",
      expected: canonicalExpected,
      actual: raw === null ? String(rawError) : canonicalRaw,
    },
  );

  const canonicalActual = parsed.ok ? canonicalizeIntent(parsed.intent) : null;
  const astCorrect = parsed.ok && isDeepStrictEqual(canonicalActual, canonicalExpected);

  buckets.selectiveOutcomes.push({
    accepted: parsed.ok,
    correct: astCorrect,
    confidence: parsed.confidence,
    negative: false,
  });

  const astCategory = parsed.ok
    ? `intent:${intentDifference(canonicalExpected, parsed.intent)}`
    : `parse:${parsed.code}`;

  record(buckets.exactAst, item.family, astCorrect, astCategory, {
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

  record(buckets.compiledExpression, item.family, expressionCorrect, compileCategory, {
    id: item.id,
    family: item.family,
    detail: "compiled expression mismatch",
    expected: goldCompiled,
    actual: compiled,
  });

  return { astCorrect };
}

/**
 * Score one decoded engine against a corpus. All four contracts share the
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
  const overlayOn = emptyParseScoreBuckets();
  const overlayOff = emptyParseScoreBuckets();
  const typoInvariance = emptyTypoReport();
  const typoOff = emptyTypoReport();
  const attribution = emptyAttributionReport();
  let speechActTotal = 0;
  let speechActCorrect = 0;
  let actionableTotal = 0;
  let actionableCorrect = 0;
  let execCorrect = 0;
  let execTotal = 0;
  let negativeTotal = 0;
  let negativeAccepted = 0;
  let positiveTotal = 0;
  let positiveAccepted = 0;

  for (const item of records) {
    const prediction = engine.predict(item.text);

    // Gold spans retain original offsets and features when normalization changes text.
    const tokens =
      prediction.normalized === item.text ? prediction.tokens : engine.tokenize(item.text);

    const predictions =
      prediction.normalized === item.text ? prediction.predictions : engine.inferTokens(tokens);

    const cursor = { index: 0 };
    const goldLabels: Label[] = [];
    const predictedLabels: Label[] = [];
    const goldActionable: boolean[] = [];
    const predictedActionable: boolean[] = [];

    for (let index = 0; index < tokens.length; index += 1) {
      const token = tokens[index]!;

      if (token.kind === 3) continue;
      const gold = goldLabelFor(token, item.spans, cursor);
      const predicted = engine.labelName(predictions.labels[index]!);
      const correct = gold === predicted;

      goldLabels.push(gold);
      predictedLabels.push(predicted as Label);
      const goldSpan = item.spans[cursor.index];

      if (goldSpan?.actionable !== undefined) goldActionable.push(goldSpan.actionable);

      if (predictions.actionable !== undefined) {
        predictedActionable.push(predictions.actionable[index] === 1);
      }

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

      if (goldSpan?.actionable !== undefined && predictions.actionable !== undefined) {
        actionableTotal += 1;
        actionableCorrect += Number((predictions.actionable[index] === 1) === goldSpan.actionable);
      }
    }

    if (item.speechAct !== undefined && predictions.speechAct !== undefined) {
      speechActTotal += 1;
      speechActCorrect += Number(SPEECH_ACTS[predictions.speechAct] === item.speechAct);
    }

    const parsed = prediction.parse();
    let raw: ReturnType<EvaluationPrediction["decodeRaw"]> | null = null;
    let rawError: unknown;

    try {
      raw = prediction.decodeRaw();
    } catch (error) {
      rawError = error;
    }

    const parsedOff = prediction.parse({ overlay: "off" });
    let rawOff: ReturnType<EvaluationPrediction["decodeRaw"]> | null = null;
    let rawOffError: unknown;

    try {
      rawOff = prediction.decodeRaw("off");
    } catch (error) {
      rawOffError = error;
    }

    const { astCorrect } = scoreParsedGold(engine, item, parsed, raw, rawError, overlayOn);

    const { astCorrect: astCorrectOff } = scoreParsedGold(
      engine,
      item,
      parsedOff,
      rawOff,
      rawOffError,
      overlayOff,
    );

    execTotal += 1;
    if (execEqual(parsedOff.ok ? parsedOff.intent : null, item.intent)) execCorrect += 1;

    if (item.intent === null) {
      negativeTotal += 1;
      if (parsedOff.ok) negativeAccepted += 1;
    } else {
      positiveTotal += 1;
      if (parsedOff.ok) positiveAccepted += 1;
    }

    const exactMiss = item.intent === null ? parsedOff.ok : !astCorrectOff;

    if (exactMiss) {
      const oracleRecord: OracleRecord = {
        id: item.id,
        text: item.text,
        spans: item.spans,
        intent: item.intent,
        ...(item.speechAct === undefined ? {} : { speechAct: item.speechAct }),
        family: item.family,
        source: "gold",
      };

      let predictedOracle: ReturnType<typeof oracleDecode> | undefined;

      if (rawOff !== null) {
        predictedOracle = { ok: true, intent: rawOff.intent };
      } else if (rawOffError !== undefined) {
        predictedOracle = {
          ok: false,
          error: rawOffError instanceof Error ? rawOffError.message : String(rawOffError),
        };
      }

      addAttribution(attribution, {
        id: item.id,
        source: "reserved",
        family: item.family,
        bucket: attributeMiss({
          text: item.text,
          expectedIntent: item.intent,
          accepted: parsedOff.ok,
          actualIntent: parsedOff.ok ? parsedOff.intent : null,
          goldSpans: item.spans,
          ...(item.speechAct === undefined ? {} : { goldSpeechAct: item.speechAct }),
          goldLabels,
          predictedLabels,
          ...(goldActionable.length === 0 ? {} : { goldActionable }),
          ...(predictedActionable.length === 0 ? {} : { predictedActionable }),
          goldOracle: oracleDecode(oracleRecord),
          ...(predictedOracle === undefined ? {} : { predictedOracle }),
        }),
      });
    }

    if (item.intent === null) continue;

    const canonicalExpected = canonicalizeIntent(item.intent);

    // Mutate only labels the decoder does not look up from the string
    // (O / EDGE). Typos on WEEKDAY/HOUR/BOUND/RECUR-monthly are unreachable gold.
    let noisy = undefined;

    for (let attempt = 0; attempt < 64 && noisy === undefined; attempt += 1) {
      noisy = spellingVariant(
        {
          ...item,
          spans: item.spans.map((span) => ({
            ...span,
            actionable: isScheduleLabel(span.label) && !isStructural(span.label),
          })),
          template: item.id,
          style: "evaluation",
          speechAct: "request",
        },
        `${item.id}/evaluation/${attempt}`,
      );
    }

    if (noisy !== undefined) {
      const noisyPrediction = engine.predict(noisy.example.text);
      const noisyParsed = noisyPrediction.parse();

      recordTypo(typoInvariance, item, canonicalExpected, astCorrect, noisy, noisyParsed);

      const noisyParsedOff = noisyPrediction.parse({
        overlay: "off",
      });

      recordTypo(typoOff, item, canonicalExpected, astCorrectOff, noisy, noisyParsedOff);
    }
  }

  const provenance = {
    artifactSha256: engine.artifactSha256,
    corpus: source,
    corpusSha256,
    familyDistribution,
    scope,
  };

  finalizeTypo(typoInvariance);
  finalizeTypo(typoOff);

  return {
    artifactSha256: engine.artifactSha256,
    corpus: source,
    corpusSha256,
    familyDistribution,
    scope,
    contracts: [
      report("tokenRoles", tokenRoles, provenance),
      report("rawDecode", overlayOn.rawDecode, provenance),
      report("exactAst", overlayOn.exactAst, provenance),
      report("compiledExpression", overlayOn.compiledExpression, provenance),
    ],
    selectivePrediction: riskCoverage(overlayOn.selectiveOutcomes),
    rawSelectivePrediction: riskCoverage(overlayOn.rawSelectiveOutcomes),
    typoInvariance,
    overlayOff: {
      contracts: [
        report("rawDecode", overlayOff.rawDecode, provenance),
        report("exactAst", overlayOff.exactAst, provenance),
        report("compiledExpression", overlayOff.compiledExpression, provenance),
      ],
      selectivePrediction: riskCoverage(overlayOff.selectiveOutcomes),
      rawSelectivePrediction: riskCoverage(overlayOff.rawSelectiveOutcomes),
      typoInvariance: typoOff,
    },
    ...(speechActTotal === 0 ? {} : { speechActAccuracy: speechActCorrect / speechActTotal }),
    ...(actionableTotal === 0 ? {} : { actionableAccuracy: actionableCorrect / actionableTotal }),
    attribution,
    exec: { correct: execCorrect, total: execTotal },
    faAtCov: faAtCoverage({
      negativeTotal,
      negativeAccepted,
      positiveTotal,
      positiveAccepted,
    }).formatted,
  };
}
