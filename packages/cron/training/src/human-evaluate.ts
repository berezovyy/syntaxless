/**
 * End-to-end evaluation for the versioned, human-style acceptance corpus.
 *
 * This corpus is intentionally not a RenderedExample: it has no spans, frame,
 * style, or training split, and every row is marked trainingEligible=false.
 * The evaluator runs the built package and scores public parse -> intent ->
 * compile behavior. Development and calibration rows are diagnostic only;
 * only the reserved acceptance partition can produce a promotion decision.
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, relative, resolve, sep } from "node:path";
import { isDeepStrictEqual } from "node:util";

import type { CompileFailureCode } from "../../src/compiler/cron";
import type { ScheduleIntent } from "../../src/types/intent";
import {
  canonicalizeIntent,
  loadEngine,
  repoRoot,
  resolveRepo,
  riskCoverage,
  sha256Hex,
} from "./evaluate";
import type { Engine, RiskCoveragePoint, SelectiveOutcome } from "./evaluate";

export const HUMAN_CORPUS_VERSION = 1;
export const MIN_OVERALL_ACCURACY = 0.95;
export const MIN_FAMILY_ACCURACY = 0.9;
export const MIN_FAMILY_SUPPORT = 5;
export const MAX_SILENT_MEANING_CHANGE_RATE = 0.005;
export const MAX_NEGATIVE_FALSE_ACCEPTANCE_RATE = 0.01;
export const MAX_FAILURES_PER_CATEGORY = 3;
const ACCEPTANCE_LEDGER = "packages/cron/evaluation/consumed.json";

export type HumanPartition = "development" | "calibration" | "acceptance";

export interface HumanEvaluationRecord {
  version: 1;
  id: string;
  partition: HumanPartition;
  family: string;
  capabilities: string[];
  critical: boolean;
  trainingEligible: false;
  text: string;
  intent: ScheduleIntent | null;
  expected:
    | { parse: "reject" }
    | { supported: true; expression: string }
    | { supported: false; code: CompileFailureCode };
}

interface Counts {
  total: number;
  correct: number;
}

export interface HumanFailure {
  id: string;
  partition: HumanPartition;
  family: string;
  capabilities: string[];
  category: string;
  text: string;
  expected: unknown;
  actual: unknown;
}

export interface HumanEvaluationReport {
  version: 1;
  artifactSha256: string;
  corpusSha256: string;
  /** Stable semantic hash recorded in the consumption ledger after first use. */
  acceptancePartitionSha256: string;
  corpus: string;
  partition: HumanPartition;
  isolation: {
    checked: true;
    outsideTrainingData: true;
    incompatibleWithTrainingSchema: true;
    absentFromGeneratorSources: true;
  };
  total: number;
  correct: number;
  accuracy: number;
  byFamily: Record<string, Counts>;
  byCapability: Record<string, Counts>;
  failureCounts: Record<string, number>;
  failureCountsByFamily: Record<string, Record<string, number>>;
  failureCountsByCapability: Record<string, Record<string, number>>;
  failures: HumanFailure[];
  negativeFalseAccepts: number;
  negativeFalseAcceptanceRate: number;
  silentMeaningChanges: number;
  silentMeaningChangeRate: number;
  selectivePrediction: RiskCoveragePoint[];
  rawSelectivePrediction: RiskCoveragePoint[];
  calibrationThreshold: RiskCoveragePoint | null;
  sequenceEvidence: Record<string, SequenceEvidenceSummary>;
  promotion: {
    eligible: boolean;
    passed: boolean;
    failures: string[];
  };
}

export interface SequenceEvidenceSummary {
  count: number;
  tokenMargin: { min: number; mean: number; max: number } | null;
  boundaryMargin: { min: number; mean: number; max: number } | null;
  scheduleKindMargin: { min: number; mean: number; max: number } | null;
  scheduleKindAgreementRate: number;
  explicitScheduleEvidenceRate: number;
  lexiconAgreementRate: number | null;
  unconsumedScheduleLikeValues: number;
  semanticConstraintViolations: number;
  compilerInconsistent: number;
}

interface AcceptanceLedger {
  version: 1;
  acceptancePartitionSha256: string[];
}

/** Refuse to reuse acceptance wording after its first promotion decision. The
 * ledger is keyed by content rather than path, so copying the file elsewhere
 * cannot make consumed evidence blind again. */
export async function assertAcceptanceFresh(
  partition: HumanPartition,
  acceptancePartitionSha256: string,
): Promise<void> {
  if (partition !== "acceptance") return;

  const ledgerText = await readFile(resolveRepo(ACCEPTANCE_LEDGER), "utf8").catch(() => {
    throw new Error("acceptance consumption ledger is missing");
  });

  const ledger = JSON.parse(ledgerText) as AcceptanceLedger;

  if (
    ledger.version !== 1 ||
    !Array.isArray(ledger.acceptancePartitionSha256) ||
    !ledger.acceptancePartitionSha256.every((hash) => /^[a-f0-9]{64}$/.test(hash))
  ) {
    throw new Error("acceptance consumption ledger is malformed");
  }

  if (ledger.acceptancePartitionSha256.includes(acceptancePartitionSha256)) {
    throw new Error("acceptance corpus has already been consumed");
  }
}

type RawEvidence = NonNullable<NonNullable<ReturnType<Engine["decodeRaw"]>>["evidence"]>;

function numericSummary(values: readonly number[]) {
  if (values.length === 0) return null;

  return {
    min: Math.min(...values),
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    max: Math.max(...values),
  };
}

function sumEvidence(
  values: readonly { evidence: RawEvidence }[],
  select: (evidence: RawEvidence) => number,
): number {
  let total = 0;

  for (const entry of values) total += select(entry.evidence);

  return total;
}

function summarizeEvidence(
  entries: readonly { evidence: RawEvidence; correct: boolean; negative: boolean }[],
): Record<string, SequenceEvidenceSummary> {
  const groups = {
    all: entries,
    correct: entries.filter((entry) => entry.correct),
    incorrect: entries.filter((entry) => !entry.correct),
    positive: entries.filter((entry) => !entry.negative),
    negative: entries.filter((entry) => entry.negative),
  };

  return Object.fromEntries(
    Object.entries(groups).map(([name, values]) => {
      const lexicon = values.filter((entry) => entry.evidence.lexiconAgreement !== undefined);

      return [
        name,
        {
          count: values.length,
          tokenMargin: numericSummary(values.map((entry) => entry.evidence.tokenMargin)),
          boundaryMargin: numericSummary(values.map((entry) => entry.evidence.boundaryMargin)),
          scheduleKindMargin: numericSummary(
            values.flatMap((entry) =>
              entry.evidence.scheduleKindMargin === undefined
                ? []
                : [entry.evidence.scheduleKindMargin],
            ),
          ),
          scheduleKindAgreementRate:
            values.filter((entry) => entry.evidence.scheduleKindAgreement).length /
            Math.max(values.length, 1),
          explicitScheduleEvidenceRate:
            values.filter((entry) => entry.evidence.explicitScheduleEvidence).length /
            Math.max(values.length, 1),
          lexiconAgreementRate:
            lexicon.length === 0
              ? null
              : lexicon.filter((entry) => entry.evidence.lexiconAgreement).length / lexicon.length,
          unconsumedScheduleLikeValues: sumEvidence(
            values,
            (evidence) => evidence.unconsumedScheduleLikeValues,
          ),
          semanticConstraintViolations: sumEvidence(
            values,
            (evidence) => evidence.semanticConstraintViolations,
          ),
          compilerInconsistent: values.filter((entry) => !entry.evidence.compilerConsistent).length,
        },
      ];
    }),
  );
}

const GENERATOR_SOURCES = [
  "packages/cron/training/src/backward.ts",
  "packages/cron/training/src/featurize.ts",
  "packages/cron/training/src/generate.ts",
  "packages/cron/training/src/kinds.ts",
  "packages/cron/training/src/renderer.ts",
  "packages/cron/training/src/sampler.ts",
] as const;

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parseHumanCorpus(contents: string): HumanEvaluationRecord[] {
  const records: HumanEvaluationRecord[] = [];
  const ids = new Set<string>();

  for (const [index, source] of contents.split("\n").entries()) {
    const line = source.trim();

    if (line.length === 0) continue;
    const value = JSON.parse(line) as unknown;

    if (!isObject(value)) throw new Error(`human row ${index + 1}: expected object`);

    for (const forbidden of ["spans", "split", "frame", "style", "template"]) {
      if (forbidden in value) {
        throw new Error(`human row ${index + 1}: training field ${forbidden}`);
      }
    }

    if (
      value.version !== HUMAN_CORPUS_VERSION ||
      typeof value.id !== "string" ||
      !["development", "calibration", "acceptance"].includes(String(value.partition)) ||
      typeof value.family !== "string" ||
      !Array.isArray(value.capabilities) ||
      value.capabilities.length === 0 ||
      !value.capabilities.every((item) => typeof item === "string" && item.length > 0) ||
      typeof value.critical !== "boolean" ||
      value.trainingEligible !== false ||
      typeof value.text !== "string" ||
      value.text.trim().length === 0 ||
      !(value.intent === null || isObject(value.intent)) ||
      !isObject(value.expected)
    ) {
      throw new Error(`human row ${index + 1}: invalid schema`);
    }

    if (ids.has(value.id)) throw new Error(`duplicate human id: ${value.id}`);
    ids.add(value.id);

    if (value.intent === null && value.expected.parse !== "reject") {
      throw new Error(`${value.id}: a negative must require rejection`);
    }

    if (value.intent !== null && value.expected.parse === "reject") {
      throw new Error(`${value.id}: a schedule needs a compiler expectation`);
    }

    records.push(value as unknown as HumanEvaluationRecord);
  }

  if (records.length === 0) throw new Error("human corpus is empty");

  return records;
}

/** Fail closed if the acceptance data looks consumable by training or is
 * referenced from a generator source. Exact sentences must also not be copied
 * into generator source code. */
export async function assertHumanIsolation(
  corpusPath: string,
  records: readonly HumanEvaluationRecord[],
): Promise<HumanEvaluationReport["isolation"]> {
  const root = repoRoot();
  const corpus = resolve(corpusPath);
  const trainingData = resolve(root, "packages/cron/training/data");

  if (corpus === trainingData || corpus.startsWith(`${trainingData}${sep}`)) {
    throw new Error("human evaluation corpus must be outside training/data");
  }

  const corpusReference = relative(root, corpus).replaceAll("\\", "/");

  for (const sourceName of GENERATOR_SOURCES) {
    const source = await readFile(resolve(root, sourceName), "utf8");

    if (source.includes(corpusReference)) {
      throw new Error(`human corpus referenced by generator: ${sourceName}`);
    }

    for (const record of records) {
      if (source.includes(record.text)) {
        throw new Error(`human wording copied into generator: ${record.id}`);
      }
    }
  }

  return {
    checked: true,
    outsideTrainingData: true,
    incompatibleWithTrainingSchema: true,
    absentFromGeneratorSources: true,
  };
}

function addCount(target: Record<string, Counts>, key: string, correct: boolean): void {
  const counts = (target[key] ??= { total: 0, correct: 0 });

  counts.total += 1;
  counts.correct += Number(correct);
}

function addFailureCount(
  target: Record<string, Record<string, number>>,
  group: string,
  category: string,
): void {
  const counts = (target[group] ??= {});

  counts[category] = (counts[category] ?? 0) + 1;
}

/** Pick maximum safe coverage on calibration data; never optimize this on the
 * development or reserved acceptance partitions. */
export function selectCalibrationThreshold(
  points: readonly RiskCoveragePoint[],
): RiskCoveragePoint | null {
  return (
    points
      .filter(
        (point) =>
          point.risk <= MAX_SILENT_MEANING_CHANGE_RATE &&
          point.negativeFalseAcceptanceRate <= MAX_NEGATIVE_FALSE_ACCEPTANCE_RATE,
      )
      .sort(
        (left, right) => right.coverage - left.coverage || left.threshold - right.threshold,
      )[0] ?? null
  );
}

function differentIntentFields(expected: ScheduleIntent, actual: ScheduleIntent): string {
  const left = canonicalizeIntent(expected);
  const right = canonicalizeIntent(actual);

  return [...new Set([...Object.keys(left), ...Object.keys(right)])]
    .filter(
      (key) =>
        !isDeepStrictEqual(left[key as keyof ScheduleIntent], right[key as keyof ScheduleIntent]),
    )
    .sort()
    .join("+");
}

export interface HumanEvaluateOptions {
  dist: string;
  corpus: string;
  out: string;
  partition: HumanPartition;
  /** Optional exported candidate module; never changes the active artifact. */
  weights?: string;
}

export async function evaluateHumanCorpus(
  options: HumanEvaluateOptions,
): Promise<HumanEvaluationReport> {
  const corpus = resolveRepo(options.corpus);
  const contents = await readFile(corpus, "utf8");
  const corpusSha256 = sha256Hex(contents);
  const allRecords = parseHumanCorpus(contents);

  const acceptancePartitionSha256 = sha256Hex(
    `${allRecords
      .filter((record) => record.partition === "acceptance")
      .map((record) => JSON.stringify(record))
      .join("\n")}\n`,
  );

  await assertAcceptanceFresh(options.partition, acceptancePartitionSha256);
  const isolation = await assertHumanIsolation(corpus, allRecords);
  const records = allRecords.filter((item) => item.partition === options.partition);

  if (records.length === 0) throw new Error(`human corpus has no ${options.partition} records`);
  const engine = await loadEngine(options.dist, options.weights);
  const byFamily: Record<string, Counts> = {};
  const byCapability: Record<string, Counts> = {};
  const failureCounts: Record<string, number> = {};
  const failureCountsByFamily: Record<string, Record<string, number>> = {};
  const failureCountsByCapability: Record<string, Record<string, number>> = {};
  const failures: HumanFailure[] = [];
  const incorrectIds = new Set<string>();
  let correctTotal = 0;
  let negativeTotal = 0;
  let negativeFalseAccepts = 0;
  let silentMeaningChanges = 0;
  const selectiveOutcomes: SelectiveOutcome[] = [];
  const rawSelectiveOutcomes: SelectiveOutcome[] = [];

  const rawEvidence: Array<{
    evidence: RawEvidence;
    correct: boolean;
    negative: boolean;
  }> = [];

  try {
    // Lint every partition's gold contract without parsing reserved acceptance
    // wording. This catches fixture mistakes while keeping acceptance inputs
    // untouched until the final gate.
    for (const item of allRecords) {
      if (item.intent !== null) {
        if ("parse" in item.expected) {
          throw new Error(`${item.id}: schedule has a rejection expectation`);
        }

        const goldCompiled = engine.compile(canonicalizeIntent(item.intent));

        const validGold = item.expected.supported
          ? goldCompiled.supported && goldCompiled.expression === item.expected.expression
          : !goldCompiled.supported && goldCompiled.code === item.expected.code;

        if (!validGold) {
          throw new Error(
            `${item.id}: invalid compiler expectation ${JSON.stringify(item.expected)} ` +
              `for ${JSON.stringify(goldCompiled)}`,
          );
        }
      }
    }

    for (const item of records) {
      const parsed = await engine.parse(item.text);
      let raw: ReturnType<typeof engine.decodeRaw> | null = null;

      try {
        raw = engine.decodeRaw(item.text);
      } catch {
        // Decoder rejection is a safe abstention in the raw policy curve.
      }

      let correct = false;
      let category = "correct";
      let actual: unknown = parsed;

      if (item.intent === null) {
        negativeTotal += 1;
        correct = !parsed.ok;

        if (!correct) {
          category = "negative-false-accept";
          negativeFalseAccepts += 1;
          silentMeaningChanges += 1;
        }
      } else if (!parsed.ok) {
        category = `parse:${parsed.code}`;
      } else {
        const expectedIntent = canonicalizeIntent(item.intent);
        const actualIntent = canonicalizeIntent(parsed.intent);

        if (!isDeepStrictEqual(expectedIntent, actualIntent)) {
          category = `intent:${differentIntentFields(expectedIntent, actualIntent)}`;
          silentMeaningChanges += 1;
          actual = actualIntent;
        } else {
          const compiled = engine.compile(parsed.intent);

          actual = { intent: actualIntent, compiled };

          if ("parse" in item.expected) {
            throw new Error(`${item.id}: impossible positive rejection expectation`);
          } else if (item.expected.supported) {
            correct = compiled.supported && compiled.expression === item.expected.expression;
            if (!correct) category = "compile:expression";
          } else {
            correct = !compiled.supported && compiled.code === item.expected.code;
            if (!correct) category = "compile:failure-code";
          }
        }
      }

      correctTotal += Number(correct);

      selectiveOutcomes.push({
        accepted: parsed.ok,
        correct,
        confidence: parsed.confidence,
        negative: item.intent === null,
      });

      const rawCorrect =
        item.intent === null
          ? raw === null
          : raw !== null &&
            isDeepStrictEqual(canonicalizeIntent(raw.intent), canonicalizeIntent(item.intent));

      rawSelectiveOutcomes.push({
        accepted: raw !== null,
        correct: rawCorrect,
        confidence: raw?.confidence ?? 0,
        negative: item.intent === null,
      });

      if (raw?.evidence !== undefined) {
        rawEvidence.push({
          evidence: raw.evidence,
          correct: rawCorrect,
          negative: item.intent === null,
        });
      }

      addCount(byFamily, item.family, correct);
      for (const capability of item.capabilities) addCount(byCapability, capability, correct);

      if (!correct) {
        incorrectIds.add(item.id);
        const count = (failureCounts[category] ?? 0) + 1;

        failureCounts[category] = count;
        addFailureCount(failureCountsByFamily, item.family, category);

        for (const capability of item.capabilities) {
          addFailureCount(failureCountsByCapability, capability, category);
        }

        if (count <= MAX_FAILURES_PER_CATEGORY) {
          failures.push({
            id: item.id,
            partition: item.partition,
            family: item.family,
            capabilities: item.capabilities,
            category,
            text: item.text,
            expected: item.intent === null ? item.expected : item.intent,
            actual,
          });
        }
      }
    }
  } finally {
    await engine.dispose();
  }

  const total = records.length;
  const promotionFailures: string[] = [];
  const accuracy = correctTotal / total;
  const silentMeaningChangeRate = silentMeaningChanges / total;
  const negativeFalseAcceptanceRate = negativeFalseAccepts / Math.max(negativeTotal, 1);
  const selectivePrediction = riskCoverage(selectiveOutcomes);
  const rawSelectivePrediction = riskCoverage(rawSelectiveOutcomes);

  const calibrationThreshold =
    options.partition === "calibration" ? selectCalibrationThreshold(rawSelectivePrediction) : null;

  if (accuracy < MIN_OVERALL_ACCURACY) {
    promotionFailures.push(`overall accuracy below ${MIN_OVERALL_ACCURACY}`);
  }

  for (const [family, counts] of Object.entries(byFamily)) {
    if (counts.total < MIN_FAMILY_SUPPORT) continue;

    if (counts.correct / counts.total < MIN_FAMILY_ACCURACY) {
      promotionFailures.push(`${family}: accuracy below ${MIN_FAMILY_ACCURACY}`);
    }
  }

  if (silentMeaningChangeRate > MAX_SILENT_MEANING_CHANGE_RATE) {
    promotionFailures.push(`silent meaning-change rate above ${MAX_SILENT_MEANING_CHANGE_RATE}`);
  }

  if (negativeFalseAcceptanceRate > MAX_NEGATIVE_FALSE_ACCEPTANCE_RATE) {
    promotionFailures.push(
      `negative false-acceptance rate above ${MAX_NEGATIVE_FALSE_ACCEPTANCE_RATE}`,
    );
  }

  for (const item of records.filter((record) => record.critical)) {
    if (incorrectIds.has(item.id)) promotionFailures.push(`critical failure: ${item.id}`);
  }

  const eligible = options.partition === "acceptance";

  return {
    version: HUMAN_CORPUS_VERSION,
    artifactSha256: engine.artifactSha256,
    corpusSha256,
    acceptancePartitionSha256,
    corpus,
    partition: options.partition,
    isolation,
    total,
    correct: correctTotal,
    accuracy,
    byFamily,
    byCapability,
    failureCounts,
    failureCountsByFamily,
    failureCountsByCapability,
    failures,
    negativeFalseAccepts,
    negativeFalseAcceptanceRate,
    silentMeaningChanges,
    silentMeaningChangeRate,
    selectivePrediction,
    rawSelectivePrediction,
    calibrationThreshold,
    sequenceEvidence: summarizeEvidence(rawEvidence),
    promotion: {
      eligible,
      passed: eligible && promotionFailures.length === 0,
      failures: promotionFailures,
    },
  };
}

function readOption(argv: readonly string[], index: number): string {
  const value = argv[index + 1];

  if (value === undefined) throw new Error(`missing value for ${argv[index]}`);

  return value;
}

export function parseArgs(argv: readonly string[]): HumanEvaluateOptions {
  const options: HumanEvaluateOptions = {
    dist: "packages/cron/dist",
    corpus: "packages/cron/evaluation/human-v1.jsonl",
    out: "packages/cron/training/results/human-v1-development.json",
    partition: "development",
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]!;

    if (arg === "--dist") {
      options.dist = readOption(argv, index++);
    } else if (arg === "--weights") {
      options.weights = readOption(argv, index++);
    } else if (arg === "--corpus") {
      options.corpus = readOption(argv, index++);
    } else if (arg === "--out") {
      options.out = readOption(argv, index++);
    } else if (arg === "--partition") {
      const partition = readOption(argv, index++);

      if (!["development", "calibration", "acceptance"].includes(partition)) {
        throw new Error(`invalid human partition: ${partition}`);
      }

      options.partition = partition as HumanPartition;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }

  return options;
}

if (import.meta.main) {
  const options = parseArgs(process.argv.slice(2));
  const report = await evaluateHumanCorpus(options);
  const out = resolveRepo(options.out);

  await mkdir(dirname(out), { recursive: true });
  await writeFile(out, `${JSON.stringify(report, null, 2)}\n`);

  process.stdout.write(
    `${report.partition}: ${report.correct}/${report.total} ` +
      `(${report.accuracy.toFixed(3)}), silent=${report.silentMeaningChanges}, ` +
      `negative-false-accept=${report.negativeFalseAccepts}\nreport -> ${out}\n`,
  );
}
