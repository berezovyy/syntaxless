/**
 * End-to-end evaluation for the versioned, human-style acceptance corpus.
 *
 * This corpus is intentionally not a RenderedExample: it has no spans, frame,
 * style, or training split, and every row is marked trainingEligible=false.
 * The evaluator runs the built package and scores public parse -> intent ->
 * compile behavior. Development and calibration rows are diagnostic only;
 * only a fresh acceptance partition can produce a promotion decision.
 * `--partition regression` scores consumed `human-v*` as development evidence.
 * Promotion uses overlay-off against the existing bars; overlay-on is
 * diagnostic. After `assertAcceptanceFresh` records the hash, further 100%
 * on that file is regression, not promotion. Using those failures to write
 * decoder regex is forbidden; using them to name a generator kind is the loop.
 */
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import { isDeepStrictEqual } from "node:util";

import type { CompileFailureCode } from "../../src/compiler/cron";
import type { Label } from "../../src/model/labels";
import { SPEECH_ACTS } from "../../src/model/speech-acts";
import type { ScheduleIntent } from "../../src/types/intent";
import {
  addAttribution,
  attributeMiss,
  emptyAttributionReport,
  writeAttributionFile,
} from "./attribution";
import type { AttributionReport } from "./attribution";
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
export type HumanEvaluatePartition = HumanPartition | "regression";

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
  partition: HumanEvaluatePartition;
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
  /**
   * Overlay-off scores on the same artifact. `promotion.passed` is this block
   * against the existing bars; overlay-on remains diagnostic.
   */
  overlayOff: {
    total: number;
    correct: number;
    accuracy: number;
    negativeFalseAccepts: number;
    negativeFalseAcceptanceRate: number;
    silentMeaningChanges: number;
    silentMeaningChangeRate: number;
    failureCounts: Record<string, number>;
    failures: HumanFailure[];
  };
  /**
   * Calibration-only signal for phase 10. Never true for development.
   * Acceptance uses `promotion.passed` instead. Does not move confidence floors.
   */
  overlayOffEligibleForDefault: boolean;
  /** Wrong intent emitted with ok:true; excludes negative false-accepts (H-silent). */
  positiveSilentMeaningChanges: number;
  attribution: AttributionReport;
}

export interface HumanOverlayRates {
  accuracy: number;
  silentMeaningChangeRate: number;
  negativeFalseAcceptanceRate: number;
  byFamily: Record<string, Counts>;
  criticalFailures: readonly string[];
}

/**
 * Pure promotion arithmetic. Overlay-off must meet the existing bars and must
 * not be worse than overlay-on on silent-change or false-accept. Overlay-on
 * 1.0 cannot rescue overlay-off 0.7.
 */
export function decideHumanPromotion(
  partition: HumanEvaluatePartition,
  overlayOn: HumanOverlayRates,
  overlayOff: HumanOverlayRates | undefined,
): {
  eligible: boolean;
  passed: boolean;
  failures: string[];
  overlayOffEligibleForDefault: boolean;
} {
  const eligible = partition === "acceptance";

  if (overlayOff === undefined) {
    return {
      eligible,
      passed: false,
      failures: ["missing overlay-off"],
      overlayOffEligibleForDefault: false,
    };
  }

  const failures: string[] = [];

  if (overlayOff.accuracy < MIN_OVERALL_ACCURACY) {
    failures.push(`overlay-off overall accuracy below ${MIN_OVERALL_ACCURACY}`);
  }

  for (const [family, counts] of Object.entries(overlayOff.byFamily)) {
    if (counts.total < MIN_FAMILY_SUPPORT) continue;

    if (counts.correct / counts.total < MIN_FAMILY_ACCURACY) {
      failures.push(`overlay-off ${family}: accuracy below ${MIN_FAMILY_ACCURACY}`);
    }
  }

  if (overlayOff.silentMeaningChangeRate > MAX_SILENT_MEANING_CHANGE_RATE) {
    failures.push(`overlay-off silent meaning-change rate above ${MAX_SILENT_MEANING_CHANGE_RATE}`);
  }

  if (overlayOff.negativeFalseAcceptanceRate > MAX_NEGATIVE_FALSE_ACCEPTANCE_RATE) {
    failures.push(
      `overlay-off negative false-acceptance rate above ${MAX_NEGATIVE_FALSE_ACCEPTANCE_RATE}`,
    );
  }

  for (const id of overlayOff.criticalFailures) {
    failures.push(`overlay-off critical failure: ${id}`);
  }

  if (overlayOff.silentMeaningChangeRate > overlayOn.silentMeaningChangeRate) {
    failures.push("overlay-off silent meaning-change worse than overlay-on");
  }

  if (overlayOff.negativeFalseAcceptanceRate > overlayOn.negativeFalseAcceptanceRate) {
    failures.push("overlay-off negative false-accept worse than overlay-on");
  }

  const overlayOffMeetsBars = failures.length === 0;

  return {
    eligible,
    passed: eligible && overlayOffMeetsBars,
    failures,
    overlayOffEligibleForDefault: partition === "calibration" && overlayOffMeetsBars,
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
  partition: HumanEvaluatePartition,
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
  "packages/cron/training/src/correction.ts",
  "packages/cron/training/src/discourse.ts",
  "packages/cron/training/src/featurize.ts",
  "packages/cron/training/src/generate.ts",
  "packages/cron/training/src/kinds.ts",
  "packages/cron/training/src/renderer.ts",
  "packages/cron/training/src/sampler.ts",
] as const;

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const INTENT_WEEKDAYS = new Set([
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday",
]);

const INTENT_UNITS = new Set(["minute", "hour", "day", "week", "month"]);
const INTENT_ORDINALS = new Set(["first", "second", "third", "fourth", "last"]);
const INTENT_DAY_TYPES = new Set(["last_business_day", "last_day_of_month"]);

function hasOnlyKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const allowed = new Set(keys);

  return Object.keys(value).every((key) => allowed.has(key));
}

/** Runtime-check the full public AST instead of trusting a TypeScript cast.
 * This keeps malformed gold (especially misspelled enum members) from passing
 * merely because the compiler rejects both the malformed and intended shape. */
function assertHumanIntent(value: Record<string, unknown>, id: string): void {
  const fail = () => {
    throw new Error(`${id}: invalid ScheduleIntent gold`);
  };

  if (
    !hasOnlyKeys(value, [
      "frequency",
      "time",
      "weekdays",
      "dayOfMonth",
      "ordinalWeekday",
      "day",
      "bounds",
      "exceptions",
    ]) ||
    !isObject(value.frequency) ||
    !hasOnlyKeys(value.frequency, ["unit", "interval"]) ||
    !INTENT_UNITS.has(String(value.frequency.unit)) ||
    !Number.isInteger(value.frequency.interval) ||
    Number(value.frequency.interval) < 1
  ) {
    fail();
  }

  if (value.time !== undefined) {
    if (
      !isObject(value.time) ||
      !hasOnlyKeys(value.time, ["hour", "minute"]) ||
      !Number.isInteger(value.time.hour) ||
      Number(value.time.hour) < 0 ||
      Number(value.time.hour) > 23 ||
      !Number.isInteger(value.time.minute) ||
      Number(value.time.minute) < 0 ||
      Number(value.time.minute) > 59
    ) {
      fail();
    }
  }

  if (
    value.weekdays !== undefined &&
    (!Array.isArray(value.weekdays) ||
      value.weekdays.length === 0 ||
      !value.weekdays.every((day) => INTENT_WEEKDAYS.has(String(day))))
  ) {
    fail();
  }

  if (
    value.dayOfMonth !== undefined &&
    (!Number.isInteger(value.dayOfMonth) ||
      Number(value.dayOfMonth) < 1 ||
      Number(value.dayOfMonth) > 31)
  ) {
    fail();
  }

  if (value.ordinalWeekday !== undefined) {
    if (
      !isObject(value.ordinalWeekday) ||
      !hasOnlyKeys(value.ordinalWeekday, ["ordinal", "weekday"]) ||
      !INTENT_ORDINALS.has(String(value.ordinalWeekday.ordinal)) ||
      !INTENT_WEEKDAYS.has(String(value.ordinalWeekday.weekday))
    ) {
      fail();
    }
  }

  if (value.day !== undefined) {
    if (
      !isObject(value.day) ||
      !hasOnlyKeys(value.day, ["type"]) ||
      !INTENT_DAY_TYPES.has(String(value.day.type))
    ) {
      fail();
    }
  }

  const validDateObject = (candidate: unknown): boolean =>
    isObject(candidate) &&
    hasOnlyKeys(candidate, ["start", "end"]) &&
    [candidate.start, candidate.end].every(
      (date) => date === undefined || /^\d{4}-\d{2}-\d{2}$/.test(String(date)),
    );

  if (value.bounds !== undefined && !validDateObject(value.bounds)) fail();

  if (value.exceptions !== undefined) {
    if (
      !isObject(value.exceptions) ||
      !hasOnlyKeys(value.exceptions, ["dates", "weekdays"]) ||
      (value.exceptions.dates !== undefined &&
        (!Array.isArray(value.exceptions.dates) ||
          !value.exceptions.dates.every((date) => /^\d{4}-\d{2}-\d{2}$/.test(String(date))))) ||
      (value.exceptions.weekdays !== undefined &&
        (!Array.isArray(value.exceptions.weekdays) ||
          !value.exceptions.weekdays.every((day) => INTENT_WEEKDAYS.has(String(day)))))
    ) {
      fail();
    }
  }
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

    if (isObject(value.intent)) assertHumanIntent(value.intent, value.id);
    records.push(value as unknown as HumanEvaluationRecord);
  }

  if (records.length === 0) throw new Error("human corpus is empty");

  return records;
}

const HUMAN_V_FILE = /^human-v\d+\.jsonl$/;

/** Consumed `human-v*` files; calibration-only corpora are not in this union. */
export async function regressionCorpusPaths(): Promise<string[]> {
  const dir = resolveRepo("packages/cron/evaluation");

  const names = (await readdir(dir))
    .filter((name) => HUMAN_V_FILE.test(name))
    .sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));

  return names.map((name) => join(dir, name));
}

/** Union of consumed human-v* records, scored as development evidence. */
export async function loadRegressionRecords(): Promise<{
  records: HumanEvaluationRecord[];
  corpus: string;
  corpusSha256: string;
  contents: string;
}> {
  const paths = await regressionCorpusPaths();

  if (paths.length === 0) throw new Error("no human-v*.jsonl files for regression");
  const records: HumanEvaluationRecord[] = [];
  const chunks: string[] = [];

  for (const path of paths) {
    const contents = await readFile(path, "utf8");

    chunks.push(contents);
    records.push(...parseHumanCorpus(contents));
  }

  const joined = chunks.join("\n");

  return {
    records,
    corpus: paths.join(","),
    corpusSha256: sha256Hex(joined),
    contents: joined,
  };
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
  partition: HumanEvaluatePartition;
  /** Optional exported candidate module; never changes the active artifact. */
  weights?: string;
}

interface HumanOverlayScore {
  correct: number;
  negativeTotal: number;
  negativeFalseAccepts: number;
  silentMeaningChanges: number;
  byFamily: Record<string, Counts>;
  byCapability: Record<string, Counts>;
  failureCounts: Record<string, number>;
  failureCountsByFamily: Record<string, Record<string, number>>;
  failureCountsByCapability: Record<string, Record<string, number>>;
  failures: HumanFailure[];
  incorrectIds: Set<string>;
  selectiveOutcomes: SelectiveOutcome[];
  rawSelectiveOutcomes: SelectiveOutcome[];
  rawEvidence: Array<{
    evidence: RawEvidence;
    correct: boolean;
    negative: boolean;
  }>;
  positiveSilentMeaningChanges: number;
  attribution: AttributionReport;
}

async function scoreHumanOverlay(
  engine: Engine,
  records: readonly HumanEvaluationRecord[],
  overlay: "on" | "off",
): Promise<HumanOverlayScore> {
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
  let positiveSilentMeaningChanges = 0;
  const attribution = emptyAttributionReport();
  const selectiveOutcomes: SelectiveOutcome[] = [];
  const rawSelectiveOutcomes: SelectiveOutcome[] = [];
  const rawEvidence: HumanOverlayScore["rawEvidence"] = [];

  for (const item of records) {
    const parsed =
      overlay === "on"
        ? await engine.parse(item.text)
        : await engine.parse(item.text, { overlay: "off" });

    let raw: ReturnType<typeof engine.decodeRaw> | null = null;

    try {
      raw = overlay === "on" ? engine.decodeRaw(item.text) : engine.decodeRaw(item.text, "off");
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
        positiveSilentMeaningChanges += 1;
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

    if (!correct && overlay === "off") {
      const tokens = engine.tokenize(item.text);
      const predictions = engine.inferTokens(tokens);
      const predictedLabels: Label[] = [];

      for (let index = 0; index < tokens.length; index += 1) {
        if (tokens[index]!.kind === 3) continue;
        predictedLabels.push(engine.labelName(predictions.labels[index]!) as Label);
      }

      const predictedSpeechAct =
        predictions.speechAct === undefined ? undefined : SPEECH_ACTS[predictions.speechAct];

      addAttribution(attribution, {
        id: item.id,
        source: "human",
        family: item.family,
        bucket: attributeMiss({
          text: item.text,
          expectedIntent: item.intent,
          accepted: parsed.ok,
          actualIntent: parsed.ok ? parsed.intent : null,
          predictedLabels,
          ...(predictedSpeechAct === undefined ? {} : { predictedSpeechAct }),
          predictedOracle:
            raw === null
              ? { ok: false, error: "raw-decode-null" }
              : { ok: true, intent: raw.intent },
        }),
      });
    }

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

  return {
    correct: correctTotal,
    negativeTotal,
    negativeFalseAccepts,
    silentMeaningChanges,
    byFamily,
    byCapability,
    failureCounts,
    failureCountsByFamily,
    failureCountsByCapability,
    failures,
    incorrectIds,
    selectiveOutcomes,
    rawSelectiveOutcomes,
    rawEvidence,
    positiveSilentMeaningChanges,
    attribution,
  };
}

export async function evaluateHumanCorpus(
  options: HumanEvaluateOptions,
): Promise<HumanEvaluationReport> {
  const regression = options.partition === "regression" ? await loadRegressionRecords() : undefined;
  const corpus = regression?.corpus ?? resolveRepo(options.corpus);
  const contents = regression?.contents ?? (await readFile(resolveRepo(options.corpus), "utf8"));
  const corpusSha256 = regression?.corpusSha256 ?? sha256Hex(contents);
  const allRecords = regression?.records ?? parseHumanCorpus(contents);

  const acceptancePartitionSha256 = sha256Hex(
    `${allRecords
      .filter((record) => record.partition === "acceptance")
      .map((record) => JSON.stringify(record))
      .join("\n")}\n`,
  );

  await assertAcceptanceFresh(options.partition, acceptancePartitionSha256);

  const isolationPath =
    options.partition === "regression"
      ? (await regressionCorpusPaths())[0]!
      : resolveRepo(options.corpus);

  const isolation = await assertHumanIsolation(isolationPath, allRecords);

  const records =
    options.partition === "regression"
      ? allRecords
      : allRecords.filter((item) => item.partition === options.partition);

  if (records.length === 0) throw new Error(`human corpus has no ${options.partition} records`);
  const engine = await loadEngine(options.dist, options.weights);
  let overlayOn!: HumanOverlayScore;
  let overlayOff!: HumanOverlayScore;

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

    overlayOn = await scoreHumanOverlay(engine, records, "on");
    overlayOff = await scoreHumanOverlay(engine, records, "off");
  } finally {
    await engine.dispose();
  }

  const total = records.length;
  const accuracy = overlayOn.correct / total;
  const silentMeaningChangeRate = overlayOn.silentMeaningChanges / total;

  const negativeFalseAcceptanceRate =
    overlayOn.negativeFalseAccepts / Math.max(overlayOn.negativeTotal, 1);

  const overlayOffSilent = overlayOff.silentMeaningChanges / total;

  const overlayOffNegative =
    overlayOff.negativeFalseAccepts / Math.max(overlayOff.negativeTotal, 1);

  const selectivePrediction = riskCoverage(overlayOn.selectiveOutcomes);
  const rawSelectivePrediction = riskCoverage(overlayOn.rawSelectiveOutcomes);

  const calibrationThreshold =
    options.partition === "calibration" ? selectCalibrationThreshold(rawSelectivePrediction) : null;

  const promotion = decideHumanPromotion(
    options.partition,
    {
      accuracy,
      silentMeaningChangeRate,
      negativeFalseAcceptanceRate,
      byFamily: overlayOn.byFamily,
      criticalFailures: records
        .filter((item) => item.critical && overlayOn.incorrectIds.has(item.id))
        .map((item) => item.id),
    },
    {
      accuracy: overlayOff.correct / total,
      silentMeaningChangeRate: overlayOffSilent,
      negativeFalseAcceptanceRate: overlayOffNegative,
      byFamily: overlayOff.byFamily,
      criticalFailures: records
        .filter((item) => item.critical && overlayOff.incorrectIds.has(item.id))
        .map((item) => item.id),
    },
  );

  return {
    version: HUMAN_CORPUS_VERSION,
    artifactSha256: engine.artifactSha256,
    corpusSha256,
    acceptancePartitionSha256,
    corpus,
    partition: options.partition,
    isolation,
    total,
    correct: overlayOn.correct,
    accuracy,
    byFamily: overlayOn.byFamily,
    byCapability: overlayOn.byCapability,
    failureCounts: overlayOn.failureCounts,
    failureCountsByFamily: overlayOn.failureCountsByFamily,
    failureCountsByCapability: overlayOn.failureCountsByCapability,
    failures: overlayOn.failures,
    negativeFalseAccepts: overlayOn.negativeFalseAccepts,
    negativeFalseAcceptanceRate,
    silentMeaningChanges: overlayOn.silentMeaningChanges,
    silentMeaningChangeRate,
    selectivePrediction,
    rawSelectivePrediction,
    calibrationThreshold,
    sequenceEvidence: summarizeEvidence(overlayOn.rawEvidence),
    promotion: {
      eligible: promotion.eligible,
      passed: promotion.passed,
      failures: promotion.failures,
    },
    overlayOff: {
      total,
      correct: overlayOff.correct,
      accuracy: overlayOff.correct / total,
      negativeFalseAccepts: overlayOff.negativeFalseAccepts,
      negativeFalseAcceptanceRate:
        overlayOff.negativeFalseAccepts / Math.max(overlayOff.negativeTotal, 1),
      silentMeaningChanges: overlayOff.silentMeaningChanges,
      silentMeaningChangeRate: overlayOff.silentMeaningChanges / total,
      failureCounts: overlayOff.failureCounts,
      failures: overlayOff.failures,
    },
    overlayOffEligibleForDefault: promotion.overlayOffEligibleForDefault,
    positiveSilentMeaningChanges: overlayOff.positiveSilentMeaningChanges,
    attribution: overlayOff.attribution,
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

  let outSet = false;

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
      outSet = true;
    } else if (arg === "--partition") {
      const partition = readOption(argv, index++);

      if (!["development", "calibration", "acceptance", "regression"].includes(partition)) {
        throw new Error(`invalid human partition: ${partition}`);
      }

      options.partition = partition as HumanEvaluatePartition;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }

  if (options.partition === "regression" && !outSet) {
    options.out = "packages/cron/training/results/human-regression.json";
  }

  return options;
}

if (import.meta.main) {
  const options = parseArgs(process.argv.slice(2));
  const report = await evaluateHumanCorpus(options);
  const out = resolveRepo(options.out);

  await mkdir(dirname(out), { recursive: true });
  await writeFile(out, `${JSON.stringify(report, null, 2)}\n`);

  await writeAttributionFile(
    resolveRepo("packages/cron/training/results/attribution.json"),
    "human",
    report.attribution,
  );

  process.stdout.write(
    `${report.partition}: overlay-on: ${report.correct}/${report.total} ` +
      `(${report.accuracy.toFixed(3)}) overlay-off: ${report.overlayOff.correct}/` +
      `${report.overlayOff.total} (${report.overlayOff.accuracy.toFixed(3)}), ` +
      `silent=${report.silentMeaningChanges}, ` +
      `negative-false-accept=${report.negativeFalseAccepts}\nreport -> ${out}\n`,
  );
}
