/**
 * Promotion gate: candidate vs baseline, decoded from shipped weight artifacts.
 *
 * Both sides load the *built* `packages/cron/dist/` runtime against their own
 * weight module and are re-scored in this one process on the same reserved-
 * carrier corpus. No stored score is read. A candidate is accepted only with a
 * strict improvement on exact-intent sequences and no per-family regression
 * beyond a two-proportion z=1.96 tolerance. A rejected candidate exits non-zero;
 * that exit is the delivery, not a bug to paper over.
 *
 * The reserved carriers are the same {@link RESERVED_PHRASES} the held-out
 * renderer uses. Before scoring, their fingerprints are recomputed and checked
 * against the Phase 08 training split, so reserved wording cannot have leaked
 * into training.
 */
import { existsSync, readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

import { RESERVED_PHRASES } from "./background";
import { generatorSha256 } from "./backward";
import {
  assertSameArtifact,
  DEFAULT_SCOPE,
  evaluateCorpus,
  loadEngine,
  resolveRepo,
  sha256Hex,
} from "./evaluate";
import type { ContractReport, Counts, Engine, EvaluationResult, Metric } from "./evaluate";
import { faAtCovMeetsTarget } from "./fa-at-cov";
import { coarseText } from "./fingerprint";

/** Two-proportion guard constants, fixed by the plan. */
export const GUARD_Z = 1.96;
export const GUARD_MINIMUM_SUPPORT = 30;
/** Absolute floors prevent a relative improvement from promoting a weak model. */
export const MIN_OVERALL_ACCURACY = 0.95;
export const MIN_FAMILY_ACCURACY = 0.9;
export const MIN_TYPO_INVARIANCE = 0.95;
export const MAX_SILENT_MEANING_CHANGE_RATE = 0.005;
export const MAX_NEGATIVE_FALSE_ACCEPTANCE_RATE = 0.01;

/** The overall promotion criterion and the terse-family bare guard. */
export const CRITERION = "reserved-carrier-exact-intent";
/** STACK-PLAN v3: G2 is a floor; promotion quotes H, EXEC, FA@cov, O. */
export const PROMOTION_REQUIRED = ["H", "EXEC", "FA@cov", "O"] as const;
export const BARE_FAMILY = "terse";

export const CRITERION_DESCRIPTION =
  "Strict improvement on reserved-carrier exact-intent sequences with no " +
  "per-family regression beyond a two-proportion z=1.96 tolerance. Both sides " +
  "are decoded from their wire artifacts and re-scored in one process; stored " +
  "scores are never read.";

export const DEFAULTS = {
  dist: "packages/cron/dist",
  candidate: "packages/cron/training/candidate/weights.gen.ts",
  baseline: "packages/cron/src/model/weights.gen.ts",
  corpus: "packages/cron/training/data/gold/reserved-carriers.jsonl",
  trainSplit: "packages/cron/training/data/synth/kinds.jsonl.fingerprints.json",
  out: "packages/cron/training/results/gate.json",
} as const;

export interface ProportionGuardResult {
  passed: boolean;
  reason: string | null;
  support: number;
  delta: number;
  tolerance: number;
}

/**
 * Add-one smoothed two-proportion z guard. `pc`/`pb` are the smoothed error
 * rates, `delta` is the baseline-minus-candidate correct proportion, and
 * `tolerance` is the 95% band. Fails on support mismatch, support below 30, or a
 * regression larger than the band; a small regression inside the band passes.
 */
export function proportionGuard(candidate: Counts, baseline: Counts): ProportionGuardResult {
  const support = candidate.total;

  if (support !== baseline.total) {
    return {
      passed: false,
      reason: "support mismatch",
      support,
      delta: 0,
      tolerance: 0,
    };
  }

  if (support < GUARD_MINIMUM_SUPPORT) {
    return {
      passed: false,
      reason: "insufficient support",
      support,
      delta: 0,
      tolerance: 0,
    };
  }

  const pc = (support - candidate.correct + 1) / (support + 2);
  const pb = (support - baseline.correct + 1) / (support + 2);
  const delta = (baseline.correct - candidate.correct) / support;
  const tolerance = GUARD_Z * Math.sqrt((pc * (1 - pc) + pb * (1 - pb)) / support);
  const passed = delta <= tolerance;

  return {
    passed,
    reason: passed ? null : "regression",
    support,
    delta,
    tolerance,
  };
}

export interface FamilyGuard {
  family: string;
  support: number;
  delta: number | null;
  tolerance: number | null;
  passed: boolean;
  reason: string | null;
}

/** One guard per family present on either side. */
export function familyGuards(
  candidate: Record<string, Counts>,
  baseline: Record<string, Counts>,
): FamilyGuard[] {
  const families = [...new Set([...Object.keys(candidate), ...Object.keys(baseline)])].sort();

  return families.map((family) => {
    const current = candidate[family];
    const before = baseline[family];

    if (current === undefined || before === undefined) {
      return {
        family,
        support: current?.total ?? before?.total ?? 0,
        delta: null,
        tolerance: null,
        passed: false,
        reason: "support mismatch",
      };
    }

    const guard = proportionGuard(current, before);

    return {
      family,
      support: guard.support,
      delta: guard.delta,
      tolerance: guard.tolerance,
      passed: guard.passed,
      reason: guard.reason,
    };
  });
}

export interface GateScores {
  total: number;
  correct: number;
  byFamily: Record<string, Counts>;
  selective?: {
    silentMeaningChangeRate: number;
    negativeFalseAcceptanceRate: number;
    typoInvariance: number;
  };
}

export interface GateDecision {
  criterion: string;
  description: string;
  improvement: number;
  guards: FamilyGuard[];
  bare: { family: string; guard: ProportionGuardResult };
  failures: string[];
  accepted: boolean;
}

/**
 * Pure decision over two overlay-off score sets. Strict improvement is
 * required; every per-family guard and the bare-expression guard must pass.
 * Overlay-on scores cannot rescue a failing overlay-off candidate; see
 * {@link decidePromotion}.
 */
export type PromotionMetrics = {
  H: string;
  EXEC: string;
  "FA@cov": string;
  O: string;
  G2?: string;
};

export function refuseIncompletePromotion(metrics: Partial<PromotionMetrics>): void {
  const missing = PROMOTION_REQUIRED.filter(
    (key) => metrics[key] === undefined || String(metrics[key]).trim() === "",
  );

  if (missing.length > 0) throw new RangeError(`gate refused: missing ${missing.join(", ")}`);
}

function fractionRate(value: string | undefined): number | undefined {
  if (value === undefined) return undefined;
  const match = /^(\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)/.exec(value.trim());

  if (match === null) return undefined;
  const den = Number(match[2]);

  if (den <= 0) return undefined;

  return Number(match[1]) / den;
}

/** G2-only lift is a floor, never a promotion pass. */
export function g2OnlyImprovementIsNotPass(input: {
  g2Improved: boolean;
  hImproved: boolean;
  execImproved: boolean;
  faAtCovOk: boolean;
}): boolean {
  if (input.g2Improved && !input.hImproved && !input.execImproved) return true;

  return false;
}

export function decide(candidate: GateScores, baseline: GateScores): GateDecision {
  const failures: string[] = [];
  let improvement = 0;

  if (candidate.total !== baseline.total) {
    failures.push("evaluation support mismatch");
  } else {
    improvement =
      candidate.total === 0 ? 0 : (candidate.correct - baseline.correct) / candidate.total;

    if (candidate.correct <= baseline.correct) {
      failures.push("reserved-carrier exact sequences did not improve");
    }
  }

  const guards = familyGuards(candidate.byFamily, baseline.byFamily);

  for (const guard of guards) {
    if (!guard.passed) failures.push(`${guard.family}: ${guard.reason ?? "failed"}`);
  }

  const overallAccuracy = candidate.total === 0 ? 0 : candidate.correct / candidate.total;

  if (overallAccuracy < MIN_OVERALL_ACCURACY) {
    failures.push(`overall accuracy below ${MIN_OVERALL_ACCURACY}`);
  }

  for (const [family, counts] of Object.entries(candidate.byFamily)) {
    const accuracy = counts.total === 0 ? 0 : counts.correct / counts.total;

    if (accuracy < MIN_FAMILY_ACCURACY) {
      failures.push(`${family}: accuracy below ${MIN_FAMILY_ACCURACY}`);
    }
  }

  if (
    candidate.selective !== undefined &&
    candidate.selective.silentMeaningChangeRate > MAX_SILENT_MEANING_CHANGE_RATE
  ) {
    failures.push(`silent meaning-change rate above ${MAX_SILENT_MEANING_CHANGE_RATE}`);
  }

  if (
    candidate.selective !== undefined &&
    candidate.selective.negativeFalseAcceptanceRate > MAX_NEGATIVE_FALSE_ACCEPTANCE_RATE
  ) {
    failures.push(`negative false-acceptance rate above ${MAX_NEGATIVE_FALSE_ACCEPTANCE_RATE}`);
  }

  if (
    candidate.selective !== undefined &&
    candidate.selective.typoInvariance < MIN_TYPO_INVARIANCE
  ) {
    failures.push(`typo invariance below ${MIN_TYPO_INVARIANCE}`);
  }

  const candidateBare = candidate.byFamily[BARE_FAMILY] ?? { total: 0, correct: 0 };
  const baselineBare = baseline.byFamily[BARE_FAMILY] ?? { total: 0, correct: 0 };
  const bareGuard = proportionGuard(candidateBare, baselineBare);

  if (!bareGuard.passed) {
    failures.push(`bare expressions: ${bareGuard.reason ?? "failed"}`);
  }

  return {
    criterion: CRITERION,
    description: CRITERION_DESCRIPTION,
    improvement,
    guards,
    bare: { family: BARE_FAMILY, guard: bareGuard },
    failures,
    accepted: failures.length === 0,
  };
}

export interface OverlayScores {
  overlayOn: GateScores;
  overlayOff?: GateScores;
}

/**
 * Promotion decision: floors and reserved-carrier improvement run on
 * overlay-off. Overlay-on is diagnostic except for the same-artifact safety
 * comparison (overlay-off must not worsen silent-change or false-accept).
 * Missing overlay-off is a failed run, not an implied pass.
 */
export function decidePromotion(
  candidate: OverlayScores,
  baseline: OverlayScores,
  ledger?: Partial<PromotionMetrics>,
): GateDecision {
  if (candidate.overlayOff === undefined) {
    const bare = proportionGuard({ total: 0, correct: 0 }, { total: 0, correct: 0 });

    return {
      criterion: CRITERION,
      description: CRITERION_DESCRIPTION,
      improvement: 0,
      guards: [],
      bare: { family: BARE_FAMILY, guard: bare },
      failures: ["missing overlay-off"],
      accepted: false,
    };
  }

  if (ledger !== undefined) {
    try {
      refuseIncompletePromotion(ledger);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const bare = proportionGuard({ total: 0, correct: 0 }, { total: 0, correct: 0 });

      return {
        criterion: CRITERION,
        description: CRITERION_DESCRIPTION,
        improvement: 0,
        guards: [],
        bare: { family: BARE_FAMILY, guard: bare },
        failures: [message],
        accepted: false,
      };
    }
  }

  const baselineOff = baseline.overlayOff ?? candidate.overlayOn;
  const decision = decide(candidate.overlayOff, baselineOff);

  const failures = decision.failures.map((failure) =>
    failure.startsWith("overlay-off ") ? failure : `overlay-off ${failure}`,
  );

  if (ledger !== undefined) {
    const hRate = fractionRate(ledger.H);
    const execRate = fractionRate(ledger.EXEC);

    if (
      g2OnlyImprovementIsNotPass({
        g2Improved: decision.improvement > 0,
        hImproved: hRate !== undefined && hRate >= MIN_OVERALL_ACCURACY,
        execImproved: execRate !== undefined && execRate >= MIN_OVERALL_ACCURACY,
        faAtCovOk: faAtCovMeetsTarget(String(ledger["FA@cov"] ?? "")),
      })
    ) {
      failures.push("G2-only improvement is not a promotion pass");
    }
  }

  const overlayOnSafety = candidate.overlayOn.selective;
  const overlayOffSafety = candidate.overlayOff.selective;

  if (overlayOnSafety !== undefined && overlayOffSafety !== undefined) {
    if (overlayOffSafety.silentMeaningChangeRate > overlayOnSafety.silentMeaningChangeRate) {
      failures.push("overlay-off silent meaning-change worse than overlay-on");
    }

    if (
      overlayOffSafety.negativeFalseAcceptanceRate > overlayOnSafety.negativeFalseAcceptanceRate
    ) {
      failures.push("overlay-off negative false-accept worse than overlay-on");
    }
  }

  return {
    ...decision,
    failures,
    accepted: failures.length === 0,
  };
}

export interface LeakReport {
  checked: boolean;
  split: string;
  carriers: string[];
  leaked: string[];
  splitOverlaps: Array<{
    left: "train" | "validation";
    right: "validation" | "heldout";
    count: number;
    examples: string[];
  }>;
  reason: string | null;
  scope: string;
}

const LEAK_SCOPE =
  "Reserved carrier fingerprints are recomputed from the Phase 07 " +
  "RESERVED_PHRASES and intersected with the generated training carrier " +
  "hashes. Train/validation, train/held-out, and validation/held-out frame " +
  "fingerprints must also be disjoint. Missing, malformed, or stale metadata " +
  "is reported as unchecked and fails promotion.";

/** Map `sha256(coarse(carrier)).slice(0, 12)` back to the carrier text. */
function carrierHashes(): Map<string, string> {
  const hashes = new Map<string, string>();

  for (const phrase of RESERVED_PHRASES) {
    hashes.set(sha256Hex(coarseText(phrase)).slice(0, 12), phrase);
  }

  return hashes;
}

/** Fail the gate when reserved carrier wording appears in the training split. */
export function checkCarrierLeak(trainSplit: string): LeakReport {
  const path = resolveRepo(trainSplit);
  const hashes = carrierHashes();
  const carriers = [...hashes.values()].sort();

  const unchecked = (reason: string): LeakReport => ({
    checked: false,
    split: path,
    carriers,
    leaked: [],
    splitOverlaps: [],
    reason,
    scope: LEAK_SCOPE,
  });

  if (!existsSync(path)) {
    return unchecked("metadata missing");
  }

  let value: unknown;

  try {
    value = JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return unchecked("metadata is not valid JSON");
  }

  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return unchecked("metadata is not an object");
  }

  const metadata = value as Record<string, unknown>;

  if (metadata.generatorSha256 !== generatorSha256()) {
    return unchecked("metadata generator hash is stale");
  }

  const splitNames = ["train", "validation", "heldout"] as const;
  const fingerprints = metadata.fingerprintsBySplit;
  const metadataCarrierHashes = metadata.carrierHashesBySplit;

  if (
    fingerprints === null ||
    typeof fingerprints !== "object" ||
    Array.isArray(fingerprints) ||
    metadataCarrierHashes === null ||
    typeof metadataCarrierHashes !== "object" ||
    Array.isArray(metadataCarrierHashes)
  ) {
    return unchecked("metadata lacks per-split fingerprints or carrier hashes");
  }

  const fingerprintSets = {} as Record<(typeof splitNames)[number], Set<string>>;
  const carrierHashSets = {} as Record<(typeof splitNames)[number], Set<string>>;

  for (const split of splitNames) {
    const splitFingerprints = (fingerprints as Record<string, unknown>)[split];
    const splitCarriers = (metadataCarrierHashes as Record<string, unknown>)[split];

    if (
      !Array.isArray(splitFingerprints) ||
      !splitFingerprints.every((entry) => typeof entry === "string") ||
      !Array.isArray(splitCarriers) ||
      !splitCarriers.every((entry) => typeof entry === "string")
    ) {
      return unchecked(`metadata has an invalid ${split} split`);
    }

    fingerprintSets[split] = new Set(splitFingerprints);
    carrierHashSets[split] = new Set(splitCarriers);
  }

  const leaked = new Set<string>();

  for (const entry of carrierHashSets.train) {
    const carrier = hashes.get(entry);

    if (carrier !== undefined) leaked.add(carrier);
  }

  const pairs = [
    ["train", "validation"],
    ["train", "heldout"],
    ["validation", "heldout"],
  ] as const;

  const splitOverlaps = pairs.map(([left, right]) => {
    const overlap: string[] = [];

    for (const fingerprint of fingerprintSets[left]) {
      if (fingerprintSets[right].has(fingerprint)) overlap.push(fingerprint);
    }

    overlap.sort();

    return { left, right, count: overlap.length, examples: overlap.slice(0, 10) };
  });

  return {
    checked: true,
    split: path,
    carriers,
    leaked: [...leaked].sort(),
    splitOverlaps,
    reason: null,
    scope: LEAK_SCOPE,
  };
}

function contractOf(result: EvaluationResult, metric: Metric): ContractReport {
  const contract = result.contracts.find((item) => item.metric === metric);

  if (contract === undefined) throw new Error(`missing contract: ${metric}`);

  return contract;
}

function overlayOffContract(result: EvaluationResult, metric: Metric): ContractReport {
  const contract = result.overlayOff.contracts.find((item) => item.metric === metric);

  if (contract === undefined) throw new Error(`missing overlay-off contract: ${metric}`);

  return contract;
}

export interface GateOptions {
  dist: string;
  candidate: string;
  baseline: string;
  corpus: string;
  trainSplit: string;
  out: string;
  metrics?: Partial<PromotionMetrics>;
}

export interface GateReport extends GateDecision {
  artifactSha256s: { candidate: string; baseline: string };
  candidate: GateScores;
  baseline: GateScores;
  overlayOn: { candidate: GateScores; baseline: GateScores };
  overlayOff: { candidate: GateScores; baseline: GateScores };
  speechActAccuracy?: { candidate?: number; baseline?: number };
  actionableAccuracy?: { candidate?: number; baseline?: number };
  leak: LeakReport;
}

function scoresOf(
  result: EvaluationResult,
  contract: ContractReport,
  overlay: "on" | "off" = "on",
): GateScores {
  const curve =
    overlay === "off" ? result.overlayOff.selectivePrediction : result.selectivePrediction;

  const selective = curve.find((point) => point.threshold === 0.5);

  if (selective === undefined) {
    throw new Error("missing selective-prediction point at confidence 0.5");
  }

  const typo = overlay === "off" ? result.overlayOff.typoInvariance : result.typoInvariance;

  return {
    total: contract.total,
    correct: contract.correct,
    byFamily: contract.byFamily,
    selective: {
      silentMeaningChangeRate: selective.silentMeaningChangeRate,
      negativeFalseAcceptanceRate: selective.negativeFalseAcceptanceRate,
      typoInvariance: typo.pairedAccuracy,
    },
  };
}

/** Load both artifacts, rescore both in this process, and decide. */
export async function runGate(options: GateOptions): Promise<GateReport> {
  const leak = checkCarrierLeak(options.trainSplit);
  const candidateEngine: Engine = await loadEngine(options.dist, options.candidate);
  const baselineEngine: Engine = await loadEngine(options.dist, options.baseline);

  try {
    const candidateResult = await evaluateCorpus(candidateEngine, options.corpus, DEFAULT_SCOPE);
    const baselineResult = await evaluateCorpus(baselineEngine, options.corpus, DEFAULT_SCOPE);

    if (candidateResult.overlayOff === undefined) {
      throw new Error("missing overlay-off evaluation; not an implied pass");
    }

    if (baselineResult.overlayOff === undefined) {
      throw new Error("missing overlay-off evaluation; not an implied pass");
    }

    assertSameArtifact([...candidateResult.contracts, ...candidateResult.overlayOff.contracts]);
    assertSameArtifact([...baselineResult.contracts, ...baselineResult.overlayOff.contracts]);

    const candidateOn = scoresOf(candidateResult, contractOf(candidateResult, "exactAst"), "on");

    const candidateOff = scoresOf(
      candidateResult,
      overlayOffContract(candidateResult, "exactAst"),
      "off",
    );

    const baselineOn = scoresOf(baselineResult, contractOf(baselineResult, "exactAst"), "on");

    const baselineOff = scoresOf(
      baselineResult,
      overlayOffContract(baselineResult, "exactAst"),
      "off",
    );

    const decision = decidePromotion(
      { overlayOn: candidateOn, overlayOff: candidateOff },
      { overlayOn: baselineOn, overlayOff: baselineOff },
      options.metrics ?? {},
    );

    if (!leak.checked) {
      decision.failures.push("training split missing; leakage check was not performed");
      decision.accepted = false;
    } else if (leak.leaked.length > 0) {
      decision.failures.push(
        `reserved carrier leaked into training split: ${leak.leaked.join(", ")}`,
      );

      decision.accepted = false;
    }

    for (const overlap of leak.splitOverlaps) {
      if (overlap.count === 0) continue;
      decision.failures.push(`${overlap.left}/${overlap.right} frame leakage: ${overlap.count}`);
      decision.accepted = false;
    }

    const report: GateReport = {
      ...decision,
      artifactSha256s: {
        candidate: candidateEngine.artifactSha256,
        baseline: baselineEngine.artifactSha256,
      },
      candidate: candidateOff,
      baseline: baselineOff,
      overlayOn: { candidate: candidateOn, baseline: baselineOn },
      overlayOff: { candidate: candidateOff, baseline: baselineOff },
      ...(candidateResult.speechActAccuracy === undefined &&
      baselineResult.speechActAccuracy === undefined
        ? {}
        : {
            speechActAccuracy: {
              ...(candidateResult.speechActAccuracy === undefined
                ? {}
                : { candidate: candidateResult.speechActAccuracy }),
              ...(baselineResult.speechActAccuracy === undefined
                ? {}
                : { baseline: baselineResult.speechActAccuracy }),
            },
          }),
      ...(candidateResult.actionableAccuracy === undefined &&
      baselineResult.actionableAccuracy === undefined
        ? {}
        : {
            actionableAccuracy: {
              ...(candidateResult.actionableAccuracy === undefined
                ? {}
                : { candidate: candidateResult.actionableAccuracy }),
              ...(baselineResult.actionableAccuracy === undefined
                ? {}
                : { baseline: baselineResult.actionableAccuracy }),
            },
          }),
      leak,
    };

    const out = resolveRepo(options.out);

    await mkdir(dirname(out), { recursive: true });
    await writeFile(out, `${JSON.stringify(report, null, 2)}\n`);

    return report;
  } finally {
    await candidateEngine.dispose();
    await baselineEngine.dispose();
  }
}

function readOption(argv: readonly string[], index: number): string {
  const value = argv[index + 1];

  if (value === undefined) throw new RangeError(`missing value for ${argv[index]}`);

  return value;
}

export function parseArgs(argv: readonly string[]): GateOptions {
  const options: GateOptions = { ...DEFAULTS };
  const metrics: Partial<PromotionMetrics> = {};

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]!;

    switch (arg) {
      case "--dist":
        options.dist = readOption(argv, index);
        index += 1;
        break;
      case "--candidate":
        options.candidate = readOption(argv, index);
        index += 1;
        break;
      case "--baseline":
        options.baseline = readOption(argv, index);
        index += 1;
        break;
      case "--corpus":
        options.corpus = readOption(argv, index);
        index += 1;
        break;
      case "--train-split":
        options.trainSplit = readOption(argv, index);
        index += 1;
        break;
      case "--out":
        options.out = readOption(argv, index);
        index += 1;
        break;
      case "--h":
        metrics.H = readOption(argv, index);
        index += 1;
        break;
      case "--exec":
        metrics.EXEC = readOption(argv, index);
        index += 1;
        break;
      case "--fa-at-cov":
        metrics["FA@cov"] = readOption(argv, index);
        index += 1;
        break;
      case "--o":
        metrics.O = readOption(argv, index);
        index += 1;
        break;

      case "--metrics": {
        const path = resolveRepo(readOption(argv, index));

        index += 1;
        const loaded = JSON.parse(readFileSync(path, "utf8")) as Partial<PromotionMetrics>;

        Object.assign(metrics, loaded);
        break;
      }

      default:
        throw new RangeError(`unknown flag: ${arg}`);
    }
  }

  if (Object.keys(metrics).length > 0) options.metrics = metrics;

  return options;
}

export { assertSameArtifact } from "./evaluate";

if (import.meta.main) {
  const options = parseArgs(process.argv.slice(2));
  const report = await runGate(options);

  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);

  if (!report.accepted) {
    process.stderr.write(`gate rejected: ${report.failures.join("; ")}\n`);
    process.exitCode = 1;
  }
}
