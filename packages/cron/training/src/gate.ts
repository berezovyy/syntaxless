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
import {
  assertSameArtifact,
  DEFAULT_SCOPE,
  evaluateCorpus,
  loadEngine,
  parseCorpus,
  resolveRepo,
  sha256Hex,
} from "./evaluate";
import type { ContractReport, Counts, Engine, EvaluationResult, Metric } from "./evaluate";
import { coarseText, fingerprintPattern } from "./fingerprint";
import type { Span } from "./sentence";

/** Two-proportion guard constants, fixed by the plan. */
export const GUARD_Z = 1.96;
export const GUARD_MINIMUM_SUPPORT = 30;
/** Absolute floors prevent a relative improvement from promoting a weak model. */
export const MIN_OVERALL_ACCURACY = 0.5;
export const MIN_FAMILY_ACCURACY = 0.25;

/** The overall promotion criterion and the terse-family bare guard. */
export const CRITERION = "reserved-carrier-exact-intent";
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
  trainSplit: "packages/cron/training/data/synth/train.jsonl",
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
 * Pure decision over two score sets. Strict improvement is required; every
 * per-family guard and the bare-expression guard must pass.
 */
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

export interface LeakReport {
  checked: boolean;
  split: string;
  carriers: string[];
  leaked: string[];
  scope: string;
}

const LEAK_SCOPE =
  "Reserved carrier fingerprints are recomputed from the Phase 07 " +
  "RESERVED_PHRASES and intersected with the Phase 08 training split. A " +
  "missing split is reported as unchecked and fails promotion.";

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

  if (!existsSync(path)) {
    return { checked: false, split: path, carriers, leaked: [], scope: LEAK_SCOPE };
  }

  const leaked = new Set<string>();

  for (const record of parseCorpus(readFileSync(path, "utf8"))) {
    const pattern = fingerprintPattern(record.text, record.spans as readonly Span[]);

    for (const entry of pattern) {
      const carrier = hashes.get(entry);

      if (carrier !== undefined) leaked.add(carrier);
    }
  }

  return {
    checked: true,
    split: path,
    carriers,
    leaked: [...leaked].sort(),
    scope: LEAK_SCOPE,
  };
}

function contractOf(result: EvaluationResult, metric: Metric): ContractReport {
  const contract = result.contracts.find((item) => item.metric === metric);

  if (contract === undefined) throw new Error(`missing contract: ${metric}`);

  return contract;
}

export interface GateOptions {
  dist: string;
  candidate: string;
  baseline: string;
  corpus: string;
  trainSplit: string;
  out: string;
}

export interface GateReport extends GateDecision {
  artifactSha256s: { candidate: string; baseline: string };
  candidate: GateScores;
  baseline: GateScores;
  leak: LeakReport;
}

function scoresOf(contract: ContractReport): GateScores {
  return {
    total: contract.total,
    correct: contract.correct,
    byFamily: contract.byFamily,
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

    assertSameArtifact(candidateResult.contracts);
    assertSameArtifact(baselineResult.contracts);

    const candidate = scoresOf(contractOf(candidateResult, "exactAst"));
    const baseline = scoresOf(contractOf(baselineResult, "exactAst"));
    const decision = decide(candidate, baseline);

    if (!leak.checked) {
      decision.failures.push("training split missing; leakage check was not performed");
      decision.accepted = false;
    } else if (leak.leaked.length > 0) {
      decision.failures.push(
        `reserved carrier leaked into training split: ${leak.leaked.join(", ")}`,
      );

      decision.accepted = false;
    }

    const report: GateReport = {
      ...decision,
      artifactSha256s: {
        candidate: candidateEngine.artifactSha256,
        baseline: baselineEngine.artifactSha256,
      },
      candidate,
      baseline,
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
      default:
        throw new RangeError(`unknown flag: ${arg}`);
    }
  }

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
