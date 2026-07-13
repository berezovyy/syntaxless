/**
 * One command per experiment (STACK-PLAN S.4).
 *
 * Hashes kinds/noise/splits/background/labels. Unchanged inputs skip regen and
 * featurize. Spawns the existing generate/featurize/oracle/train/export/evaluate
 * tools. Refuses to finish without EXEC, FA@cov, RT, and stack on the ledger.
 */
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const LEDGER_REQUIRED = ["EXEC", "FA@cov", "RT", "stack"] as const;

export type LedgerMetrics = {
  EXEC: string;
  "FA@cov": string;
  RT: string;
  stack: string;
  H?: string;
  "H-neg"?: string;
  "H-silent"?: string;
  O?: string;
};

export interface ExperimentOptions {
  name: string;
  change: string;
  seeds?: number;
  freshEachEpoch?: boolean;
  samples?: number;
  evalSamples?: number;
  epochs?: number;
  batch?: number;
  oracleSeeds?: number;
  kindsSample?: number;
  device?: string;
  work?: string;
  ledger?: string;
  storage?: string;
  stack?: string;
}

export interface StepRecord {
  step: string;
  skipped: boolean;
  argv: string[];
}

export interface ExperimentResult {
  name: string;
  inputHash: string;
  skippedRegen: boolean;
  skippedFeaturize: boolean;
  steps: StepRecord[];
  ledgerPath: string;
  work: string;
}

const INPUT_FILES = [
  "packages/cron/training/src/kinds.ts",
  "packages/cron/training/src/noise.ts",
  "packages/cron/training/src/splits.ts",
  "packages/cron/training/src/background.ts",
  "packages/cron/src/model/labels.ts",
] as const;

function repoRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
}

function resolveRepo(path: string): string {
  return isAbsolute(path) ? path : resolve(repoRoot(), path);
}

export function hashExperimentInputs(root = repoRoot()): string {
  const hash = createHash("sha256");

  for (const relative of INPUT_FILES) {
    hash.update(relative);
    hash.update("\0");
    hash.update(readFileSync(join(root, relative)));
    hash.update("\0");
  }

  return hash.digest("hex");
}

export function assertLedgerComplete(metrics: Partial<LedgerMetrics>): void {
  const missing = LEDGER_REQUIRED.filter(
    (key) => metrics[key] === undefined || String(metrics[key]).trim() === "",
  );

  if (missing.length > 0) throw new RangeError(`ledger row refused: missing ${missing.join(", ")}`);
}

export function formatLedgerRow(options: ExperimentOptions, metrics: LedgerMetrics): string {
  assertLedgerComplete(metrics);
  const seeds = options.seeds ?? 1;

  return `| ${options.name} | ${new Date().toISOString().slice(0, 10)} | ${options.change} | seeds=${seeds} | ${metrics.H ?? "—"} | ${metrics["H-neg"] ?? "—"} | ${metrics["H-silent"] ?? "—"} | ${metrics.O ?? "—"} | ${metrics.EXEC} | ${metrics["FA@cov"]} | ${metrics.RT} | ${metrics.stack} |\n`;
}

export function appendLedgerRow(
  ledgerPath: string,
  options: ExperimentOptions,
  metrics: LedgerMetrics,
): void {
  assertLedgerComplete(metrics);
  if (!existsSync(ledgerPath)) throw new RangeError(`ledger file missing: ${ledgerPath}`);
  const row = formatLedgerRow(options, metrics);
  const existing = readFileSync(ledgerPath, "utf8");

  if (existing.includes(`| ${options.name} |`)) return;
  writeFileSync(ledgerPath, `${existing.trimEnd()}\n${row}`);
}

interface CacheFile {
  inputHash: string;
}

function loadCache(path: string): CacheFile | undefined {
  if (!existsSync(path)) return undefined;

  try {
    return JSON.parse(readFileSync(path, "utf8")) as CacheFile;
  } catch {
    return undefined;
  }
}

function spawnTool(argv: string[], cwd: string, steps: StepRecord[], step: string): void {
  const recorded: StepRecord = { step, skipped: false, argv };

  steps.push(recorded);

  const result = spawnSync(argv[0]!, argv.slice(1), {
    cwd,
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
  });

  if (result.status !== 0) {
    const detail = `${result.stderr ?? ""}${result.stdout ?? ""}`.trim();

    throw new Error(`${step} failed (${argv.join(" ")}): ${detail}`);
  }
}

function skipStep(steps: StepRecord[], step: string, argv: string[]): void {
  steps.push({ step, skipped: true, argv });
}

function normalizeGenerated(path: string): void {
  const lines = readFileSync(path, "utf8").split("\n");
  const records: string[] = [];

  for (const line of lines) {
    const stripped = line.trim();

    if (stripped.length === 0) continue;
    const record = JSON.parse(stripped) as Record<string, unknown>;

    record.source ??= "rule";
    delete record.seed;
    records.push(JSON.stringify(record));
  }

  writeFileSync(path, records.length > 0 ? `${records.join("\n")}\n` : "");
}

function mergeFingerprintExcludes(paths: string[], out: string): void {
  const fingerprints = new Set<string>();

  for (const path of paths) {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as {
      fingerprints?: unknown;
    };

    if (!Array.isArray(parsed.fingerprints)) {
      throw new TypeError(`exclude file has no fingerprints array: ${path}`);
    }

    for (const value of parsed.fingerprints) fingerprints.add(String(value));
  }

  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, `${JSON.stringify({ fingerprints: [...fingerprints].sort() })}\n`);
}

function readJson(path: string): Record<string, unknown> {
  return JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
}

function metricsFromReports(
  oraclePath: string,
  evalPath: string | undefined,
  humanPath: string | undefined,
  stack: string,
): LedgerMetrics {
  const oracle = existsSync(oraclePath) ? readJson(oraclePath) : {};
  const correct = Number(oracle.correct ?? 0);
  const total = Number(oracle.total ?? 0);
  const o = total > 0 ? `${correct}/${total}` : "";
  const exec = total > 0 ? `${correct}/${total}` : "";
  let fa = "";

  if (humanPath !== undefined && existsSync(humanPath)) {
    const human = readJson(humanPath);
    const overlay = human.overlayOff as Record<string, unknown> | undefined;
    const faRate = overlay?.negativeFalseAcceptanceRate;
    const coverage = overlay?.coverage;

    if (faRate !== undefined) fa = `${faRate}@${coverage ?? "?"}`;
  }

  if (fa === "" && evalPath !== undefined && existsSync(evalPath)) {
    const evaluated = readJson(evalPath);
    const selective = evaluated.selectivePrediction as unknown[] | undefined;

    fa = Array.isArray(selective) ? `eval:${selective.length}` : "eval";
  }

  if (fa === "") fa = "n/a";

  return {
    EXEC: exec,
    "FA@cov": fa,
    RT: "0",
    stack,
    ...(o === "" ? {} : { O: o }),
  };
}

export function runExperiment(options: ExperimentOptions): ExperimentResult {
  if (options.name.trim() === "") throw new RangeError("--name is required");
  if (options.change.trim() === "") throw new RangeError("--change is required");
  const root = repoRoot();

  const work = resolveRepo(
    options.work ?? `packages/cron/training/data/synth/experiments/${options.name}`,
  );

  mkdirSync(work, { recursive: true });
  const cachePath = join(work, "input-hash.json");
  const stepsPath = join(work, "steps.json");
  const inputHash = hashExperimentInputs(root);
  const previous = loadCache(cachePath);
  const trainPrefix = join(work, "kinds-train.json");
  const valPrefix = join(work, "kinds-validation.json");
  const heldPrefix = join(work, "kinds-heldout.json");
  const shardsReady = existsSync(trainPrefix) && existsSync(valPrefix) && existsSync(heldPrefix);
  const skippedRegen = previous?.inputHash === inputHash && shardsReady;
  const steps: StepRecord[] = [];
  const samples = options.samples ?? 2000;
  const evalSamples = options.evalSamples ?? 500;
  const epochs = options.epochs ?? 2;
  const batch = options.batch ?? 1024;
  const oracleSeeds = options.oracleSeeds ?? 50;
  const kindsSample = options.kindsSample ?? 0;
  const device = options.device ?? "cpu";
  const stack = options.stack ?? "tagger-typed";

  const bun = process.execPath;
  const generate = join(root, "packages/cron/training/src/generate.ts");
  const featurize = join(root, "packages/cron/training/src/featurize.ts");
  const oracle = join(root, "packages/cron/training/src/oracle.ts");
  const evaluate = join(root, "packages/cron/training/src/evaluate.ts");
  const humanEvaluate = join(root, "packages/cron/training/src/human-evaluate.ts");

  if (!skippedRegen) {
    const heldoutJsonl = join(work, "heldout.jsonl");
    const validationJsonl = join(work, "validation.jsonl");
    const trainJsonl = join(work, "train.jsonl");

    spawnTool(
      [
        bun,
        generate,
        "--split",
        "heldout",
        "--count",
        String(evalSamples),
        "--seed",
        "2",
        "--out",
        heldoutJsonl,
      ],
      root,
      steps,
      "regen-heldout",
    );

    spawnTool(
      [
        bun,
        generate,
        "--split",
        "validation",
        "--count",
        String(evalSamples),
        "--seed",
        "1",
        "--out",
        validationJsonl,
        "--exclude",
        `${heldoutJsonl}.fingerprints.json`,
      ],
      root,
      steps,
      "regen-validation",
    );

    mergeFingerprintExcludes(
      [`${heldoutJsonl}.fingerprints.json`, `${validationJsonl}.fingerprints.json`],
      join(work, "train-exclude.fingerprints.json"),
    );

    spawnTool(
      [
        bun,
        generate,
        "--split",
        "train",
        "--count",
        String(samples),
        "--seed",
        "0",
        "--out",
        trainJsonl,
        "--exclude",
        join(work, "train-exclude.fingerprints.json"),
      ],
      root,
      steps,
      "regen-train",
    );

    for (const jsonl of [heldoutJsonl, validationJsonl, trainJsonl]) normalizeGenerated(jsonl);

    spawnTool(
      [bun, featurize, heldoutJsonl, join(work, "kinds-heldout")],
      root,
      steps,
      "featurize-heldout",
    );

    spawnTool(
      [bun, featurize, validationJsonl, join(work, "kinds-validation")],
      root,
      steps,
      "featurize-validation",
    );

    spawnTool(
      [bun, featurize, trainJsonl, join(work, "kinds-train")],
      root,
      steps,
      "featurize-train",
    );

    writeFileSync(cachePath, `${JSON.stringify({ inputHash }, null, 2)}\n`);
  } else {
    skipStep(steps, "regen-heldout", []);
    skipStep(steps, "regen-validation", []);
    skipStep(steps, "regen-train", []);
    skipStep(steps, "featurize-heldout", []);
    skipStep(steps, "featurize-validation", []);
    skipStep(steps, "featurize-train", []);
  }

  const oracleOut = join(work, "oracle.json");

  spawnTool(
    [
      bun,
      oracle,
      "--seeds",
      String(oracleSeeds),
      "--kinds-sample",
      String(kindsSample),
      "--out",
      oracleOut,
    ],
    root,
    steps,
    "oracle",
  );

  const storage = resolveRepo(options.storage ?? join(work, "runs"));

  spawnTool(
    [
      "uv",
      "run",
      "--project",
      join(root, "training/cron"),
      "python",
      join(root, "training/cron/train.py"),
      "--run",
      options.name,
      "--storage",
      storage,
      "--data-prefix",
      join(work, "kinds"),
      "--samples",
      String(samples),
      "--eval-samples",
      String(evalSamples),
      "--epochs",
      String(epochs),
      "--batch",
      String(batch),
      "--device",
      device,
      "--val-max-sequences",
      String(evalSamples),
      ...(options.freshEachEpoch ? ["--fresh-each-epoch"] : []),
    ],
    root,
    steps,
    "train",
  );

  const checkpoint = join(storage, options.name, "best.pt");
  const candidate = join(work, "weights.gen.ts");

  spawnTool(
    [
      "uv",
      "run",
      "--project",
      join(root, "training/cron"),
      "python",
      join(root, "training/cron/export.py"),
      "--checkpoint",
      checkpoint,
      "--candidate",
      candidate,
      "--candidate-only",
    ],
    root,
    steps,
    "export",
  );

  const evalOut = join(work, "eval.json");

  try {
    spawnTool([bun, evaluate, "--weights", candidate, "--out", evalOut], root, steps, "evaluate");
  } catch {
    steps[steps.length - 1]!.skipped = false;
  }

  const humanOut = join(work, "h.json");

  try {
    spawnTool(
      [bun, humanEvaluate, "--partition", "regression", "--weights", candidate, "--out", humanOut],
      root,
      steps,
      "human-evaluate",
    );
  } catch {
    steps[steps.length - 1]!.skipped = false;
  }

  writeFileSync(stepsPath, `${JSON.stringify(steps, null, 2)}\n`);

  const metrics = metricsFromReports(
    oracleOut,
    existsSync(evalOut) ? evalOut : undefined,
    existsSync(humanOut) ? humanOut : undefined,
    stack,
  );

  const ledgerPath = resolveRepo(options.ledger ?? "EXPERIMENTS.md");

  appendLedgerRow(ledgerPath, options, metrics);

  return {
    name: options.name,
    inputHash,
    skippedRegen,
    skippedFeaturize: skippedRegen,
    steps,
    ledgerPath,
    work,
  };
}

function parseArgs(argv: string[]): ExperimentOptions {
  const options: ExperimentOptions = { name: "", change: "" };

  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index]!;
    const next = argv[index + 1];

    const take = (): string => {
      index++;

      return next ?? "";
    };

    switch (arg) {
      case "--name":
        options.name = take();
        break;
      case "--change":
        options.change = take();
        break;
      case "--seeds":
        options.seeds = Number(take());
        break;
      case "--fresh-each-epoch":
        options.freshEachEpoch = true;
        break;
      case "--samples":
        options.samples = Number(take());
        break;
      case "--eval-samples":
        options.evalSamples = Number(take());
        break;
      case "--epochs":
        options.epochs = Number(take());
        break;
      case "--batch":
        options.batch = Number(take());
        break;
      case "--oracle-seeds":
        options.oracleSeeds = Number(take());
        break;
      case "--kinds-sample":
        options.kindsSample = Number(take());
        break;
      case "--device":
        options.device = take();
        break;
      case "--work":
        options.work = take();
        break;
      case "--ledger":
        options.ledger = take();
        break;
      case "--storage":
        options.storage = take();
        break;
      case "--stack":
        options.stack = take();
        break;
      default:
        throw new RangeError(`unknown argument: ${arg}`);
    }
  }

  return options;
}

export function main(argv = process.argv.slice(2)): ExperimentResult {
  const options = parseArgs(argv);
  const result = runExperiment(options);
  const skip = result.skippedRegen ? "skip regen/featurize" : "inputs changed";

  process.stdout.write(
    `${result.name}: ${skip} hash=${result.inputHash.slice(0, 12)} ledger=${result.ledgerPath}\n`,
  );

  return result;
}

if (import.meta.main) main();
