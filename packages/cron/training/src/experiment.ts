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

import { writeBackward } from "./backward";
import { featurizeKindsStreaming } from "./featurize-kinds";
import { runIncrementalKinds } from "./incremental-kinds";
import { kindIds } from "./kinds";
import type { KindId } from "./kinds";
import { loadAcceptedReal } from "./real/accepted";
import { mixAcceptedIntoTrain } from "./real/mix";
import type { MixResult } from "./real/mix";

export const LEDGER_REQUIRED = ["EXEC", "FA@cov", "RT", "stack"] as const;

export type LedgerMetrics = {
  EXEC: string;
  "FA@cov": string;
  RT: string;
  stack: string;
  /** Positives with the exact intent, of all positives (decision metric). */
  "H-pos"?: string;
  /** Negatives correctly rejected, of all negatives (safety metric). */
  "H-negOK"?: string;
  "H-silent"?: string;
  O?: string;
};

export interface ExperimentOptions {
  name: string;
  change: string;
  seeds?: number;
  seed?: number;
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
  kinds?: readonly KindId[];
  kindCap?: number;
  sweep?: boolean;
  earlyStoppingPatience?: number;
  /** Extra argv appended to train.py (e.g. --balance-kinds, --learning-rate 1e-3). */
  trainArgs?: string[];
  init?: string;
  dataPrefix?: string;
  rt?: number;
  real?: string;
  /** Generate via R/H/E mix quotas instead of catalog walk / per-kind concat. */
  mixQuotas?: boolean;
  mixLimit?: number;
}

/** A/B grid over train.py flags (STACK-PLAN S.5). */
export const SWEEP: readonly { id: string; extra: readonly string[] }[] = [
  { id: "base", extra: [] },
  { id: "fresh", extra: ["--fresh-each-epoch"] },
];

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
  renderedKinds: string[];
  skippedKinds: string[];
  seedRuns: number;
}

const INPUT_FILES = [
  "packages/cron/training/src/kinds.ts",
  "packages/cron/training/src/noise.ts",
  "packages/cron/training/src/splits.ts",
  "packages/cron/training/src/background.ts",
  "packages/cron/training/src/mix/quotas.ts",
  "packages/cron/training/src/mix/draw.ts",
  "packages/cron/training/src/mix/filter.ts",
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

  return `| ${options.name} | ${new Date().toISOString().slice(0, 10)} | ${options.change} | seeds=${seeds} | ${metrics["H-pos"] ?? "—"} | ${metrics["H-negOK"] ?? "—"} | ${metrics["H-silent"] ?? "—"} | ${metrics.O ?? "—"} | ${metrics.EXEC} | ${metrics["FA@cov"]} | ${metrics.RT} | ${metrics.stack} |\n`;
}

function parseFraction(value: string | undefined): { num: number; den: number } | undefined {
  if (value === undefined) return undefined;
  const match = /^(\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)/.exec(value);

  if (match === null) return undefined;

  return { num: Number(match[1]), den: Number(match[2]) };
}

export function formatMeanRange(values: readonly string[]): string {
  const parsed = values
    .map(parseFraction)
    .filter((item): item is { num: number; den: number } => item !== undefined);

  if (parsed.length === 0) return values.find((item) => item.trim() !== "") ?? "n/a";
  const dens = parsed.map((item) => item.den);
  const den = dens[0]!;
  const nums = parsed.map((item) => item.num);
  const mean = nums.reduce((sum, n) => sum + n, 0) / nums.length;
  const min = Math.min(...nums);
  const max = Math.max(...nums);

  if (min === max) return `${min}/${den}`;

  return `${mean.toFixed(1)}/${den} (${min}-${max})`;
}

export function aggregateSeedMetrics(rows: readonly LedgerMetrics[], stack: string): LedgerMetrics {
  if (rows.length === 0) throw new RangeError("aggregateSeedMetrics: no seed rows");

  return {
    EXEC: formatMeanRange(rows.map((row) => row.EXEC)),
    "FA@cov": rows.every((row) => row["FA@cov"] === rows[0]!["FA@cov"])
      ? rows[0]!["FA@cov"]
      : rows.map((row) => row["FA@cov"]).join(","),
    RT: rows[0]!.RT,
    stack,
    "H-pos": formatMeanRange(rows.map((row) => row["H-pos"] ?? "")),
    "H-negOK": formatMeanRange(rows.map((row) => row["H-negOK"] ?? "")),
    "H-silent": formatMeanRange(rows.map((row) => row["H-silent"] ?? "")),
    O: formatMeanRange(rows.map((row) => row.O ?? "")),
  };
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
  const marker = `| ${options.name} |`;

  if (existing.includes(marker)) {
    const updated = existing
      .split("\n")
      .map((line) => (line.includes(marker) ? row.trimEnd() : line))
      .join("\n");

    writeFileSync(ledgerPath, updated.endsWith("\n") ? updated : `${updated}\n`);

    return;
  }

  writeFileSync(ledgerPath, `${existing.trimEnd()}\n${row}`);
}

interface CacheFile {
  inputHash: string;
  rt?: number;
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

function writeMix(work: string, mix: MixResult): void {
  writeFileSync(join(work, "mix.json"), `${JSON.stringify(mix, null, 2)}\n`);
}

function readMixFraction(work: string): string | undefined {
  const path = join(work, "mix.json");

  if (!existsSync(path)) return undefined;
  const mix = readJson(path);

  return typeof mix.fraction === "string" ? mix.fraction : undefined;
}

function writeInputCache(path: string, inputHash: string, rt: number): void {
  writeFileSync(path, `${JSON.stringify({ inputHash, rt }, null, 2)}\n`);
}

export function metricsFromReports(
  oraclePath: string,
  evalPath: string | undefined,
  humanPath: string | undefined,
  stack: string,
  rt: string,
): LedgerMetrics {
  const oracle = existsSync(oraclePath) ? readJson(oraclePath) : {};
  const oCorrect = Number(oracle.correct ?? 0);
  const oTotal = Number(oracle.total ?? 0);
  const o = oTotal > 0 ? `${oCorrect}/${oTotal}` : "";
  let exec = "";
  let fa = "";

  if (evalPath !== undefined && existsSync(evalPath)) {
    const evaluated = readJson(evalPath);
    const execBlock = evaluated.exec as { correct?: unknown; total?: unknown } | undefined;
    const execTotal = Number(execBlock?.total ?? 0);

    if (execTotal > 0) exec = `${Number(execBlock?.correct ?? 0)}/${execTotal}`;

    if (typeof evaluated.faAtCov === "string" && evaluated.faAtCov.trim() !== "") {
      fa = evaluated.faAtCov;
    }
  }

  let hPos: string | undefined;
  let hNegOk: string | undefined;
  let hSilent: string | undefined;

  if (humanPath !== undefined && existsSync(humanPath)) {
    const human = readJson(humanPath);
    const overlay = (human.overlayOff ?? human) as Record<string, unknown>;
    const posTotal = Number(overlay.positiveTotal ?? 0);
    const negTotal = Number(overlay.negativeTotal ?? 0);

    if (posTotal > 0) hPos = `${Number(overlay.positiveCorrect ?? 0)}/${posTotal}`;
    if (negTotal > 0) hNegOk = `${Number(overlay.negativeCorrect ?? 0)}/${negTotal}`;
    const humanFa = overlay.faAtCov ?? human.faAtCov;

    if (typeof humanFa === "string" && humanFa.trim() !== "") fa = humanFa;
    const silent = human.positiveSilentMeaningChanges ?? overlay.silentMeaningChanges;

    if (silent !== undefined) hSilent = String(silent);
  }

  if (exec === "") exec = "n/a";
  if (fa === "") fa = "n/a";

  return {
    EXEC: exec,
    "FA@cov": fa,
    RT: rt,
    stack,
    ...(o === "" ? {} : { O: o }),
    ...(hPos === undefined ? {} : { "H-pos": hPos }),
    ...(hNegOk === undefined ? {} : { "H-negOK": hNegOk }),
    ...(hSilent === undefined ? {} : { "H-silent": hSilent }),
  };
}

export async function runExperiment(options: ExperimentOptions): Promise<ExperimentResult> {
  if (options.name.trim() === "") throw new RangeError("--name is required");
  if (options.change.trim() === "") throw new RangeError("--change is required");
  const rt = options.rt ?? 0;

  if (rt < 0 || rt >= 1) throw new RangeError("--rt must be in [0, 1)");
  const realDir = resolveRepo(options.real ?? "packages/cron/training/data/real");
  const accepted = rt > 0 ? loadAcceptedReal(realDir) : [];

  if (rt > 0 && accepted.length === 0) {
    throw new RangeError(
      `C1/C2 refused: no accepted real jsonl in ${realDir} (need agree:3, exec:true, roundtrip:true)`,
    );
  }

  if (rt > 0 && (options.dataPrefix !== undefined || options.kinds !== undefined)) {
    throw new RangeError("--rt mix requires generate path (no --data-prefix/--kinds)");
  }

  if (options.sweep) {
    let last: ExperimentResult | undefined;

    for (const entry of SWEEP) {
      last = await runExperiment({
        ...options,
        sweep: false,
        name: `${options.name}-${entry.id}`,
        freshEachEpoch:
          options.freshEachEpoch === true || entry.extra.includes("--fresh-each-epoch"),
      });
    }

    if (last === undefined) throw new RangeError("sweep list is empty");

    return last;
  }

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

  const shardsReadyOnDisk =
    existsSync(trainPrefix) && existsSync(valPrefix) && existsSync(heldPrefix);

  const steps: StepRecord[] = [];
  const samples = options.samples ?? 2000;
  const evalSamples = options.evalSamples ?? 500;
  const epochs = options.epochs ?? 2;
  const batch = options.batch ?? 1024;
  const oracleSeeds = options.oracleSeeds ?? 50;
  const kindsSample = options.kindsSample ?? 0;
  const device = options.device ?? "cpu";
  const stack = options.stack ?? "tagger-typed";
  const seedCount = Math.max(1, options.seeds ?? 1);
  const baseSeed = options.seed ?? 20260913;

  const bun = process.execPath;
  const generate = join(root, "packages/cron/training/src/generate.ts");
  const featurize = join(root, "packages/cron/training/src/featurize.ts");
  const oracle = join(root, "packages/cron/training/src/oracle.ts");
  const evaluate = join(root, "packages/cron/training/src/evaluate.ts");
  const humanEvaluate = join(root, "packages/cron/training/src/human-evaluate.ts");

  let skippedRegen = previous?.inputHash === inputHash && previous?.rt === rt && shardsReadyOnDisk;
  let renderedKinds: string[] = [];
  let skippedKinds: string[] = [];
  let rtLedger = "0";
  let kindsJsonlForOracle: string | undefined;

  const dataPrefix =
    options.dataPrefix === undefined ? join(work, "kinds") : resolveRepo(options.dataPrefix);

  if (options.dataPrefix !== undefined) {
    skippedRegen = true;
    skipStep(steps, "regen-data-prefix", [dataPrefix]);
  } else if (options.mixQuotas) {
    const pooledJsonl = join(work, "kinds.jsonl");

    kindsJsonlForOracle = pooledJsonl;

    if (skippedRegen && existsSync(pooledJsonl)) {
      skipStep(steps, "mix-quotas-generate", [pooledJsonl]);
      skipStep(steps, "featurize-mix", [dataPrefix]);
    } else {
      const mixLimit = options.mixLimit ?? 200_000;

      writeBackward({ out: pooledJsonl, limit: mixLimit });

      steps.push({
        step: "mix-quotas-generate",
        skipped: false,
        argv: [pooledJsonl, `--limit`, String(mixLimit)],
      });

      await featurizeKindsStreaming(pooledJsonl, dataPrefix);

      steps.push({
        step: "featurize-mix",
        skipped: false,
        argv: [pooledJsonl, dataPrefix],
      });

      skippedRegen = false;
      writeInputCache(cachePath, inputHash, rt);
    }
  } else if (options.kinds !== undefined || options.kindCap !== undefined) {
    const incremental = await runIncrementalKinds({
      dir: work,
      ...(options.kinds === undefined ? {} : { kinds: options.kinds }),
      ...(options.kindCap === undefined ? {} : { cap: options.kindCap }),
      steps,
    });

    skippedRegen = incremental.rendered.length === 0;
    renderedKinds = incremental.rendered;
    skippedKinds = incremental.skipped;
    // One pooled jsonl so the oracle's --kinds-sample scores the corpus we
    // are about to train on, not the legacy data/synth/kinds.jsonl.
    const pooledJsonl = join(work, "kinds.jsonl");

    if (!skippedRegen || !existsSync(pooledJsonl)) {
      const parts = incremental.plan.current
        .map((record) => record.jsonl)
        .filter((path) => existsSync(path));

      const cat = spawnSync("sh", ["-c", `cat "$@" > "${pooledJsonl}"`, "cat", ...parts], {
        stdio: "inherit",
      });

      if (cat.status !== 0) throw new Error("failed to pool per-kind jsonl files");
      steps.push({ step: "pool-kinds-jsonl", skipped: false, argv: [pooledJsonl] });
    } else {
      skipStep(steps, "pool-kinds-jsonl", [pooledJsonl]);
    }

    kindsJsonlForOracle = pooledJsonl;
    writeInputCache(cachePath, inputHash, rt);
  } else if (!skippedRegen) {
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

    if (rt > 0) {
      const mix = mixAcceptedIntoTrain(trainJsonl, accepted, rt);

      writeMix(work, mix);
      rtLedger = mix.fraction;

      steps.push({
        step: "mix-real",
        skipped: false,
        argv: ["mix", "--rt", String(rt), "--n-real", String(mix.nReal)],
      });
    } else {
      skipStep(steps, "mix-real", []);
    }

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

    writeInputCache(cachePath, inputHash, rt);
  } else {
    skipStep(steps, "regen-heldout", []);
    skipStep(steps, "regen-validation", []);
    skipStep(steps, "regen-train", []);
    skipStep(steps, "mix-real", []);
    skipStep(steps, "featurize-heldout", []);
    skipStep(steps, "featurize-validation", []);
    skipStep(steps, "featurize-train", []);
    if (rt > 0) rtLedger = readMixFraction(work) ?? String(rt);
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
      ...(kindsJsonlForOracle === undefined ? [] : ["--kinds-jsonl", kindsJsonlForOracle]),
      "--out",
      oracleOut,
    ],
    root,
    steps,
    "oracle",
  );

  if (stack === "seq2seq") {
    const jsonl = existsSync(join(work, "train.jsonl"))
      ? join(work, "train.jsonl")
      : resolveRepo("packages/cron/training/data/synth/kinds-typed.jsonl");

    const b1Out = join(work, "b1.json");

    spawnTool(
      [
        "uv",
        "run",
        "--project",
        join(root, "training/cron"),
        "python",
        join(root, "training/cron/seq2seq.py"),
        "--b1",
        "--jsonl",
        jsonl,
        "--proxy",
        String(samples),
        "--epochs",
        String(epochs),
        "--device",
        device,
        "--gold",
        resolveRepo("packages/cron/training/data/gold/reserved-carriers.jsonl"),
        "--human-dir",
        resolveRepo("packages/cron/evaluation"),
        "--out",
        b1Out,
      ],
      root,
      steps,
      "seq2seq-b1",
    );

    const b1 = existsSync(b1Out) ? readJson(b1Out) : {};
    const oracle = existsSync(oracleOut) ? readJson(oracleOut) : {};
    const oTotal = Number(oracle.total ?? 0);
    const oCorrect = Number(oracle.correct ?? 0);

    const metrics: LedgerMetrics = {
      EXEC: String(b1.EXEC ?? "n/a"),
      "FA@cov": String(b1["FA@cov"] ?? "n/a"),
      RT: rtLedger,
      stack: "seq2seq",
      "H-pos": String(b1["H-pos"] ?? "n/a"),
      "H-negOK": String(b1["H-negOK"] ?? "n/a"),
      "H-silent": String(b1["H-silent"] ?? "n/a"),
      ...(oTotal > 0 ? { O: `${oCorrect}/${oTotal}` } : {}),
    };

    writeFileSync(join(work, "steps.json"), `${JSON.stringify(steps, null, 2)}\n`);
    const ledgerPath = resolveRepo(options.ledger ?? "EXPERIMENTS.md");

    appendLedgerRow(ledgerPath, { ...options, seeds: seedCount }, metrics);

    return {
      name: options.name,
      inputHash,
      skippedRegen,
      skippedFeaturize: skippedRegen,
      steps,
      ledgerPath,
      work,
      renderedKinds,
      skippedKinds,
      seedRuns: seedCount,
    };
  }

  const storage = resolveRepo(options.storage ?? join(work, "runs"));
  const seedMetrics: LedgerMetrics[] = [];

  for (let seedIndex = 0; seedIndex < seedCount; seedIndex++) {
    const seed = baseSeed + seedIndex;
    const runName = seedCount === 1 ? options.name : `${options.name}-seed-${seed}`;

    spawnTool(
      [
        "uv",
        "run",
        "--project",
        join(root, "training/cron"),
        "python",
        join(root, "training/cron/train.py"),
        "--run",
        runName,
        "--seed",
        String(seed),
        "--storage",
        storage,
        "--data-prefix",
        dataPrefix,
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
        "--h-per-epoch",
        "--early-stopping-patience",
        String(options.earlyStoppingPatience ?? 2),
        ...(options.freshEachEpoch ? ["--fresh-each-epoch"] : []),
        ...(options.init === undefined ? [] : ["--init", options.init]),
        ...(options.trainArgs ?? []),
      ],
      root,
      steps,
      `train-seed-${seed}`,
    );

    const checkpoint = join(storage, runName, "best.pt");
    const candidate = join(work, `weights-seed-${seed}.gen.ts`);

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
      `export-seed-${seed}`,
    );

    const evalOut = join(work, `eval-seed-${seed}.json`);

    try {
      spawnTool(
        [bun, evaluate, "--weights", candidate, "--out", evalOut],
        root,
        steps,
        `evaluate-seed-${seed}`,
      );
    } catch {
      steps[steps.length - 1]!.skipped = false;
    }

    const humanOut = join(work, `h-seed-${seed}.json`);

    try {
      spawnTool(
        [
          bun,
          humanEvaluate,
          "--partition",
          "regression",
          "--weights",
          candidate,
          "--out",
          humanOut,
        ],
        root,
        steps,
        `human-evaluate-seed-${seed}`,
      );
    } catch {
      steps[steps.length - 1]!.skipped = false;
    }

    seedMetrics.push(
      metricsFromReports(
        oracleOut,
        existsSync(evalOut) ? evalOut : undefined,
        existsSync(humanOut) ? humanOut : undefined,
        stack,
        rtLedger,
      ),
    );
  }

  writeFileSync(stepsPath, `${JSON.stringify(steps, null, 2)}\n`);

  const metrics = aggregateSeedMetrics(seedMetrics, stack);
  const ledgerPath = resolveRepo(options.ledger ?? "EXPERIMENTS.md");

  appendLedgerRow(ledgerPath, { ...options, seeds: seedCount }, metrics);

  return {
    name: options.name,
    inputHash,
    skippedRegen,
    skippedFeaturize: skippedRegen,
    steps,
    ledgerPath,
    work,
    renderedKinds,
    skippedKinds,
    seedRuns: seedCount,
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
      case "--seed":
        options.seed = Number(take());
        break;

      case "--kinds": {
        const value = take();

        // `all` = every kind through the parallel incremental path, uncapped.
        options.kinds =
          value.trim() === "all"
            ? kindIds()
            : (value
                .split(",")
                .map((part) => part.trim())
                .filter((part) => part.length > 0) as KindId[]);

        break;
      }

      case "--kind-cap":
        options.kindCap = Number(take());
        break;
      case "--mix-quotas":
        options.mixQuotas = true;
        break;
      case "--mix-limit":
        options.mixLimit = Number(take());
        break;
      case "--sweep":
        options.sweep = true;
        break;
      case "--early-stopping-patience":
        options.earlyStoppingPatience = Number(take());
        break;
      case "--init":
        options.init = take();
        break;
      case "--data-prefix":
        options.dataPrefix = take();
        break;
      case "--rt":
        options.rt = Number(take());
        break;
      case "--real":
        options.real = take();
        break;
      case "--fresh-each-epoch":
        options.freshEachEpoch = true;
        break;
      case "--train-arg":
        (options.trainArgs ??= []).push(take());
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

export async function main(argv = process.argv.slice(2)): Promise<ExperimentResult> {
  const options = parseArgs(argv);
  const result = await runExperiment(options);
  const skip = result.skippedRegen ? "skip regen/featurize" : "inputs changed";

  process.stdout.write(
    `${result.name}: ${skip} hash=${result.inputHash.slice(0, 12)} seeds=${result.seedRuns} ledger=${result.ledgerPath}\n`,
  );

  return result;
}

if (import.meta.main) await main();
