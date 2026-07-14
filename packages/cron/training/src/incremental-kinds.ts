/**
 * Incremental per-kind regen + featurize (STACK-PLAN S.2).
 *
 * Unchanged KindDef hashes are no-ops. Stale kinds render and featurize in
 * parallel Bun.spawn workers, then shards concatenate into one Dataset prefix.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { writeBackward } from "./backward";
import { concatKindPrefixes } from "./concat-shards";
import { featurizeKindsStreaming } from "./featurize-kinds";
import { planKindRegen, shardsReady } from "./kind-hash";
import type { KindHashRecord, KindRegenPlan } from "./kind-hash";
import type { KindId } from "./kinds";

export interface RegenStep {
  step: string;
  skipped: boolean;
  argv: string[];
}

const SPLITS = ["train", "validation", "heldout"] as const;

function repoRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
}

function resolveRepo(path: string): string {
  return isAbsolute(path) ? path : resolve(repoRoot(), path);
}

export interface IncrementalKindsOptions {
  dir: string;
  kinds?: readonly KindId[];
  cap?: number;
  workers?: number;
  steps?: RegenStep[];
}

export interface IncrementalKindsResult {
  plan: KindRegenPlan;
  skipped: string[];
  rendered: string[];
}

async function renderAndFeaturize(record: KindHashRecord, cap?: number): Promise<void> {
  mkdirSync(dirname(record.jsonl), { recursive: true });

  writeBackward({
    out: record.jsonl,
    kinds: [record.id],
    pureHashSplit: true,
    ...(cap === undefined ? {} : { cap }),
  });

  await featurizeKindsStreaming(record.jsonl, record.prefix);
}

const WORKER_CONCURRENCY = 16;

async function spawnWorkers(
  stale: readonly KindHashRecord[],
  dir: string,
  cap: number | undefined,
  workers: number,
  steps: RegenStep[],
): Promise<void> {
  const bun = process.execPath;
  const self = fileURLToPath(import.meta.url);
  const root = repoRoot();
  const queue = [...stale];
  const running: Promise<void>[] = [];

  const launch = (record: KindHashRecord): Promise<void> => {
    const argv = [
      bun,
      self,
      "--worker",
      "--kind",
      record.id,
      "--hash",
      record.hash,
      "--dir",
      dir,
      ...(cap === undefined ? [] : ["--cap", String(cap)]),
    ];

    steps.push({ step: `regen-kind:${record.id}`, skipped: false, argv });
    const proc = Bun.spawn(argv, { cwd: root, stdout: "pipe", stderr: "pipe" });

    return proc.exited.then(async (code) => {
      if (code !== 0) {
        const err = await new Response(proc.stderr).text();
        const out = await new Response(proc.stdout).text();

        throw new Error(`regen-kind ${record.id} failed: ${err}${out}`);
      }
    });
  };

  while (queue.length > 0 || running.length > 0) {
    while (queue.length > 0 && running.length < workers) {
      const record = queue.shift()!;

      const task = launch(record).finally(() => {
        const index = running.indexOf(task);

        if (index >= 0) running.splice(index, 1);
      });

      running.push(task);
    }

    if (running.length > 0) await Promise.race(running);
  }
}

export async function runIncrementalKinds(
  options: IncrementalKindsOptions,
): Promise<IncrementalKindsResult> {
  const dir = resolveRepo(options.dir);

  mkdirSync(dir, { recursive: true });
  const plan = planKindRegen(dir, options.kinds);
  const steps = options.steps ?? [];
  const skipped: string[] = [];
  const rendered: string[] = [];

  for (const record of plan.ready) {
    skipped.push(record.id);

    steps.push({
      step: `regen-kind:${record.id}`,
      skipped: true,
      argv: [],
    });
  }

  const workers = Math.max(1, options.workers ?? WORKER_CONCURRENCY);

  if (plan.stale.length > 0) {
    await spawnWorkers(plan.stale, dir, options.cap, workers, steps);
    rendered.push(...plan.stale.map((record) => record.id));
  }

  for (const record of plan.stale) {
    if (!shardsReady(record.prefix)) throw new Error(`kind ${record.id} did not produce shards`);
  }

  const ready = planKindRegen(dir, options.kinds).current;

  for (const split of SPLITS) {
    concatKindPrefixes(
      ready.map((record) => `${record.prefix}-${split}`),
      join(dir, `kinds-${split}`),
    );
  }

  writeFileSync(
    join(dir, "kind-hashes.json"),
    `${JSON.stringify(
      ready.map((record) => ({ id: record.id, hash: record.hash })),
      null,
      2,
    )}\n`,
  );

  return { plan, skipped, rendered };
}

function parseWorkerArgs(argv: string[]): {
  kind: KindId;
  hash: string;
  dir: string;
  cap?: number;
} {
  let kind: string | undefined;
  let hash: string | undefined;
  let dir: string | undefined;
  let cap: number | undefined;

  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index]!;
    const next = argv[index + 1];

    if (arg === "--kind") {
      kind = next;
      index++;
    } else if (arg === "--hash") {
      hash = next;
      index++;
    } else if (arg === "--dir") {
      dir = next;
      index++;
    } else if (arg === "--cap") {
      cap = Number(next);
      index++;
    }
  }

  if (kind === undefined || hash === undefined || dir === undefined) {
    throw new RangeError("worker requires --kind --hash --dir");
  }

  return { kind: kind as KindId, hash, dir, ...(cap === undefined ? {} : { cap }) };
}

if (import.meta.main) {
  const argv = process.argv.slice(2);

  if (argv[0] === "--worker") {
    const options = parseWorkerArgs(argv.slice(1));

    const record: KindHashRecord = {
      id: options.kind,
      hash: options.hash,
      jsonl: join(options.dir, `${options.kind}.${options.hash}.jsonl`),
      prefix: join(options.dir, `${options.kind}.${options.hash}`),
    };

    await renderAndFeaturize(record, options.cap);
  } else {
    const dir = argv[0] ?? "packages/cron/training/data/synth/kinds-incremental";

    const kinds = argv.includes("--kinds")
      ? (argv[argv.indexOf("--kinds") + 1]!.split(",") as KindId[])
      : undefined;

    const capIndex = argv.indexOf("--cap");
    const cap = capIndex >= 0 ? Number(argv[capIndex + 1]) : undefined;

    const result = await runIncrementalKinds({
      dir,
      ...(kinds === undefined ? {} : { kinds }),
      ...(cap === undefined ? {} : { cap }),
    });

    process.stdout.write(
      `incremental-kinds: rendered ${result.rendered.join(",") || "none"} skipped ${result.skipped.length}\n`,
    );
  }
}
