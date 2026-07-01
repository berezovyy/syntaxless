/**
 * Opencode corpus runner.
 *
 * A fixed, small worker pool drives the angle registry through real sessions,
 * retries only transient network failures with capped backoff, writes one JSON
 * candidate per line, and records a manifest. Parse failures are terminal (and
 * already counted by the workflow); retrying them would just repay for the same
 * bad text. The pool width is a module constant on purpose: reproducibility and
 * provider rate limits both prefer a fixed fan-out.
 */
import { randomUUID } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import { ANGLES } from "./angles.js";
import type { Angle } from "./angles.js";
import { close, model } from "./client.js";
import { runnerSha256, writeManifest } from "./manifest.js";
import type { RunManifest } from "./manifest.js";
import type { AngleRun } from "./workflow.js";
import { runAngle } from "./workflow.js";

/** Fixed pool width. Not a flag: a run should look the same every time. */
export const WORKERS = 4;
export const MAX_ATTEMPTS = 3;
export const BASE_BACKOFF_MS = 250;
export const MAX_BACKOFF_MS = 4000;

const TRANSIENT_CODES: ReadonlySet<string> = new Set([
  "ECONNRESET",
  "ECONNREFUSED",
  "ETIMEDOUT",
  "EAI_AGAIN",
  "ENOTFOUND",
  "EPIPE",
]);

/** A failure worth retrying: 429/5xx, a known socket code, or a fetch error. */
export function isTransient(error: unknown): boolean {
  const record = error as {
    status?: unknown;
    statusCode?: unknown;
    code?: unknown;
    message?: unknown;
  } | null;

  const status =
    typeof record?.status === "number"
      ? record.status
      : typeof record?.statusCode === "number"
        ? record.statusCode
        : undefined;

  if (status !== undefined) return status === 429 || status >= 500;

  if (typeof record?.code === "string" && TRANSIENT_CODES.has(record.code)) {
    return true;
  }

  const message =
    error instanceof Error
      ? error.message.toLowerCase()
      : typeof record?.message === "string"
        ? record.message.toLowerCase()
        : "";

  return (
    message.includes("fetch failed") || message.includes("network") || message.includes("timeout")
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Retry `fn` while failures are transient, up to `attempts` tries. */
export async function withRetry<T>(fn: () => Promise<T>, attempts = MAX_ATTEMPTS): Promise<T> {
  let lastError: unknown;

  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt >= attempts || !isTransient(error)) throw error;
      const backoff = Math.min(BASE_BACKOFF_MS * 2 ** (attempt - 1), MAX_BACKOFF_MS);

      await sleep(backoff);
    }
  }

  throw lastError;
}

/** Run `worker` over `items` with a fixed pool, preserving input order. */
export async function runPool<T, R>(
  items: readonly T[],
  worker: (item: T) => Promise<R>,
  workers = WORKERS,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  const width = Math.max(1, Math.min(workers, items.length || 1));

  async function drain(): Promise<void> {
    for (;;) {
      const index = next++;

      if (index >= items.length) return;
      results[index] = await worker(items[index]!);
    }
  }

  await Promise.all(Array.from({ length: width }, () => drain()));

  return results;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Match an angle pattern. `*` matches any run of characters, and a pattern also
 * matches deeper ids, so `ops/weekday-list` selects every registration under it
 * and `ops/*` selects the whole ops domain.
 */
export function matchesAngle(pattern: string, angleId: string): boolean {
  const source = pattern.split("*").map(escapeRegExp).join(".*");

  return new RegExp(`^${source}(?:/.*)?$`).test(angleId);
}

/** Select registry angles matching any pattern, in registry order. */
export function selectAngles(
  patterns: readonly string[],
  angles: readonly Angle[] = ANGLES,
): Angle[] {
  return angles.filter((angle) => patterns.some((pattern) => matchesAngle(pattern, angle.id)));
}

export interface RunOptions {
  patterns: string[];
  count: number;
  out: string;
}

function readOption(argv: readonly string[], index: number): string {
  const value = argv[index + 1];

  if (value === undefined) throw new RangeError(`missing value for ${argv[index]}`);

  return value;
}

/** Parse `--angles <glob,glob> --count <n> --out <path>`. */
export function parseArgs(argv: readonly string[]): RunOptions {
  let angles: string | undefined;
  let count: number | undefined;
  let out: string | undefined;

  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index]!;

    switch (arg) {
      case "--angles":
        angles = readOption(argv, index);
        index++;
        break;
      case "--count":
        count = Number(readOption(argv, index));
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

  const patterns = (angles ?? "")
    .split(",")
    .map((pattern) => pattern.trim())
    .filter((pattern) => pattern.length > 0);

  if (patterns.length === 0) throw new RangeError("--angles is required");

  if (count === undefined || !Number.isInteger(count) || count < 1) {
    throw new RangeError("--count must be a positive integer");
  }

  if (out === undefined || out.length === 0) throw new RangeError("--out is required");

  return { patterns, count, out };
}

export interface CorpusResult {
  runs: AngleRun[];
  manifest: RunManifest;
  out: string;
  manifestPath: string;
}

/** Run every selected angle, write the raw JSONL, then write the manifest. */
export async function runCorpus(
  options: RunOptions,
  angles: readonly Angle[] = ANGLES,
): Promise<CorpusResult> {
  const selected = selectAngles(options.patterns, angles);

  if (selected.length === 0) {
    throw new RangeError(`no angles match: ${options.patterns.join(", ")}`);
  }

  const runId = `${Date.now().toString(36)}-${randomUUID().slice(0, 8)}`;
  const ctx = { runId, count: options.count };
  const runs = await runPool(selected, (angle) => withRetry(() => runAngle(angle, ctx)));

  const lines = runs.flatMap((run) => run.candidates.map((candidate) => JSON.stringify(candidate)));

  mkdirSync(dirname(options.out), { recursive: true });
  writeFileSync(options.out, lines.length > 0 ? `${lines.join("\n")}\n` : "");

  const manifest: RunManifest = {
    runId,
    model: model(),
    angles: selected.map((angle) => ({
      angleId: angle.id,
      promptId: angle.promptId,
      promptVersion: angle.promptVersion,
    })),
    requested: selected.length * options.count,
    produced: runs.reduce((total, run) => total + run.candidates.length, 0),
    rejected: runs.reduce((total, run) => total + run.rejected, 0),
    cacheHits: runs.filter((run) => run.cacheHit).length,
    runnerSha256: runnerSha256(),
  };

  const manifestPath = writeManifest(manifest);

  return { runs, manifest, out: options.out, manifestPath };
}

if (import.meta.main) {
  const options = parseArgs(process.argv.slice(2));

  try {
    const result = await runCorpus(options);

    process.stdout.write(
      `corpus: ${result.manifest.produced} candidates from ` +
        `${result.manifest.angles.length} angles ` +
        `(${result.manifest.cacheHits} cache hits, ` +
        `${result.manifest.rejected} rejected) -> ${result.out}\n` +
        `manifest -> ${result.manifestPath}\n`,
    );
  } finally {
    await close();
  }
}
