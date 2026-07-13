/**
 * Corpus generator CLI.
 *
 * Draws weighted families, renders each one for the requested split, and emits
 * one JSON record per line. Determinism is total: every example comes from the
 * run seed, and held-out/validation runs default to frozen seeds so their
 * membership cannot drift between generator runs. Alongside the corpus it
 * writes `<out>.fingerprints.json` — the split's sorted fingerprints plus a
 * hash of the generator sources — which later phases and the train/validation
 * guard consume as their `--exclude` set.
 */
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { SpeechAct } from "../../src/model/speech-acts";
import type { ScheduleIntent } from "../../src/types/intent";
import { fingerprintExample } from "./fingerprint";
import type { RenderedExample } from "./renderer";
import { renderIntent, renderNegative } from "./renderer";
import type { Rng } from "./rng";
import { createRng } from "./rng";
import type { Family } from "./sampler";
import { FAMILY_WEIGHTS, sampleIntent } from "./sampler";
import type { Span } from "./sentence";
import { renderHeldout } from "./splits";

export type Split = "train" | "validation" | "heldout";

export interface CorpusRecord {
  id: string;
  split: Split;
  seed: number;
  family: Family;
  template: string;
  fingerprint: string;
  text: string;
  intent: ScheduleIntent | null;
  speechAct: SpeechAct;
  spans: Span[];
}

export interface GenerateOptions {
  split: Split;
  count: number;
  seed: number;
  out: string;
  exclude?: string;
}

export interface GenerateResult {
  records: CorpusRecord[];
  fingerprints: string[];
  generatorSha256: string;
  excluded: number;
  out: string;
  fingerprintsPath: string;
}

/** Frozen default seeds: held-out and validation must be reproducible forever. */
export const FROZEN_SEEDS: Record<Split, number> = {
  train: 20260913,
  validation: 20260914,
  heldout: 20260915,
};

/** Default record counts per split for the package scripts. */
export const DEFAULT_COUNTS: Record<Split, number> = {
  train: 4000,
  validation: 500,
  heldout: 500,
};

const SPLITS: readonly Split[] = ["train", "validation", "heldout"];

/**
 * Generator sources hashed into the artifact so provenance can detect drift.
 * Order is normalized before hashing, so the list may stay readable.
 */
const GENERATOR_SOURCES: readonly string[] = [
  "background.ts",
  "correction.ts",
  "discourse.ts",
  "fingerprint.ts",
  "generate.ts",
  "renderer.ts",
  "rng.ts",
  "sampler.ts",
  "sentence.ts",
  "splits.ts",
];

/** Hash the generator sources, path-tagged and sorted for stable output. */
export function generatorSha256(): string {
  const base = dirname(fileURLToPath(import.meta.url));
  const digest = createHash("sha256");

  for (const name of [...GENERATOR_SOURCES].sort()) {
    digest.update(name, "utf8");
    digest.update("\0", "utf8");
    digest.update(readFileSync(join(base, name)));
    digest.update("\0", "utf8");
  }

  return digest.digest("hex");
}

interface Rendered {
  example: RenderedExample;
  intent: ScheduleIntent | null;
}

function render(rng: Rng, family: Family, split: Split): Rendered {
  if (split === "heldout") {
    const heldout = renderHeldout(family, rng);

    return { example: heldout, intent: heldout.intent };
  }

  const sample = sampleIntent(rng, family);

  const example =
    sample.intent === null ? renderNegative(rng) : renderIntent(sample.intent, family, rng);

  return { example, intent: sample.intent };
}

function loadExcluded(path: string | undefined): ReadonlySet<string> {
  if (path === undefined) return new Set<string>();

  const parsed = JSON.parse(readFileSync(path, "utf8")) as {
    fingerprints?: unknown;
  };

  if (!Array.isArray(parsed.fingerprints)) {
    throw new TypeError(`exclude file has no fingerprints array: ${path}`);
  }

  return new Set(parsed.fingerprints.map((value) => String(value)));
}

/**
 * Generate one split. Returns the records in draw order plus the sorted unique
 * fingerprint set. Non-held-out splits drop any example whose fingerprint is in
 * `exclude` and count it; held-out ignores `exclude`.
 */
export function generate(options: GenerateOptions): GenerateResult {
  const excluded = loadExcluded(options.exclude);
  const families = Object.entries(FAMILY_WEIGHTS) as Array<[Family, number]>;
  const rng = createRng(options.seed);
  const records: CorpusRecord[] = [];
  let skipped = 0;
  const limit = options.count * 1000 + 10000;

  for (let attempts = 0; records.length < options.count && attempts < limit; attempts++) {
    const family = rng.weighted(families);
    const { example, intent } = render(rng, family, options.split);
    const fingerprint = fingerprintExample(example);

    if (options.split !== "heldout" && excluded.has(fingerprint)) {
      skipped++;
      continue;
    }

    records.push({
      id: `${options.split}-${String(records.length).padStart(6, "0")}`,
      split: options.split,
      seed: options.seed,
      family,
      template: example.template,
      fingerprint,
      text: example.text,
      intent,
      speechAct: example.speechAct,
      spans: example.spans,
    });
  }

  if (records.length < options.count) {
    throw new Error(`generate: emitted ${records.length}/${options.count} (${skipped} excluded)`);
  }

  mkdirSync(dirname(options.out), { recursive: true });
  const body = records.map((record) => JSON.stringify(record)).join("\n");

  writeFileSync(options.out, body.length > 0 ? `${body}\n` : "");

  const fingerprints = [...new Set(records.map((record) => record.fingerprint))].sort();
  const fingerprintsPath = `${options.out}.fingerprints.json`;

  writeFileSync(
    fingerprintsPath,
    `${JSON.stringify({ fingerprints, generatorSha256: generatorSha256() }, null, 2)}\n`,
  );

  return {
    records,
    fingerprints,
    generatorSha256: generatorSha256(),
    excluded: skipped,
    out: options.out,
    fingerprintsPath,
  };
}

function parseSplit(value: string): Split {
  if ((SPLITS as readonly string[]).includes(value)) return value as Split;
  throw new RangeError(`unknown split: ${value}`);
}

function readOption(argv: readonly string[], index: number): string {
  const value = argv[index + 1];

  if (value === undefined) throw new RangeError(`missing value for ${argv[index]}`);

  return value;
}

/** Parse CLI flags; unset `--seed`/`--count` fall back to the frozen defaults. */
export function parseArgs(argv: readonly string[]): GenerateOptions {
  let split: Split | undefined;
  let count: number | undefined;
  let seed: number | undefined;
  let out: string | undefined;
  let exclude: string | undefined;

  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index]!;

    switch (arg) {
      case "--split":
        split = parseSplit(readOption(argv, index));
        index++;
        break;
      case "--count":
        count = Number(readOption(argv, index));
        index++;
        break;
      case "--seed":
        seed = Number(readOption(argv, index));
        index++;
        break;
      case "--out":
        out = readOption(argv, index);
        index++;
        break;
      case "--exclude":
        exclude = readOption(argv, index);
        index++;
        break;
      default:
        throw new RangeError(`unknown flag: ${arg}`);
    }
  }

  const resolved = split ?? "train";

  if (count !== undefined && (!Number.isInteger(count) || count < 0)) {
    throw new RangeError(`--count must be a non-negative integer: ${count}`);
  }

  if (seed !== undefined && !Number.isInteger(seed)) {
    throw new RangeError(`--seed must be an integer: ${seed}`);
  }

  return {
    split: resolved,
    count: count ?? DEFAULT_COUNTS[resolved],
    seed: seed ?? FROZEN_SEEDS[resolved],
    out: out ?? join("data", "synth", `${resolved}.jsonl`),
    ...(exclude === undefined ? {} : { exclude }),
  };
}

if (import.meta.main) {
  const options = parseArgs(process.argv.slice(2));
  const result = generate(options);

  process.stdout.write(
    `${options.split}: ${result.records.length} records, ` +
      `${result.fingerprints.length} fingerprints, ${result.excluded} excluded -> ${result.out}\n`,
  );
}
