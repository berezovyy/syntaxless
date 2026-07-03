/**
 * Backward corpus generator CLI.
 *
 * Enumerates the sentence-kind catalog (schedule shape × wording template)
 * instead of sampling families or calling an LLM. Each example is labeled at
 * insertion time, fingerprinted, and written as one JSON record per line in
 * the Phase 08 generate.ts shape so featurize and the Python harness can
 * consume it without a new contract.
 *
 * Held-out templates are different frames of the same kinds. A slice of the
 * train templates is re-tagged validation so the three splits are produced
 * from one expansion.
 */
import { createHash } from "node:crypto";
import { closeSync, mkdirSync, openSync, readFileSync, writeFileSync, writeSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { coarseText, fingerprintExample, structuralPattern } from "./fingerprint";
import type { CorpusRecord, Split } from "./generate";
import { expandKinds, iterateKinds, kindIds } from "./kinds";
import type { ExampleSplit, KindExample, KindId } from "./kinds";

/** Provenance seed stored on every record; the expansion is not random. */
export const BACKWARD_SEED = 20260913;

/** Versioned split salt. Never depend on row order, values, or --kinds filters. */
export const SPLIT_VERSION = "kinds-frames-v1";

const SPLITS: readonly Split[] = ["train", "validation", "heldout"];

const GENERATOR_SOURCES: readonly string[] = [
  "../../src/parser/lexicon.ts",
  "../../src/parser/decode.ts",
  "backward.ts",
  "fingerprint.ts",
  "kinds.ts",
  "sentence.ts",
];

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

export interface BackwardOptions {
  split?: Split;
  out: string;
  kinds?: readonly KindId[];
}

export interface BackwardResult {
  records: CorpusRecord[];
  fingerprints: string[];
  generatorSha256: string;
  out: string;
  fingerprintsPath: string;
  byKind: Record<string, number>;
  bySplit: Record<Split, number>;
}

export type BackwardSummary = Omit<BackwardResult, "records">;

function parseSplit(value: string): Split {
  if ((SPLITS as readonly string[]).includes(value)) return value as Split;
  throw new RangeError(`unknown split: ${value}`);
}

function readOption(argv: readonly string[], index: number): string {
  const value = argv[index + 1];

  if (value === undefined) throw new RangeError(`missing value for ${argv[index]}`);

  return value;
}

/** Parse `--split`, `--out`, `--kinds`. Unset `--split` emits every split. */
export function parseArgs(argv: readonly string[]): BackwardOptions {
  let split: Split | undefined;
  let out: string | undefined;
  let kinds: KindId[] | undefined;

  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index]!;

    switch (arg) {
      case "--split":
        split = parseSplit(readOption(argv, index));
        index++;
        break;
      case "--out":
        out = readOption(argv, index);
        index++;
        break;
      case "--kinds":
        kinds = readOption(argv, index)
          .split(",")
          .map((value) => value.trim())
          .filter((value) => value.length > 0) as KindId[];

        if (kinds.length === 0) throw new RangeError("--kinds is empty");
        index++;
        break;
      default:
        throw new RangeError(`unknown flag: ${arg}`);
    }
  }

  return {
    out: out ?? join("data", "synth", "kinds.jsonl"),
    ...(split === undefined ? {} : { split }),
    ...(kinds === undefined ? {} : { kinds }),
  };
}

/** Carrier wording cannot turn a known schedule construction into a new frame.
 * Negatives have no schedule slots; group them by normalized prose instead. */
export function frameKey(example: KindExample): string {
  if (example.intent === null) return `negative:${coarseText(example.example.text)}`;

  return JSON.stringify(
    structuralPattern(example.example.text, example.example.spans).filter((part) => part !== "O"),
  );
}

/** Assign entire frames, globally and deterministically, rather than rows.
 * Roughly 75% of frames train, 12.5% validation, 12.5% held-out. */
export function assignSplits(examples: readonly KindExample[]): KindExample[] {
  return examples.map(assignSplit);
}

export function assignSplit(example: KindExample): KindExample {
  const bucket =
    createHash("sha256")
      .update(`${SPLIT_VERSION}\0${frameKey(example)}`)
      .digest()
      .readUInt32BE(0) % 8;

  const split: ExampleSplit = bucket === 6 ? "validation" : bucket === 7 ? "heldout" : "train";

  return { ...example, split };
}

function toRecord(example: KindExample, index: number): CorpusRecord {
  const split = example.split;

  return {
    id: `${split}-${String(index).padStart(6, "0")}`,
    split,
    seed: BACKWARD_SEED,
    family: example.family,
    template: example.template,
    fingerprint: fingerprintExample(example.example),
    text: example.example.text,
    intent: example.intent,
    spans: example.example.spans,
  };
}

function countByKind(records: readonly CorpusRecord[]): Record<string, number> {
  const counts: Record<string, number> = {};

  for (const record of records) {
    const kind = record.template.split("/")[0] ?? record.template;

    counts[kind] = (counts[kind] ?? 0) + 1;
  }

  return counts;
}

function countBySplit(records: readonly CorpusRecord[]): Record<Split, number> {
  const counts: Record<Split, number> = { train: 0, validation: 0, heldout: 0 };

  for (const record of records) counts[record.split]++;

  return counts;
}

/** Expand the catalog, assign splits, and write JSONL plus fingerprints. */
export function generateBackward(options: BackwardOptions): BackwardResult {
  const wanted = options.kinds === undefined ? undefined : new Set(options.kinds);

  const expanded = expandKinds().filter(
    (example) => wanted === undefined || wanted.has(example.kind),
  );

  if (wanted !== undefined) {
    for (const id of wanted) {
      if (!expanded.some((example) => example.kind === id)) {
        throw new RangeError(`unknown or empty kind: ${id}`);
      }
    }
  }

  const tagged = assignSplits(expanded);

  const selected =
    options.split === undefined
      ? tagged
      : tagged.filter((example) => example.split === options.split);

  const records = selected.map((example, index) => toRecord(example, index));

  mkdirSync(dirname(options.out), { recursive: true });
  const body = records.map((record) => JSON.stringify(record)).join("\n");

  writeFileSync(options.out, body.length > 0 ? `${body}\n` : "");

  const fingerprints = [...new Set(records.map((record) => record.fingerprint))].sort();
  const fingerprintsPath = `${options.out}.fingerprints.json`;
  const sha = generatorSha256();

  writeFileSync(
    fingerprintsPath,
    `${JSON.stringify({ fingerprints, generatorSha256: sha }, null, 2)}\n`,
  );

  return {
    records,
    fingerprints,
    generatorSha256: sha,
    out: options.out,
    fingerprintsPath,
    byKind: countByKind(records),
    bySplit: countBySplit(records),
  };
}

/** Stream a large catalog to JSONL while retaining only compact summary data. */
export function writeBackward(options: BackwardOptions): BackwardSummary {
  const wanted = options.kinds === undefined ? undefined : new Set(options.kinds);

  if (wanted !== undefined) {
    const known = new Set(kindIds());

    for (const id of wanted) {
      if (!known.has(id)) throw new RangeError(`unknown or empty kind: ${id}`);
    }
  }

  mkdirSync(dirname(options.out), { recursive: true });
  const file = openSync(options.out, "w");
  const fingerprints = new Set<string>();
  const byKind: Record<string, number> = {};
  const bySplit: Record<Split, number> = { train: 0, validation: 0, heldout: 0 };
  let index = 0;
  let chunk = "";

  try {
    for (const expanded of iterateKinds()) {
      if (wanted !== undefined && !wanted.has(expanded.kind)) continue;
      const example = assignSplit(expanded);

      if (options.split !== undefined && example.split !== options.split) continue;
      const record = toRecord(example, index++);

      chunk += `${JSON.stringify(record)}\n`;
      fingerprints.add(record.fingerprint);
      byKind[example.kind] = (byKind[example.kind] ?? 0) + 1;
      bySplit[record.split]++;

      if (chunk.length >= 4 * 1024 * 1024) {
        writeSync(file, chunk);
        chunk = "";
      }
    }

    if (chunk.length > 0) writeSync(file, chunk);
  } finally {
    closeSync(file);
  }

  const sortedFingerprints = [...fingerprints].sort();
  const fingerprintsPath = `${options.out}.fingerprints.json`;
  const sha = generatorSha256();

  writeFileSync(
    fingerprintsPath,
    `${JSON.stringify({ fingerprints: sortedFingerprints, generatorSha256: sha }, null, 2)}\n`,
  );

  return {
    fingerprints: sortedFingerprints,
    generatorSha256: sha,
    out: options.out,
    fingerprintsPath,
    byKind,
    bySplit,
  };
}

if (import.meta.main) {
  const options = parseArgs(process.argv.slice(2));
  const result = writeBackward(options);

  const kinds = Object.entries(result.byKind)
    .map(([kind, count]) => `${kind}=${count}`)
    .join(" ");

  process.stdout.write(
    `kinds: ${Object.values(result.bySplit).reduce((sum, count) => sum + count, 0)} records ` +
      `(train ${result.bySplit.train}, ` +
      `validation ${result.bySplit.validation}, ` +
      `heldout ${result.bySplit.heldout}) ` +
      `-> ${result.out}\n` +
      `${kinds}\n`,
  );
}
