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

import { coarseText, fingerprintExample, fingerprintPattern, framePattern } from "./fingerprint";
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
  "backward.ts",
  "correction.ts",
  "discourse.ts",
  "fingerprint.ts",
  "kinds.ts",
  "noise.ts",
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
  /** Stop after this many examples per kind (tiny fixtures / pool draw). */
  cap?: number;
  /** Stop after this many records total. */
  limit?: number;
}

export interface BackwardResult {
  records: CorpusRecord[];
  fingerprints: string[];
  fingerprintsBySplit: Record<Split, string[]>;
  carrierHashesBySplit: Record<Split, string[]>;
  generatorSha256: string;
  out: string;
  fingerprintsPath: string;
  byKind: Record<string, number>;
  bySplit: Record<Split, number>;
}

export type BackwardSummary = Omit<BackwardResult, "records">;

interface SplitMetadataSets {
  fingerprints: Record<Split, Set<string>>;
  carrierHashes: Record<Split, Set<string>>;
}

function splitMetadataSets(): SplitMetadataSets {
  return {
    fingerprints: {
      train: new Set<string>(),
      validation: new Set<string>(),
      heldout: new Set<string>(),
    },
    carrierHashes: {
      train: new Set<string>(),
      validation: new Set<string>(),
      heldout: new Set<string>(),
    },
  };
}

function addSplitMetadata(metadata: SplitMetadataSets, record: CorpusRecord): void {
  metadata.fingerprints[record.split].add(record.fingerprint);

  for (const entry of fingerprintPattern(record.text, record.spans)) {
    if (/^[0-9a-f]{12}$/.test(entry)) metadata.carrierHashes[record.split].add(entry);
  }
}

function sortedSplitMetadata(metadata: SplitMetadataSets): {
  fingerprintsBySplit: Record<Split, string[]>;
  carrierHashesBySplit: Record<Split, string[]>;
} {
  return {
    fingerprintsBySplit: Object.fromEntries(
      SPLITS.map((split) => [split, [...metadata.fingerprints[split]].sort()]),
    ) as Record<Split, string[]>,
    carrierHashesBySplit: Object.fromEntries(
      SPLITS.map((split) => [split, [...metadata.carrierHashes[split]].sort()]),
    ) as Record<Split, string[]>,
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

/** Parse `--split`, `--out`, `--kinds`. Unset `--split` emits every split. */
export function parseArgs(argv: readonly string[]): BackwardOptions {
  let split: Split | undefined;
  let out: string | undefined;
  let kinds: KindId[] | undefined;
  let cap: number | undefined;
  let limit: number | undefined;

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
      case "--cap":
        cap = Number(readOption(argv, index));
        index++;
        break;
      case "--limit":
        limit = Number(readOption(argv, index));
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
    ...(cap === undefined ? {} : { cap }),
    ...(limit === undefined ? {} : { limit }),
  };
}

/** Carrier wording cannot turn a known schedule construction into a new frame.
 * Negatives have no schedule slots; group them by normalized prose instead. */
export function frameKey(example: KindExample): string {
  const splitExample = example.splitExample ?? example.example;
  const hasScheduleLabels = splitExample.spans.some((span) => span.label !== "O");

  if (example.intent === null && !hasScheduleLabels) {
    return `negative:${coarseText(splitExample.text)}`;
  }

  const speechAct = splitExample.speechAct ?? (example.intent === null ? "none" : "request");
  const pattern = framePattern(splitExample.text, splitExample.spans, speechAct);

  if (example.intent === null) return JSON.stringify(pattern);

  return JSON.stringify(pattern.filter((part) => part !== "O"));
}

/** Assign entire frames, globally and deterministically, rather than rows.
 * Roughly 75% of frames train, 12.5% validation, 12.5% held-out. */
export function assignSplits(examples: readonly KindExample[]): KindExample[] {
  const plan = splitPlan(examples);

  return examples.map((example) => assignSplit(example, plan));
}

type SplitPlan = ReadonlyMap<KindId, ReadonlyMap<string, ExampleSplit>>;

function splitBucket(key: string): number {
  return createHash("sha256").update(`${SPLIT_VERSION}\0${key}`).digest().readUInt32BE(0) % 8;
}

/** Assign equal structural frames globally, then repair sparse kinds so train,
 * validation, and held-out all contain at least one distinct frame. A frame can
 * occur in more than one kind (for example an explicit day-of-month and a
 * named day rule can have the same labeled surface shape), so assigning it
 * independently per kind would leak that frame across splits. */
function splitPlan(examples: Iterable<KindExample>): SplitPlan {
  const keys = new Map<KindId, Set<string>>();

  for (const example of examples) {
    const set = keys.get(example.kind) ?? new Set<string>();

    set.add(frameKey(example));
    keys.set(example.kind, set);
  }

  const orderedByKind = new Map<KindId, string[]>();

  for (const [kind, values] of keys) {
    const ordered = [...values].sort((left, right) => {
      const leftHash = createHash("sha256").update(left).digest("hex");
      const rightHash = createHash("sha256").update(right).digest("hex");

      return leftHash.localeCompare(rightHash);
    });

    if (ordered.length < SPLITS.length) {
      throw new Error(`${kind}: at least ${SPLITS.length} distinct frames are required`);
    }

    orderedByKind.set(kind, ordered);
  }

  const assignments = new Map<string, ExampleSplit>();

  for (const ordered of orderedByKind.values()) {
    for (const key of ordered) {
      if (assignments.has(key)) continue;
      const bucket = splitBucket(key);

      assignments.set(key, bucket === 6 ? "validation" : bucket === 7 ? "heldout" : "train");
    }
  }

  const kindsForKey = new Map<string, KindId[]>();

  for (const [kind, ordered] of orderedByKind) {
    for (const key of ordered) {
      const owners = kindsForKey.get(key) ?? [];

      owners.push(kind);
      kindsForKey.set(key, owners);
    }
  }

  const count = (kind: KindId, split: ExampleSplit): number =>
    orderedByKind.get(kind)!.filter((key) => assignments.get(key) === split).length;

  // Moving a shared frame is safe only when none of its owner kinds loses its
  // last frame in the donor split. Prefer moves that fill the same missing
  // split for several owner kinds at once.
  for (let pass = 0; pass < orderedByKind.size * SPLITS.length; pass++) {
    let changed = false;

    for (const [kind, ordered] of orderedByKind) {
      for (const split of SPLITS) {
        if (count(kind, split) !== 0) continue;

        const candidates = ordered
          .filter((key) => {
            const donor = assignments.get(key)!;

            return (
              donor !== split && kindsForKey.get(key)!.every((owner) => count(owner, donor) > 1)
            );
          })
          .sort((left, right) => {
            const leftGain = kindsForKey
              .get(left)!
              .filter((owner) => count(owner, split) === 0).length;

            const rightGain = kindsForKey
              .get(right)!
              .filter((owner) => count(owner, split) === 0).length;

            return rightGain - leftGain || left.localeCompare(right);
          });

        const chosen = candidates[0];

        if (chosen === undefined) {
          throw new Error(`${kind}: cannot assign a disjoint ${split} frame`);
        }

        assignments.set(chosen, split);
        changed = true;
      }
    }

    const complete = [...orderedByKind].every(([kind]) =>
      SPLITS.every((split) => count(kind, split) > 0),
    );

    if (complete) break;
    if (!changed) throw new Error("cannot make all kind splits structurally disjoint");
  }

  const plan = new Map<KindId, Map<string, ExampleSplit>>();

  for (const [kind, ordered] of orderedByKind) {
    plan.set(kind, new Map(ordered.map((key) => [key, assignments.get(key)!])));
  }

  return plan;
}

export function assignSplit(example: KindExample, plan?: SplitPlan): KindExample {
  const planned = plan?.get(example.kind)?.get(frameKey(example));

  if (planned !== undefined) return { ...example, split: planned };
  const bucket = splitBucket(frameKey(example));
  const split: ExampleSplit = bucket === 6 ? "validation" : bucket === 7 ? "heldout" : "train";

  return { ...example, split };
}

function toRecord(example: KindExample, index: number): CorpusRecord {
  const split = example.split;

  const fingerprint =
    example.intent === null
      ? createHash("sha256")
          .update(`negative\0${coarseText(example.example.text)}`)
          .digest("hex")
      : fingerprintExample(example.example);

  return {
    id: `${split}-${String(index).padStart(6, "0")}`,
    split,
    seed: BACKWARD_SEED,
    family: example.family,
    template: example.template,
    fingerprint,
    text: example.example.text,
    intent: example.intent,
    speechAct: example.example.speechAct,
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
  const expanded = expandKinds(wanted);

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
  const metadata = splitMetadataSets();

  for (const record of records) addSplitMetadata(metadata, record);
  const { fingerprintsBySplit, carrierHashesBySplit } = sortedSplitMetadata(metadata);

  mkdirSync(dirname(options.out), { recursive: true });
  const body = records.map((record) => JSON.stringify(record)).join("\n");

  writeFileSync(options.out, body.length > 0 ? `${body}\n` : "");

  const fingerprints = [...new Set(records.map((record) => record.fingerprint))].sort();
  const fingerprintsPath = `${options.out}.fingerprints.json`;
  const sha = generatorSha256();

  writeFileSync(
    fingerprintsPath,
    `${JSON.stringify(
      { fingerprints, fingerprintsBySplit, carrierHashesBySplit, generatorSha256: sha },
      null,
      2,
    )}\n`,
  );

  return {
    records,
    fingerprints,
    fingerprintsBySplit,
    carrierHashesBySplit,
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
  const metadata = splitMetadataSets();
  const byKind: Record<string, number> = {};
  const bySplit: Record<Split, number> = { train: 0, validation: 0, heldout: 0 };
  let index = 0;
  let chunk = "";
  // Only compact frame keys are retained. The catalog itself remains streamed,
  // so millions of rendered rows are never held in memory at once.
  const plan = options.cap === undefined ? splitPlan(iterateKinds(wanted)) : undefined;
  const perKind = new Map<string, number>();
  const fingerprintSplit = new Map<string, Split>();

  try {
    for (const expanded of iterateKinds(wanted)) {
      if (options.cap !== undefined) {
        const taken = perKind.get(expanded.kind) ?? 0;

        if (taken >= options.cap) continue;
        perKind.set(expanded.kind, taken + 1);
      }

      let example = assignSplit(expanded, plan);

      if (options.cap !== undefined) {
        const taken = (perKind.get(expanded.kind) ?? 1) - 1;

        if (taken === 0) example = { ...example, split: "train" };
        else if (taken === 1) example = { ...example, split: "validation" };
        else if (taken === 2) example = { ...example, split: "heldout" };
      }

      if (options.split !== undefined && example.split !== options.split) continue;
      if (options.limit !== undefined && index >= options.limit) break;
      const record = toRecord(example, index);
      const prior = fingerprintSplit.get(record.fingerprint);

      if (prior !== undefined && prior !== record.split) continue;
      fingerprintSplit.set(record.fingerprint, record.split);
      index += 1;
      chunk += `${JSON.stringify(record)}\n`;
      fingerprints.add(record.fingerprint);
      addSplitMetadata(metadata, record);
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
  const { fingerprintsBySplit, carrierHashesBySplit } = sortedSplitMetadata(metadata);
  const fingerprintsPath = `${options.out}.fingerprints.json`;
  const sha = generatorSha256();

  writeFileSync(
    fingerprintsPath,
    `${JSON.stringify(
      {
        fingerprints: sortedFingerprints,
        fingerprintsBySplit,
        carrierHashesBySplit,
        generatorSha256: sha,
      },
      null,
      2,
    )}\n`,
  );

  return {
    fingerprints: sortedFingerprints,
    fingerprintsBySplit,
    carrierHashesBySplit,
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
