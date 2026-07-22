/**
 * Featurizer: generator JSONL in, binary training shards out.
 *
 * Spans are UTF-16 offsets from the renderer; the tokenizer emits UTF-16
 * tokens. A span that does not cover whole tokens is a generator bug.
 */
import { existsSync, readFileSync } from "node:fs";
import { dirname, isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";

import { SHARD_FORMAT, writeShards } from "../../../shared/training/binary.js";
import type { Sequence } from "../../../shared/training/binary.js";
import { readJsonl } from "../../../shared/training/jsonl.js";
import { LABEL_COUNT, labelId, LABELS } from "../../src/model/labels.js";
import {
  EMBEDDING_ROWS,
  FEATURE_COUNT,
  featureRows,
  PADDING_ROW,
  tokenize,
} from "../../src/model/tokenizer.js";
import type { CorpusRecord, Split } from "./generate.js";

export const MAX_TOKENS = 128;
export const PADDING_LABEL = LABEL_COUNT;

export { SHARD_FORMAT } from "../../../shared/training/binary.js";
export type ShardFormat = typeof SHARD_FORMAT;

export interface FeaturizeManifest {
  fingerprints: string[];
  roleClasses: number;
  rowsPerToken: number;
  featureRows: number;
  labelCounts: Record<string, number>;
  frameNames: string[];
  frameCounts: number[];
  format: ShardFormat;
}

export interface FeaturizeOptions {
  input: string;
  out: string;
  split?: Split;
}

export interface FeaturizeSummary {
  input: string;
  out: string;
  sequences: number;
  tokens: number;
  skippedEmpty: number;
  skippedLong: number;
  labelCounts: Record<string, number>;
  manifest: FeaturizeManifest;
  files: Record<keyof ShardFormat, string> & { manifest: string };
}

export interface PerRecord {
  rows: Uint16Array;
  labels: Uint8Array;
  boundaries: Uint8Array;
  kinds: Uint8Array;
  neighbors: Int16Array;
}

export function featurizeRecord(
  record: CorpusRecord,
  tokens: ReturnType<typeof tokenize>,
): PerRecord {
  if ((record as { trainingEligible?: boolean }).trainingEligible === false) {
    throw new Error("evaluation-only records cannot be featurized for training");
  }

  const rows = new Uint16Array(tokens.length * FEATURE_COUNT).fill(PADDING_ROW);
  const labels = new Uint8Array(tokens.length).fill(PADDING_LABEL);
  const boundaries = new Uint8Array(tokens.length);
  const kinds = new Uint8Array(tokens.length);
  const neighbors = new Int16Array(tokens.length * 2).fill(-1);

  let spanIndex = 0;
  let previousNonSpace = -1;

  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]!;

    kinds[index] = token.kind;
    rows.set(featureRows(token), index * FEATURE_COUNT);

    while (spanIndex < record.spans.length && record.spans[spanIndex]!.end <= token.start) {
      spanIndex++;
    }

    if (token.kind !== 3) {
      const span = record.spans[spanIndex];

      if (span === undefined || token.start < span.start || token.end > span.end) {
        const covering = span === undefined ? "no span" : `span ${span.start}-${span.end}`;

        throw new Error(
          `Unaligned supervision in ${record.id}: token ` +
            `${token.start}-${token.end} "${token.text}" is not covered by ${covering}`,
        );
      }

      const label = labelId[span.label];

      if (!Number.isInteger(label) || label < 0 || label >= LABEL_COUNT) {
        throw Error("unknown training role");
      }

      labels[index] = label;
      boundaries[index] = Number(span.clauseStart && token.start === span.start);
    }

    neighbors[index * 2] = previousNonSpace;
    if (token.kind !== 3) previousNonSpace = index;
  }

  let nextNonSpace = -1;

  for (let index = tokens.length - 1; index >= 0; index--) {
    neighbors[index * 2 + 1] = nextNonSpace;
    if (tokens[index]!.kind !== 3) nextNonSpace = index;
  }

  return {
    rows,
    labels,
    boundaries,
    kinds,
    neighbors,
  };
}

function shardPath(prefix: string, key: keyof ShardFormat): string {
  return `${prefix}${SHARD_FORMAT[key].suffix}`;
}

function asRecord(value: unknown, index: number): CorpusRecord {
  if (value === null || typeof value !== "object") {
    throw new TypeError(`record ${index + 1} is not an object`);
  }

  if ((value as { trainingEligible?: boolean }).trainingEligible === false) {
    throw new Error("evaluation-only records cannot be featurized for training");
  }

  const record = value as CorpusRecord;

  if (typeof record.id !== "string" || typeof record.text !== "string") {
    throw new TypeError(`record ${index + 1} is missing id/text`);
  }

  if (!Array.isArray(record.spans)) {
    throw new TypeError(`record ${index + 1} is missing spans`);
  }

  return record;
}

export function featurizeFiles(options: FeaturizeOptions): FeaturizeSummary {
  const input = resolvePath(options.input);
  const out = resolvePath(options.out);

  let skippedEmpty = 0;
  let skippedLong = 0;

  function* sequences(): Generator<Sequence> {
    const owners = new Map<string, Split>();
    let index = 0;

    for (const value of readJsonl(input)) {
      const record = asRecord(value, index++);
      const owner = owners.get(record.fingerprint);

      if (owner !== undefined && owner !== record.split) throw Error("structural split leak");
      owners.set(record.fingerprint, record.split);
      if (options.split !== undefined && record.split !== options.split) continue;
      const tokens = tokenize(record.text);

      if (!tokens.length) {
        skippedEmpty++;
        continue;
      }

      if (tokens.length > MAX_TOKENS) {
        skippedLong++;
        continue;
      }

      const features = featurizeRecord(record, tokens);

      yield {
        ...features,
        fingerprint: record.fingerprint,
        frame: record.template ?? `${record.family}/unspecified`,
      };
    }
  }

  const result = writeShards(out, sequences(), {
    labels: LABELS,
    paddingLabel: PADDING_LABEL,
    rowsPerToken: FEATURE_COUNT,
    featureRows: EMBEDDING_ROWS,
  });

  const files: FeaturizeSummary["files"] = {
    rows: shardPath(out, "rows"),
    labels: shardPath(out, "labels"),
    boundaries: shardPath(out, "boundaries"),
    kinds: shardPath(out, "kinds"),
    neighbors: shardPath(out, "neighbors"),
    offsets: shardPath(out, "offsets"),
    frames: shardPath(out, "frames"),
    manifest: `${out}.json`,
  };

  return {
    input,
    out,
    sequences: result.sequences,
    tokens: result.tokens,
    skippedEmpty,
    skippedLong,
    labelCounts: result.manifest.labelCounts,
    manifest: result.manifest,
    files,
  };
}

function repoRoot(): string {
  let dir = dirname(fileURLToPath(import.meta.url));

  for (;;) {
    const manifest = join(dir, "package.json");

    if (existsSync(manifest)) {
      try {
        const pkg = JSON.parse(readFileSync(manifest, "utf8")) as {
          workspaces?: unknown;
        };

        if (Array.isArray(pkg.workspaces)) return dir;
      } catch {
        // keep walking
      }
    }

    const parent = dirname(dir);

    if (parent === dir) return process.cwd();
    dir = parent;
  }
}

function resolvePath(value: string): string {
  return isAbsolute(value) ? value : join(repoRoot(), value);
}

export function parseArgs(argv: readonly string[]): FeaturizeOptions {
  let split: Split | undefined;
  const positional: string[] = [];

  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index]!;

    if (arg === "--split") {
      const value = argv[index + 1];

      if (value === undefined) throw new RangeError("missing value for --split");

      if (value !== "train" && value !== "validation" && value !== "heldout") {
        throw new RangeError(`unknown split: ${value}`);
      }

      split = value;
      index++;
      continue;
    }

    if (arg.startsWith("--")) throw new RangeError(`unknown flag: ${arg}`);
    positional.push(arg);
  }

  if (positional.length !== 2) {
    throw new RangeError(
      `usage: featurize [--split train|validation|heldout] <input.jsonl> <prefix>`,
    );
  }

  return {
    input: positional[0]!,
    out: positional[1]!,
    ...(split === undefined ? {} : { split }),
  };
}

if (import.meta.main) {
  const options = parseArgs(process.argv.slice(2));
  const summary = featurizeFiles(options);

  process.stdout.write(
    `${summary.sequences} sequences, ${summary.tokens} tokens ` +
      `(${summary.skippedEmpty} empty, ${summary.skippedLong} too long) -> ${summary.out}\n`,
  );
}
