/**
 * Featurizer: merged JSONL corpus in, binary training shards out.
 *
 * The corpus records recorded spans with UTF-16 offsets; the tokenizer turns
 * the same text into UTF-16-offset tokens. This module maps one onto the other
 * and screams if a span does not cover a whole number of tokens — the Phase 07
 * alignment contract. It never guesses a label and never falls back to a
 * parser: a misaligned fragment is a generator bug, not a runtime condition.
 *
 * Rule and LLM records take the identical path. The shard contract is owned
 * here and is authoritative for the Python reader: `uint16` rows (17 per
 * token), `uint8` labels/boundaries/kinds, `int16` neighbour pairs, and a
 * `uint32` sequence-offset table of length `N + 1`.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";

import { LABEL_COUNT, labelId, LABELS } from "../../src/model/labels.js";
import {
  EMBEDDING_ROWS,
  FEATURE_COUNT,
  featureRows,
  PADDING_ROW,
  tokenize,
} from "../../src/model/tokenizer.js";
import {
  assertDisjoint,
  corpusRecordSchema,
  heldoutFingerprintsOf,
  readJsonl,
} from "./corpus/merge.js";
import type { CorpusRecord } from "./corpus/merge.js";

/** Longest sequence the model will see; longer examples are skipped. */
export const MAX_TOKENS = 128;

/**
 * Whitespace has no role, so it carries an out-of-range sentinel rather than a
 * real label id. Every real id is in `[0, LABEL_COUNT)`; Python masks the rest.
 */
export const PADDING_LABEL = LABEL_COUNT;

/** The six shard files, each named by suffix and storage dtype. */
export const SHARD_FORMAT = {
  rows: { suffix: ".rows.bin", dtype: "uint16" },
  labels: { suffix: ".labels.bin", dtype: "uint8" },
  boundaries: { suffix: ".boundaries.bin", dtype: "uint8" },
  kinds: { suffix: ".kinds.bin", dtype: "uint8" },
  neighbors: { suffix: ".neighbors.bin", dtype: "int16" },
  offsets: { suffix: ".offsets.bin", dtype: "uint32" },
} as const;

export type ShardFormat = typeof SHARD_FORMAT;

/** The manifest layout contract. Six fields, `labelCounts` the only aggregate. */
export interface FeaturizeManifest {
  fingerprints: string[];
  roleClasses: number;
  rowsPerToken: number;
  featureRows: number;
  labelCounts: Record<string, number>;
  format: ShardFormat;
}

export interface FeaturizeOptions {
  input: string;
  out: string;
}

/** The in-memory shard, before it is flattened to disk. */
export interface FeaturizedShard {
  rows: Uint16Array;
  labels: Uint8Array;
  boundaries: Uint8Array;
  kinds: Uint8Array;
  neighbors: Int16Array;
  offsets: Uint32Array;
  sequences: number;
  tokens: number;
  skippedEmpty: number;
  skippedLong: number;
  labelCounts: Record<string, number>;
  fingerprints: string[];
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
  tokenCount: number;
  labelCounts: Record<string, number>;
}

/**
 * Featurize a record's already-tokenized text. The tokenizer and the recorded
 * spans share UTF-16 offsets, so alignment is a comparison of half-open
 * intervals.
 */
export function featurizeRecord(
  record: CorpusRecord,
  tokens: ReturnType<typeof tokenize>,
): PerRecord {
  const rows = new Uint16Array(tokens.length * FEATURE_COUNT).fill(PADDING_ROW);
  const labels = new Uint8Array(tokens.length).fill(PADDING_LABEL);
  const boundaries = new Uint8Array(tokens.length);
  const kinds = new Uint8Array(tokens.length);
  const neighbors = new Int16Array(tokens.length * 2).fill(-1);
  const labelCounts: Record<string, number> = {};

  for (const label of LABELS) labelCounts[label] = 0;

  let spanIndex = 0;
  let previousNonSpace = -1;

  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]!;

    kinds[index] = token.kind;
    rows.set(featureRows(token.features), index * FEATURE_COUNT);

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

      labels[index] = labelId[span.label];
      labelCounts[span.label] = (labelCounts[span.label] ?? 0) + 1;
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
    tokenCount: tokens.length,
    labelCounts,
  };
}

function concatUint16(chunks: readonly Uint16Array[]): Uint16Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Uint16Array(total);
  let offset = 0;

  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  return merged;
}

function concatUint8(chunks: readonly Uint8Array[]): Uint8Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Uint8Array(total);
  let offset = 0;

  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  return merged;
}

function concatInt16(chunks: readonly Int16Array[]): Int16Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Int16Array(total);
  let offset = 0;

  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  return merged;
}

/**
 * Featurize a batch of records into one flat shard. The leak gate is re-run
 * here; a failed re-check aborts rather than emitting a corpus that would
 * silently inflate validation scores.
 */
export function featurizeRecords(records: readonly CorpusRecord[]): FeaturizedShard {
  const heldout = new Set(heldoutFingerprintsOf(records));

  assertDisjoint(records, heldout);

  const rowChunks: Uint16Array[] = [];
  const labelChunks: Uint8Array[] = [];
  const boundaryChunks: Uint8Array[] = [];
  const kindChunks: Uint8Array[] = [];
  const neighborChunks: Int16Array[] = [];
  const offsets: number[] = [0];
  const labelCounts: Record<string, number> = {};

  for (const label of LABELS) labelCounts[label] = 0;

  let sequences = 0;
  let tokens = 0;
  let skippedEmpty = 0;
  let skippedLong = 0;

  for (const record of records) {
    const recordTokens = tokenize(record.text);

    if (recordTokens.length === 0) {
      skippedEmpty++;
      continue;
    }

    if (recordTokens.length > MAX_TOKENS) {
      skippedLong++;
      continue;
    }

    const shard = featurizeRecord(record, recordTokens);

    rowChunks.push(shard.rows);
    labelChunks.push(shard.labels);
    boundaryChunks.push(shard.boundaries);
    kindChunks.push(shard.kinds);
    neighborChunks.push(shard.neighbors);
    sequences++;
    tokens += shard.tokenCount;
    offsets.push(offsets.at(-1)! + shard.tokenCount);

    for (const label of LABELS) {
      labelCounts[label]! += shard.labelCounts[label]!;
    }
  }

  return {
    rows: concatUint16(rowChunks),
    labels: concatUint8(labelChunks),
    boundaries: concatUint8(boundaryChunks),
    kinds: concatUint8(kindChunks),
    neighbors: concatInt16(neighborChunks),
    offsets: Uint32Array.from(offsets),
    sequences,
    tokens,
    skippedEmpty,
    skippedLong,
    labelCounts,
    fingerprints: [...new Set(records.map((record) => record.fingerprint))].sort(),
  };
}

function shardPath(prefix: string, key: keyof ShardFormat): string {
  return `${prefix}${SHARD_FORMAT[key].suffix}`;
}

function writeBinary(path: string, view: ArrayBufferView): void {
  writeFileSync(path, Buffer.from(view.buffer, view.byteOffset, view.byteLength));
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`);
}

/** Read a corpus JSONL, featurize it, and write the shards and manifest. */
export function featurizeFiles(options: FeaturizeOptions): FeaturizeSummary {
  const input = resolvePath(options.input);
  const out = resolvePath(options.out);

  const records = readJsonl(input).map((value, index) => {
    const parsed = corpusRecordSchema.safeParse(value);

    if (!parsed.success) {
      throw new Error(
        `${input} record ${index + 1}: invalid corpus record: ${parsed.error.issues
          .map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`)
          .join("; ")}`,
      );
    }

    return parsed.data as CorpusRecord;
  });

  const shard = featurizeRecords(records);

  mkdirSync(dirname(out), { recursive: true });

  const files: FeaturizeSummary["files"] = {
    rows: shardPath(out, "rows"),
    labels: shardPath(out, "labels"),
    boundaries: shardPath(out, "boundaries"),
    kinds: shardPath(out, "kinds"),
    neighbors: shardPath(out, "neighbors"),
    offsets: shardPath(out, "offsets"),
    manifest: `${out}.json`,
  };

  writeBinary(files.rows, shard.rows);
  writeBinary(files.labels, shard.labels);
  writeBinary(files.boundaries, shard.boundaries);
  writeBinary(files.kinds, shard.kinds);
  writeBinary(files.neighbors, shard.neighbors);
  writeBinary(files.offsets, shard.offsets);

  const manifest: FeaturizeManifest = {
    fingerprints: shard.fingerprints,
    roleClasses: LABEL_COUNT,
    rowsPerToken: FEATURE_COUNT,
    featureRows: EMBEDDING_ROWS,
    labelCounts: shard.labelCounts,
    format: SHARD_FORMAT,
  };

  writeJson(files.manifest, manifest);

  return {
    input,
    out,
    sequences: shard.sequences,
    tokens: shard.tokens,
    skippedEmpty: shard.skippedEmpty,
    skippedLong: shard.skippedLong,
    labelCounts: shard.labelCounts,
    manifest,
    files,
  };
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/** Walk up to the workspace root (the package.json that lists `workspaces`). */
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
        // Not a readable manifest; keep walking up.
      }
    }

    const parent = dirname(dir);

    if (parent === dir) return process.cwd();
    dir = parent;
  }
}

/** Resolve a CLI path against the repo root so `--filter` cwd does not matter. */
function resolvePath(value: string): string {
  return isAbsolute(value) ? value : join(repoRoot(), value);
}

/** Parse the two positional arguments: `<input.jsonl> <prefix>`. */
export function parseArgs(argv: readonly string[]): FeaturizeOptions {
  const positional = argv.filter((arg) => !arg.startsWith("--"));

  if (positional.length !== 2) {
    throw new RangeError(
      `usage: featurize <input.jsonl> <prefix> (got ${positional.length} arguments)`,
    );
  }

  return { input: positional[0]!, out: positional[1]! };
}

if (import.meta.main) {
  const options = parseArgs(process.argv.slice(2));
  const summary = featurizeFiles(options);

  process.stdout.write(
    `${summary.sequences} sequences, ${summary.tokens} tokens ` +
      `(${summary.skippedEmpty} empty, ${summary.skippedLong} too long) -> ${summary.out}\n`,
  );
}
