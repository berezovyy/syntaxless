/**
 * Bounded-memory, payload-free planning for the complete corpus union.
 *
 * Pass one writes only fixed-width hashes and source ordinals to partition
 * files. Reduction keeps one deterministic source reference for each
 * runtime-normalized sentence. A single deterministic replay then emits the
 * selected records themselves. Record JSON is never copied into scratch.
 *
 * Memory does not grow with the candidate count (see the training README's
 * memory rule):
 *  - ingest holds one fixed write buffer per partition;
 *  - reduction loads at most `reduceWindowBytes` of one partition file at a
 *    time. A larger file is split on further hash bits first, recursively, so
 *    a partition never has to fit in memory however large the corpus grows;
 *  - every reduced window writes its winners as a sorted run of ordinals, and
 *    runs are merged on disk with a bounded fan-in into one ascending file;
 *  - replay streams that file with a fixed buffer instead of loading a bitmap.
 */
import { createHash, type Hash } from "node:crypto";
import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readdirSync,
  readFileSync,
  readSync,
  rmSync,
  statSync,
  unlinkSync,
  writeFileSync,
  writeSync,
} from "node:fs";
import { join } from "node:path";

import { canonicalizeIntent } from "../../canonical-intent";
import type { CorpusRecord } from "../merge";
import { corpusTextKey } from "./protected-text";

/** v2: winners are an ascending ordinal file, not a one-bit-per-candidate bitmap. */
const FORMAT_VERSION = 2;
const HASH_BYTES = 32;
const ORDINAL_BYTES = 8;
const SIGNATURE_OFFSET = HASH_BYTES;
const REFERENCE_OFFSET = HASH_BYTES * 2;
const ORDINAL_OFFSET = HASH_BYTES * 3;
const PRIORITY_OFFSET = ORDINAL_OFFSET + ORDINAL_BYTES;
const CANDIDATE_BYTES = PRIORITY_OFFSET + 4 + 1;
const SPLIT_OFFSET = CANDIDATE_BYTES - 1;
/** Ingest buffer per partition writer; at most `partitions` writers exist. */
const WRITE_BUFFER_BYTES = 256 * 1024;
/** Sequential read buffer for every streamed scratch file. */
const READ_BUFFER_BYTES = 64 * 1024;
/** Default largest partition file reduced in memory (about 300k candidates). */
const DEFAULT_REDUCE_WINDOW_BYTES = 32 * 1024 ** 2;
/** Default number of sorted ordinal runs merged at once. */
const DEFAULT_MERGE_FAN_IN = 64;
/** An oversized file splits 16 ways on the next unused 4-bit slice of the hash. */
const SPLIT_FANOUT = 16;
const SPLIT_LEVELS = (HASH_BYTES * 8) / 4;

/** Files that make up a persisted plan; callers move exactly these. */
export const CORPUS_UNION_PLAN_FILES = ["manifest.json", "selected-ordinals.bin"] as const;
const SELECTED_FILE = CORPUS_UNION_PLAN_FILES[1];

type Split = CorpusRecord["split"];

const SPLIT_CODE: Readonly<Record<Split, number>> = {
  train: 0,
  validation: 1,
  heldout: 2,
};

export interface CorpusUnionOptions {
  /** Empty directory owned by this operation. */
  directory: string;
  /** Ingest fan-out. Reduction memory is bounded by `reduceWindowBytes` instead. */
  partitions?: number;
  /**
   * Largest scratch file reduced in memory. Production keeps the default;
   * tests lower it to force recursive splitting on tiny inputs.
   */
  reduceWindowBytes?: number;
  /** Sorted runs merged per pass. Tests lower it to force multi-pass merges. */
  mergeFanIn?: number;
}

export interface CorpusUnionCandidateOptions {
  /** Lower values win. Defaults to an authorship-aware source priority. */
  priority?: number;
  /** Stable source reference. Defaults to provenance plus the record id. */
  reference?: string;
}

export interface CorpusUnionSummary {
  version: 2;
  candidates: number;
  uniqueTexts: number;
  duplicates: number;
  partitions: number;
  ingestSha256: string;
  /** SHA-256 of the ascending big-endian uint64 winner ordinals. */
  selectedReferencesSha256: string;
}

export type CorpusUnionDecision =
  | { selected: false }
  | {
      selected: true;
      /** The stable source reference chosen during reduction. */
      reference: string;
      /** The chosen record's original split; the union never resplits data. */
      split: Split;
      /** SHA-256 of the runtime-normalized text. */
      textHash: string;
    };

interface Manifest extends CorpusUnionSummary {
  candidateBytes: number;
  selectionBytes: number;
}

function sha256(value: string | Buffer): Buffer {
  return createHash("sha256").update(value).digest();
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const object = value as Record<string, unknown>;

  return `{${Object.keys(object)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(object[key])}`)
    .join(",")}}`;
}

function supervisionHash(record: CorpusRecord): Buffer {
  return sha256(
    canonicalJson({
      intent: record.intent === null ? null : canonicalizeIntent(record.intent),
      speechAct: record.speechAct ?? null,
      spans: record.spans.map((span) => [
        span.start,
        span.end,
        span.label,
        span.clauseStart,
        span.actionable,
      ]),
    }),
  );
}

/** Human data wins an exact duplicate by default; callers can override this. */
export function defaultCorpusPriority(record: CorpusRecord): number {
  switch (record.provenance?.authorship) {
    case "human-authored":
      return 0;
    case "programmatic":
      return 100;
    case "public-source":
      return 200;
    case "ai-authored":
      return 300;
    default:
      return record.source === "rule" ? 100 : 300;
  }
}

/** Stable identity used to refer back to a source record during replay. */
export function corpusSourceReference(record: CorpusRecord): string {
  const provenance = record.provenance;

  if (!provenance) return `${record.source}:${record.id}`;

  return [
    provenance.sourceId,
    provenance.sourceRecordId ?? "",
    provenance.groupId,
    provenance.constructionId,
    record.id,
  ]
    .map(encodeURIComponent)
    .join("/");
}

/** Big-endian uint64 without BigInt; ordinals stay below 2^53. */
function writeOrdinal(buffer: Buffer, offset: number, value: number): void {
  buffer.writeUInt32BE(Math.floor(value / 2 ** 32), offset);
  buffer.writeUInt32BE(value >>> 0, offset + 4);
}

function readOrdinal(buffer: Buffer, offset: number): number {
  return buffer.readUInt32BE(offset) * 2 ** 32 + buffer.readUInt32BE(offset + 4);
}

function candidateBytes(
  record: CorpusRecord,
  ordinal: number,
  options: CorpusUnionCandidateOptions,
): { bytes: Buffer; textHash: Buffer; reference: string } {
  const priority = options.priority ?? defaultCorpusPriority(record);

  if (!Number.isInteger(priority) || priority < -0x8000_0000 || priority > 0x7fff_ffff) {
    throw new Error(`Corpus union priority must be a signed 32-bit integer: ${priority}`);
  }

  const reference = options.reference ?? corpusSourceReference(record);

  if (!reference) throw new Error("Corpus union source reference must not be empty");
  const textHash = sha256(corpusTextKey(record.text));
  const bytes = Buffer.allocUnsafe(CANDIDATE_BYTES);

  textHash.copy(bytes, 0);
  supervisionHash(record).copy(bytes, SIGNATURE_OFFSET);
  sha256(reference).copy(bytes, REFERENCE_OFFSET);
  writeOrdinal(bytes, ORDINAL_OFFSET, ordinal);
  bytes.writeInt32BE(priority, PRIORITY_OFFSET);
  bytes.writeUInt8(SPLIT_CODE[record.split], SPLIT_OFFSET);

  return { bytes, textHash, reference };
}

class BufferedBinaryWriter {
  readonly #fd: number;
  readonly #buffer: Buffer;
  #used = 0;
  #closed = false;

  constructor(path: string, bufferBytes = WRITE_BUFFER_BYTES) {
    this.#fd = openSync(path, "wx");
    this.#buffer = Buffer.allocUnsafe(bufferBytes);
  }

  write(bytes: Buffer): void {
    if (this.#closed) throw new Error("Corpus union writer is closed");

    if (bytes.length > this.#buffer.length) {
      this.flush();
      writeAll(this.#fd, bytes);

      return;
    }

    if (this.#used + bytes.length > this.#buffer.length) this.flush();
    bytes.copy(this.#buffer, this.#used);
    this.#used += bytes.length;
  }

  flush(): void {
    if (this.#used > 0) {
      writeAll(this.#fd, this.#buffer.subarray(0, this.#used));
      this.#used = 0;
    }
  }

  close(): void {
    if (this.#closed) return;
    this.#closed = true;

    try {
      this.flush();
    } finally {
      closeSync(this.#fd);
    }
  }
}

function writeAll(fd: number, bytes: Buffer): void {
  let written = 0;

  while (written < bytes.length) written += writeSync(fd, bytes, written, bytes.length - written);
}

/**
 * Sequential reader over a file of fixed-width records with one fixed buffer.
 * `next()` returns a view that is valid only until the following call.
 */
class FixedRecordReader {
  readonly #path: string;
  readonly #fd: number;
  readonly #recordBytes: number;
  readonly #buffer: Buffer;
  readonly #size: number;
  #position = 0;
  #offset = 0;
  #end = 0;
  #closed = false;

  constructor(path: string, recordBytes: number) {
    this.#path = path;
    this.#size = statSync(path).size;

    if (this.#size % recordBytes !== 0) {
      throw new Error(`Truncated corpus union scratch file: ${path}`);
    }

    this.#recordBytes = recordBytes;

    this.#buffer = Buffer.allocUnsafe(
      Math.max(1, Math.floor(READ_BUFFER_BYTES / recordBytes)) * recordBytes,
    );

    this.#fd = openSync(path, "r");
  }

  next(): Buffer | undefined {
    if (this.#closed) throw new Error(`Corpus union reader is closed: ${this.#path}`);

    if (this.#offset === this.#end) {
      if (this.#position >= this.#size) return undefined;
      const wanted = Math.min(this.#buffer.length, this.#size - this.#position);
      let read = 0;

      while (read < wanted) {
        const bytes = readSync(this.#fd, this.#buffer, read, wanted - read, this.#position + read);

        if (bytes === 0) throw new Error(`Short read from corpus union scratch: ${this.#path}`);
        read += bytes;
      }

      this.#position += read;
      this.#offset = 0;
      this.#end = read;
    }

    const record = this.#buffer.subarray(this.#offset, this.#offset + this.#recordBytes);

    this.#offset += this.#recordBytes;

    return record;
  }

  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    closeSync(this.#fd);
  }
}

function hashFile(path: string): string {
  const hash = createHash("sha256");
  const buffer = Buffer.allocUnsafe(READ_BUFFER_BYTES);
  const fd = openSync(path, "r");

  try {
    for (let position = 0; ;) {
      const read = readSync(fd, buffer, 0, buffer.length, position);

      if (read === 0) break;
      hash.update(buffer.subarray(0, read));
      position += read;
    }
  } finally {
    closeSync(fd);
  }

  return hash.digest("hex");
}

/** Order candidates by text, then winner precedence, without allocating. */
function compareCandidates(data: Buffer, left: number, right: number): number {
  const byText = data.compare(data, right, right + HASH_BYTES, left, left + HASH_BYTES);

  if (byText !== 0) return byText;

  const byPriority =
    data.readInt32BE(left + PRIORITY_OFFSET) - data.readInt32BE(right + PRIORITY_OFFSET);

  if (byPriority !== 0) return byPriority;

  const byReference = data.compare(
    data,
    right + REFERENCE_OFFSET,
    right + REFERENCE_OFFSET + HASH_BYTES,
    left + REFERENCE_OFFSET,
    left + REFERENCE_OFFSET + HASH_BYTES,
  );

  if (byReference !== 0) return byReference;

  return readOrdinal(data, left + ORDINAL_OFFSET) - readOrdinal(data, right + ORDINAL_OFFSET);
}

function sameRange(
  left: Buffer,
  leftStart: number,
  right: Buffer,
  rightStart: number,
  bytes: number,
): boolean {
  return left.compare(right, rightStart, rightStart + bytes, leftStart, leftStart + bytes) === 0;
}

function conflict(textHash: Buffer, first: number, second: number): Error {
  return new Error(
    `Corpus union conflicting supervision for normalized-text ${textHash.toString("hex")}: ` +
      `ordinals ${first} and ${second}`,
  );
}

/** Which of 16 children a candidate falls into at a given split level. */
function splitChild(record: Buffer, level: number): number {
  // Partitioning already used the first four hash bytes; start after them
  // and wrap so every bit of the hash can eventually separate two texts.
  const byte = record[(4 + (level >> 1)) % HASH_BYTES]!;

  return level % 2 === 0 ? byte >> 4 : byte & 0x0f;
}

/**
 * Merge ascending ordinal runs into one ascending file with `inputs.length`
 * fixed read buffers. Ordinals are unique across runs, so equality is a bug.
 */
function mergeOrdinalRuns(inputs: readonly string[], output: string): number {
  const readers: FixedRecordReader[] = [];
  const writer = new BufferedBinaryWriter(output);

  try {
    for (const input of inputs) readers.push(new FixedRecordReader(input, ORDINAL_BYTES));
    const heads = new Float64Array(readers.length);
    const heap: number[] = [];
    const less = (a: number, b: number) => heads[a]! < heads[b]!;

    const siftDown = (start: number) => {
      let index = start;

      for (;;) {
        const left = index * 2 + 1;

        if (left >= heap.length) return;
        const right = left + 1;
        const child = right < heap.length && less(heap[right]!, heap[left]!) ? right : left;

        if (!less(heap[child]!, heap[index]!)) return;
        [heap[index], heap[child]] = [heap[child]!, heap[index]!];
        index = child;
      }
    };

    readers.forEach((reader, index) => {
      const record = reader.next();

      if (record === undefined) return;
      heads[index] = readOrdinal(record, 0);
      heap.push(index);
    });

    for (let index = (heap.length >> 1) - 1; index >= 0; index--) siftDown(index);
    const encoded = Buffer.allocUnsafe(ORDINAL_BYTES);
    let previous = -1;
    let count = 0;

    while (heap.length > 0) {
      const source = heap[0]!;
      const value = heads[source]!;

      if (value <= previous) {
        throw new Error(`Corpus union winner runs repeat or reorder ordinal ${value}`);
      }

      previous = value;
      writeOrdinal(encoded, 0, value);
      writer.write(encoded);
      count++;
      const record = readers[source]!.next();

      if (record === undefined) {
        heap[0] = heap[heap.length - 1]!;
        heap.pop();
      } else {
        heads[source] = readOrdinal(record, 0);
      }

      siftDown(0);
    }

    return count;
  } finally {
    for (const reader of readers) reader.close();
    writer.close();
  }
}

/** First-pass collector and reducer. */
export class StreamingCorpusUnion {
  readonly #directory: string;
  readonly #partitions: number;
  readonly #reduceWindowBytes: number;
  readonly #mergeFanIn: number;
  readonly #writers: (BufferedBinaryWriter | undefined)[];
  readonly #digest: Hash = createHash("sha256");
  #candidates = 0;
  #runs = 0;
  #phase: "collect" | "finalized" | "aborted" = "collect";

  constructor(options: CorpusUnionOptions) {
    const partitions = options.partitions ?? 64;

    if (!Number.isInteger(partitions) || partitions < 1 || partitions > 256) {
      throw new Error("Corpus union partitions must be an integer from 1 to 256");
    }

    const reduceWindowBytes = options.reduceWindowBytes ?? DEFAULT_REDUCE_WINDOW_BYTES;

    if (!Number.isSafeInteger(reduceWindowBytes) || reduceWindowBytes < CANDIDATE_BYTES) {
      throw new Error(
        `Corpus union reduce window must hold at least one ${CANDIDATE_BYTES}-byte candidate`,
      );
    }

    const mergeFanIn = options.mergeFanIn ?? DEFAULT_MERGE_FAN_IN;

    if (!Number.isInteger(mergeFanIn) || mergeFanIn < 2) {
      throw new Error("Corpus union merge fan-in must be an integer of at least 2");
    }

    if (existsSync(options.directory)) {
      if (readdirSync(options.directory).length) {
        throw new Error(`Corpus union directory is not empty: ${options.directory}`);
      }
    } else {
      mkdirSync(options.directory, { recursive: true });
    }

    this.#directory = options.directory;
    this.#partitions = partitions;
    this.#reduceWindowBytes = reduceWindowBytes;
    this.#mergeFanIn = mergeFanIn;
    this.#writers = Array.from({ length: partitions });
  }

  add(record: CorpusRecord, options: CorpusUnionCandidateOptions = {}): void {
    if (this.#phase !== "collect") throw new Error("Corpus union collection is closed");
    const candidate = candidateBytes(record, this.#candidates, options);
    const partition = candidate.textHash.readUInt32BE(0) % this.#partitions;

    // Lazy writers avoid one descriptor and one buffer per empty partition.
    const writer = (this.#writers[partition] ??= new BufferedBinaryWriter(
      join(this.#directory, `candidates-${String(partition).padStart(4, "0")}.bin`),
    ));

    writer.write(candidate.bytes);
    this.#digest.update(candidate.bytes);
    this.#candidates++;
  }

  finalize(): CorpusUnionPlan {
    if (this.#phase !== "collect") throw new Error("Corpus union can only finalize once");
    for (const writer of this.#writers) writer?.close();
    const ingestSha256 = this.#digest.digest("hex");

    try {
      // Winner runs are file paths, about one per reduce window: O(scratch
      // bytes / window), not O(candidates).
      const runs: string[] = [];
      let uniqueTexts = 0;

      for (let partition = 0; partition < this.#partitions; partition++) {
        const source = join(
          this.#directory,
          `candidates-${String(partition).padStart(4, "0")}.bin`,
        );

        if (existsSync(source)) uniqueTexts += this.#reduceFile(source, 0, runs);
      }

      const selectedPath = join(this.#directory, SELECTED_FILE);
      const selected = this.#mergeRuns(runs, selectedPath);

      if (selected !== uniqueTexts) {
        throw new Error("Corpus union winner merge lost or duplicated ordinals");
      }

      const manifest: Manifest = {
        version: FORMAT_VERSION,
        candidates: this.#candidates,
        uniqueTexts,
        duplicates: this.#candidates - uniqueTexts,
        partitions: this.#partitions,
        ingestSha256,
        selectedReferencesSha256: hashFile(selectedPath),
        candidateBytes: CANDIDATE_BYTES,
        selectionBytes: uniqueTexts * ORDINAL_BYTES,
      };

      writeFileSync(
        join(this.#directory, "manifest.json"),
        `${JSON.stringify(manifest, null, 2)}\n`,
      );

      this.#phase = "finalized";

      return new CorpusUnionPlan(this.#directory, manifest);
    } catch (error) {
      this.#phase = "aborted";
      throw error;
    }
  }

  /** Reduce one scratch file, splitting it first if it exceeds the window. */
  #reduceFile(path: string, level: number, runs: string[]): number {
    const size = statSync(path).size;

    if (size % CANDIDATE_BYTES !== 0) {
      throw new Error(`Truncated corpus union scratch file: ${path}`);
    }

    if (size === 0) {
      unlinkSync(path);

      return 0;
    }

    if (size <= this.#reduceWindowBytes) return this.#reduceWindow(path, runs);
    if (this.#singleText(path)) return this.#reduceSingleText(path, runs);

    if (level >= SPLIT_LEVELS) {
      throw new Error(`Corpus union cannot split scratch file further: ${path}`);
    }

    const children = Array.from(
      { length: SPLIT_FANOUT },
      (_, child) => `${path}.${child.toString(16)}`,
    );

    const writers: (BufferedBinaryWriter | undefined)[] = Array.from({
      length: SPLIT_FANOUT,
    });

    const reader = new FixedRecordReader(path, CANDIDATE_BYTES);

    try {
      for (let record = reader.next(); record !== undefined; record = reader.next()) {
        const child = splitChild(record, level);

        (writers[child] ??= new BufferedBinaryWriter(children[child]!)).write(record);
      }
    } finally {
      reader.close();
      for (const writer of writers) writer?.close();
    }

    unlinkSync(path);
    let unique = 0;

    for (const child of children) {
      if (existsSync(child)) unique += this.#reduceFile(child, level + 1, runs);
    }

    return unique;
  }

  /** In-memory reduction of a file no larger than the reduce window. */
  #reduceWindow(path: string, runs: string[]): number {
    const data = readFileSync(path);
    const count = data.length / CANDIDATE_BYTES;
    const order = new Uint32Array(count);

    for (let index = 0; index < count; index++) order[index] = index;
    order.sort((a, b) => compareCandidates(data, a * CANDIDATE_BYTES, b * CANDIDATE_BYTES));
    const winners = new Float64Array(count);
    let unique = 0;

    for (let index = 0; index < count;) {
      const first = order[index]! * CANDIDATE_BYTES;
      let next = index + 1;

      for (; next < count; next++) {
        const other = order[next]! * CANDIDATE_BYTES;

        if (!sameRange(data, first, data, other, HASH_BYTES)) break;
        const signature = first + SIGNATURE_OFFSET;

        if (!sameRange(data, signature, data, other + SIGNATURE_OFFSET, HASH_BYTES)) {
          throw conflict(
            data.subarray(first, first + HASH_BYTES),
            readOrdinal(data, first + ORDINAL_OFFSET),
            readOrdinal(data, other + ORDINAL_OFFSET),
          );
        }
      }

      // The sort puts the winning candidate first within its text.
      winners[unique++] = readOrdinal(data, first + ORDINAL_OFFSET);
      index = next;
    }

    this.#writeRun(winners.subarray(0, unique).sort(), runs);
    unlinkSync(path);

    return unique;
  }

  /** Whether every candidate in a file has the same text (cannot be split). */
  #singleText(path: string): boolean {
    const reader = new FixedRecordReader(path, CANDIDATE_BYTES);

    try {
      const first = Buffer.from(reader.next()!.subarray(0, HASH_BYTES));

      for (let record = reader.next(); record !== undefined; record = reader.next()) {
        if (!sameRange(record, 0, first, 0, HASH_BYTES)) return false;
      }

      return true;
    } finally {
      reader.close();
    }
  }

  /** Streaming reduction for one text repeated beyond the reduce window. */
  #reduceSingleText(path: string, runs: string[]): number {
    const reader = new FixedRecordReader(path, CANDIDATE_BYTES);
    // Two candidates side by side so compareCandidates can order them.
    const pair = Buffer.alloc(CANDIDATE_BYTES * 2);

    try {
      reader.next()!.copy(pair, 0);

      for (let record = reader.next(); record !== undefined; record = reader.next()) {
        record.copy(pair, CANDIDATE_BYTES);
        const other = CANDIDATE_BYTES + SIGNATURE_OFFSET;

        if (!sameRange(pair, SIGNATURE_OFFSET, pair, other, HASH_BYTES)) {
          throw conflict(
            pair.subarray(0, HASH_BYTES),
            readOrdinal(pair, ORDINAL_OFFSET),
            readOrdinal(pair, CANDIDATE_BYTES + ORDINAL_OFFSET),
          );
        }

        if (compareCandidates(pair, CANDIDATE_BYTES, 0) < 0) {
          pair.copy(pair, 0, CANDIDATE_BYTES, CANDIDATE_BYTES * 2);
        }
      }
    } finally {
      reader.close();
    }

    this.#writeRun(Float64Array.of(readOrdinal(pair, ORDINAL_OFFSET)), runs);
    unlinkSync(path);

    return 1;
  }

  #writeRun(ordinals: Float64Array, runs: string[]): void {
    if (ordinals.length === 0) return;
    const bytes = Buffer.allocUnsafe(ordinals.length * ORDINAL_BYTES);

    for (let index = 0; index < ordinals.length; index++) {
      writeOrdinal(bytes, index * ORDINAL_BYTES, ordinals[index]!);
    }

    const name = `winners-${String(this.#runs++).padStart(6, "0")}.bin`;
    const path = join(this.#directory, name);

    writeFileSync(path, bytes, { flag: "wx" });
    runs.push(path);
  }

  /** Multi-pass merge with at most `mergeFanIn` open runs at a time. */
  #mergeRuns(runs: readonly string[], output: string): number {
    let level = runs;

    for (let pass = 0; level.length > this.#mergeFanIn; pass++) {
      const next: string[] = [];

      for (let start = 0; start < level.length; start += this.#mergeFanIn) {
        const group = level.slice(start, start + this.#mergeFanIn);
        const merged = join(this.#directory, `merge-${pass}-${next.length}.bin`);

        mergeOrdinalRuns(group, merged);
        for (const path of group) unlinkSync(path);
        next.push(merged);
      }

      level = next;
    }

    const count = mergeOrdinalRuns(level, output);

    for (const path of level) unlinkSync(path);

    return count;
  }

  /** Remove only this operation's scratch directory. */
  abort(): void {
    if (this.#phase === "collect") for (const writer of this.#writers) writer?.close();
    rmSync(this.#directory, { recursive: true, force: true });
    this.#phase = "aborted";
  }
}

/** Persisted winner plan. It can be reopened in another process. */
export class CorpusUnionPlan {
  readonly directory: string;
  readonly summary: CorpusUnionSummary;
  readonly #manifest: Manifest;

  constructor(directory: string, manifest?: Manifest) {
    this.directory = directory;

    this.#manifest =
      manifest ?? (JSON.parse(readFileSync(join(directory, "manifest.json"), "utf8")) as Manifest);

    if (
      this.#manifest.version !== FORMAT_VERSION ||
      this.#manifest.candidateBytes !== CANDIDATE_BYTES ||
      this.#manifest.selectionBytes !== this.#manifest.uniqueTexts * ORDINAL_BYTES
    ) {
      throw new Error("Unsupported corpus union plan format");
    }

    const selected = join(directory, SELECTED_FILE);

    if (!existsSync(selected) || statSync(selected).size !== this.#manifest.selectionBytes) {
      throw new Error("Corpus union winner plan has the wrong size");
    }

    if (hashFile(selected) !== this.#manifest.selectedReferencesSha256) {
      throw new Error("Corpus union winner plan checksum mismatch");
    }

    this.summary = {
      version: this.#manifest.version,
      candidates: this.#manifest.candidates,
      uniqueTexts: this.#manifest.uniqueTexts,
      duplicates: this.#manifest.duplicates,
      partitions: this.#manifest.partitions,
      ingestSha256: this.#manifest.ingestSha256,
      selectedReferencesSha256: this.#manifest.selectedReferencesSha256,
    };
  }

  replay(): CorpusUnionReplay {
    return new CorpusUnionReplay(this.directory, this.#manifest);
  }
}

/**
 * One deterministic replay of the exact candidate sequence from pass one.
 * Winner ordinals stream from disk in the order replay needs them.
 */
export class CorpusUnionReplay {
  readonly #manifest: Manifest;
  readonly #winners: FixedRecordReader;
  readonly #digest: Hash = createHash("sha256");
  #nextWinner: number;
  #ordinal = 0;
  #selected = 0;
  #finished = false;

  constructor(directory: string, manifest: Manifest) {
    this.#manifest = manifest;
    this.#winners = new FixedRecordReader(join(directory, SELECTED_FILE), ORDINAL_BYTES);
    this.#nextWinner = this.#readWinner();
  }

  #readWinner(): number {
    const record = this.#winners.next();

    return record === undefined ? Number.POSITIVE_INFINITY : readOrdinal(record, 0);
  }

  add(record: CorpusRecord, options: CorpusUnionCandidateOptions = {}): CorpusUnionDecision {
    if (this.#finished) throw new Error("Corpus union replay is finished");
    const ordinal = this.#ordinal;
    const candidate = candidateBytes(record, ordinal, options);

    this.#digest.update(candidate.bytes);
    this.#ordinal++;
    if (ordinal !== this.#nextWinner) return { selected: false };
    this.#nextWinner = this.#readWinner();
    this.#selected++;

    return {
      selected: true,
      reference: candidate.reference,
      split: record.split,
      textHash: candidate.textHash.toString("hex"),
    };
  }

  /** Release the winner file without verifying; `finish` also closes it. */
  close(): void {
    this.#winners.close();
  }

  finish(): void {
    if (this.#finished) throw new Error("Corpus union replay is already finished");
    this.#finished = true;
    this.close();

    if (
      this.#ordinal !== this.#manifest.candidates ||
      this.#selected !== this.#manifest.uniqueTexts ||
      this.#digest.digest("hex") !== this.#manifest.ingestSha256
    ) {
      throw new Error("Corpus union replay differs from the planning pass");
    }
  }
}
