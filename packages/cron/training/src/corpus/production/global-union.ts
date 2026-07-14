/**
 * Bounded-memory, payload-free planning for the complete corpus union.
 *
 * Pass one writes only fixed-width hashes and source ordinals to partition
 * files. Reduction keeps one deterministic source reference for each
 * runtime-normalized sentence. A single deterministic replay then emits the
 * selected records themselves. Record JSON is never copied into scratch.
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

const FORMAT_VERSION = 1;
const HASH_BYTES = 32;
const ORDINAL_BYTES = 8;
const CANDIDATE_BYTES = HASH_BYTES * 3 + ORDINAL_BYTES + 4 + 1;
const WRITE_BUFFER_BYTES = 1 << 20;
const READ_RECORDS = 8_192;

type Split = CorpusRecord["split"];

const SPLIT_CODE: Readonly<Record<Split, number>> = {
  train: 0,
  validation: 1,
  heldout: 2,
};

export interface CorpusUnionOptions {
  /** Empty directory owned by this operation. */
  directory: string;
  /** More partitions reduce peak memory without requiring more source passes. */
  partitions?: number;
}

export interface CorpusUnionCandidateOptions {
  /** Lower values win. Defaults to an authorship-aware source priority. */
  priority?: number;
  /** Stable source reference. Defaults to provenance plus the record id. */
  reference?: string;
}

export interface CorpusUnionSummary {
  version: 1;
  candidates: number;
  uniqueTexts: number;
  duplicates: number;
  partitions: number;
  ingestSha256: string;
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

interface ReducedEntry {
  signature: Buffer;
  referenceHash: Buffer;
  ordinal: bigint;
  priority: number;
  split: number;
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

function candidateBytes(
  record: CorpusRecord,
  ordinal: bigint,
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
  supervisionHash(record).copy(bytes, HASH_BYTES);
  sha256(reference).copy(bytes, HASH_BYTES * 2);
  bytes.writeBigUInt64BE(ordinal, HASH_BYTES * 3);
  bytes.writeInt32BE(priority, HASH_BYTES * 3 + ORDINAL_BYTES);
  bytes.writeUInt8(SPLIT_CODE[record.split], CANDIDATE_BYTES - 1);

  return { bytes, textHash, reference };
}

class BufferedBinaryWriter {
  readonly #fd: number;
  readonly #buffer = Buffer.allocUnsafe(WRITE_BUFFER_BYTES);
  #used = 0;
  #closed = false;

  constructor(path: string) {
    this.#fd = openSync(path, "wx");
  }

  write(bytes: Buffer): void {
    if (this.#closed) throw new Error("Corpus union writer is closed");

    if (bytes.length > this.#buffer.length) {
      this.flush();
      writeSync(this.#fd, bytes);

      return;
    }

    if (this.#used + bytes.length > this.#buffer.length) this.flush();
    bytes.copy(this.#buffer, this.#used);
    this.#used += bytes.length;
  }

  flush(): void {
    if (this.#used > 0) {
      writeSync(this.#fd, this.#buffer, 0, this.#used);
      this.#used = 0;
    }
  }

  close(): void {
    if (this.#closed) return;
    this.flush();
    closeSync(this.#fd);
    this.#closed = true;
  }
}

function partitionPath(directory: string, partition: number): string {
  return join(directory, `candidates-${String(partition).padStart(4, "0")}.bin`);
}

function compareEntry(left: ReducedEntry, right: ReducedEntry): number {
  if (left.priority !== right.priority) return left.priority - right.priority;
  const byReference = Buffer.compare(left.referenceHash, right.referenceHash);

  if (byReference !== 0) return byReference;

  return left.ordinal < right.ordinal ? -1 : left.ordinal > right.ordinal ? 1 : 0;
}

function forEachFixedRecord(
  path: string,
  recordBytes: number,
  visit: (record: Buffer) => void,
): void {
  const size = statSync(path).size;

  if (size % recordBytes !== 0) throw new Error(`Truncated corpus union scratch file: ${path}`);
  const fd = openSync(path, "r");
  const buffer = Buffer.allocUnsafe(recordBytes * READ_RECORDS);

  try {
    let position = 0;

    while (position < size) {
      const wanted = Math.min(buffer.length, size - position);
      const read = readSync(fd, buffer, 0, wanted, position);

      if (read !== wanted) throw new Error(`Short read from corpus union scratch: ${path}`);

      for (let offset = 0; offset < read; offset += recordBytes) {
        visit(buffer.subarray(offset, offset + recordBytes));
      }

      position += read;
    }
  } finally {
    closeSync(fd);
  }
}

function selectOrdinal(selection: Buffer, ordinal: bigint): void {
  const value = Number(ordinal);
  const byte = Math.floor(value / 8);

  selection[byte] = selection[byte]! | (1 << (value & 7));
}

function isOrdinalSelected(selection: Buffer, ordinal: number): boolean {
  return (selection[Math.floor(ordinal / 8)]! & (1 << (ordinal & 7))) !== 0;
}

/** First-pass collector and reducer. */
export class StreamingCorpusUnion {
  readonly #directory: string;
  readonly #partitions: number;
  readonly #writers: (BufferedBinaryWriter | undefined)[];
  readonly #digest: Hash = createHash("sha256");
  #candidates = 0;
  #phase: "collect" | "finalized" | "aborted" = "collect";

  constructor(options: CorpusUnionOptions) {
    const partitions = options.partitions ?? 64;

    if (!Number.isInteger(partitions) || partitions < 1 || partitions > 256) {
      throw new Error("Corpus union partitions must be an integer from 1 to 256");
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
    this.#writers = Array.from({ length: partitions });
  }

  add(record: CorpusRecord, options: CorpusUnionCandidateOptions = {}): void {
    if (this.#phase !== "collect") throw new Error("Corpus union collection is closed");
    const ordinal = BigInt(this.#candidates);
    const candidate = candidateBytes(record, ordinal, options);
    const partition = candidate.textHash.readUInt32BE(0) % this.#partitions;

    const writer = (this.#writers[partition] ??= new BufferedBinaryWriter(
      partitionPath(this.#directory, partition),
    ));

    writer.write(candidate.bytes);
    this.#digest.update(candidate.bytes);
    this.#candidates++;
  }

  finalize(): CorpusUnionPlan {
    if (this.#phase !== "collect") throw new Error("Corpus union can only finalize once");
    for (const writer of this.#writers) writer?.close();
    const ingestSha256 = this.#digest.digest("hex");
    let uniqueTexts = 0;
    // One bit per input candidate is the complete selected-reference plan:
    // about 2.75 MiB for 22 million candidates.
    const selection = Buffer.alloc(Math.ceil(this.#candidates / 8));

    try {
      for (let partition = 0; partition < this.#partitions; partition++) {
        const source = partitionPath(this.#directory, partition);

        // Lazy writers avoid holding one descriptor and one MiB buffer per
        // partition. Materialize only a zero-byte marker for empty buckets.
        if (!existsSync(source)) writeFileSync(source, Buffer.alloc(0));
        const byText = new Map<string, ReducedEntry>();

        forEachFixedRecord(source, CANDIDATE_BYTES, (bytes) => {
          const key = bytes.subarray(0, HASH_BYTES).toString("hex");

          const next: ReducedEntry = {
            signature: Buffer.from(bytes.subarray(HASH_BYTES, HASH_BYTES * 2)),
            referenceHash: Buffer.from(bytes.subarray(HASH_BYTES * 2, HASH_BYTES * 3)),
            ordinal: bytes.readBigUInt64BE(HASH_BYTES * 3),
            priority: bytes.readInt32BE(HASH_BYTES * 3 + ORDINAL_BYTES),
            split: bytes.readUInt8(CANDIDATE_BYTES - 1),
          };

          const prior = byText.get(key);

          if (!prior) {
            byText.set(key, next);

            return;
          }

          if (!prior.signature.equals(next.signature)) {
            throw new Error(
              `Corpus union conflicting supervision for normalized-text ${key}: ` +
                `ordinals ${prior.ordinal} and ${next.ordinal}`,
            );
          }

          if (compareEntry(next, prior) < 0) byText.set(key, next);
        });

        uniqueTexts += byText.size;
        for (const winner of byText.values()) selectOrdinal(selection, winner.ordinal);
        unlinkSync(source);
      }

      const selectedPath = join(this.#directory, "selected-references.bitmap");

      writeFileSync(selectedPath, selection);

      const summary: CorpusUnionSummary = {
        version: FORMAT_VERSION,
        candidates: this.#candidates,
        uniqueTexts,
        duplicates: this.#candidates - uniqueTexts,
        partitions: this.#partitions,
        ingestSha256,
        selectedReferencesSha256: sha256(selection).toString("hex"),
      };

      const manifest: Manifest = {
        ...summary,
        candidateBytes: CANDIDATE_BYTES,
        selectionBytes: selection.length,
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
      this.#manifest.selectionBytes !== Math.ceil(this.#manifest.candidates / 8)
    ) {
      throw new Error("Unsupported corpus union plan format");
    }

    const selected = readFileSync(join(directory, "selected-references.bitmap"));

    if (selected.length !== this.#manifest.selectionBytes) {
      throw new Error("Corpus union winner plan has the wrong size");
    }

    if (sha256(selected).toString("hex") !== this.#manifest.selectedReferencesSha256) {
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

/** One deterministic replay of the exact candidate sequence from pass one. */
export class CorpusUnionReplay {
  readonly #manifest: Manifest;
  readonly #selection: Buffer;
  readonly #digest: Hash = createHash("sha256");
  #ordinal = 0;
  #selected = 0;
  #finished = false;

  constructor(directory: string, manifest: Manifest) {
    this.#manifest = manifest;
    this.#selection = readFileSync(join(directory, "selected-references.bitmap"));
  }

  add(record: CorpusRecord, options: CorpusUnionCandidateOptions = {}): CorpusUnionDecision {
    if (this.#finished) throw new Error("Corpus union replay is finished");
    const ordinal = this.#ordinal;
    const candidate = candidateBytes(record, BigInt(ordinal), options);

    this.#digest.update(candidate.bytes);
    this.#ordinal++;
    if (!isOrdinalSelected(this.#selection, ordinal)) return { selected: false };
    this.#selected++;

    return {
      selected: true,
      reference: candidate.reference,
      split: record.split,
      textHash: candidate.textHash.toString("hex"),
    };
  }

  finish(): void {
    if (this.#finished) throw new Error("Corpus union replay is already finished");
    this.#finished = true;

    if (
      this.#ordinal !== this.#manifest.candidates ||
      this.#selected !== this.#manifest.uniqueTexts ||
      this.#digest.digest("hex") !== this.#manifest.ingestSha256
    ) {
      throw new Error("Corpus union replay differs from the planning pass");
    }
  }
}
