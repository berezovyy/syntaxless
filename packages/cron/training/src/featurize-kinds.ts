/**
 * Write production binary shards and metadata in one streaming pass.
 *
 * Read one JSONL record at a time and write all three split shards with bounded
 * buffers. The shard contract is consumed by `training/cron/dataset.py`.
 */
import { closeSync, mkdirSync, openSync, writeFileSync, writeSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { LABEL_COUNT, LABELS } from "../../src/model/labels.js";
import { SPEECH_ACT_COUNT, SPEECH_ACTS } from "../../src/model/speech-acts.js";
import { EMBEDDING_ROWS, FEATURE_COUNT, tokenize } from "../../src/model/tokenizer.js";
import { jsonlByteChunks, jsonlReader } from "./corpus/jsonl-io.js";
import { corpusRecordSchema } from "./corpus/merge.js";
import type { CorpusRecord, Split } from "./corpus/merge.js";
import { featurizeRecord, MAX_TOKENS, POPULATION_IDS, SHARD_FORMAT } from "./featurize.js";
import type { FeaturizeManifest, PerRecord } from "./featurize.js";
import { classifyPopulation } from "./mix/quotas.js";
import { METADATA_SUFFIX, sequenceMetadata, SequenceMetadataWriter } from "./sequence-metadata.js";

const SPLITS: readonly Split[] = ["train", "validation", "heldout"];
const repo = fileURLToPath(new URL("../../../../", import.meta.url));
// Keep one fixed buffer per output file.  Retaining thousands of small typed
// array backing stores until Buffer.concat() was cheap for small corpora, but
// caused Bun's RSS to grow until Linux killed production featurization.
const WRITE_BUFFER_BYTES = 1024 * 1024;

class BufferedWriter {
  readonly #fd: number;
  readonly #buffer = Buffer.allocUnsafe(WRITE_BUFFER_BYTES);
  #offset = 0;

  constructor(fd: number) {
    this.#fd = fd;
  }

  append(view: ArrayBufferView): void {
    if (view.byteLength === 0) return;
    const input = Buffer.from(view.buffer, view.byteOffset, view.byteLength);
    let inputOffset = 0;

    while (inputOffset < input.length) {
      if (this.#offset === this.#buffer.length) this.flush();
      const bytes = Math.min(input.length - inputOffset, this.#buffer.length - this.#offset);

      input.copy(this.#buffer, this.#offset, inputOffset, inputOffset + bytes);
      this.#offset += bytes;
      inputOffset += bytes;
    }
  }

  flush(): void {
    let written = 0;

    while (written < this.#offset) {
      written += writeSync(this.#fd, this.#buffer, written, this.#offset - written);
    }

    this.#offset = 0;
  }
}

interface SplitState {
  metadata: SequenceMetadataWriter;
  split: Split;
  prefix: string;
  writers: Record<keyof typeof SHARD_FORMAT, BufferedWriter>;
  sequences: number;
  tokens: number;
  skippedEmpty: number;
  skippedLong: number;
  labelCounts: Record<string, number>;
  fingerprints: number;
  frameIds: Map<string, number>;
  frameNames: string[];
  frameCounts: number[];
  actionableCounts: Record<0 | 1, number>;
  populationCounts: Record<"R" | "H" | "E", number>;
}

export interface StreamingSplitSummary {
  split: Split;
  prefix: string;
  sequences: number;
  tokens: number;
  skippedEmpty: number;
  skippedLong: number;
  fingerprints: number;
}

export interface StreamingFeaturizeSummary {
  input: string;
  outputBase: string;
  records: number;
  splits: Record<Split, StreamingSplitSummary>;
}

function createState(outputBase: string, split: Split, files: number[]): SplitState {
  const prefix = `${outputBase}-${split}`;

  mkdirSync(dirname(prefix), { recursive: true });
  const writers = {} as SplitState["writers"];

  for (const key of Object.keys(SHARD_FORMAT) as Array<keyof typeof SHARD_FORMAT>) {
    const fd = openSync(`${prefix}${SHARD_FORMAT[key].suffix}`, "w");

    files.push(fd);
    writers[key] = new BufferedWriter(fd);
  }

  const metadataFile = openSync(`${prefix}${METADATA_SUFFIX}`, "w");

  files.push(metadataFile);
  writers.offsets.append(Uint32Array.of(0));

  return {
    split,
    prefix,
    metadata: new SequenceMetadataWriter(metadataFile),
    writers,
    sequences: 0,
    tokens: 0,
    skippedEmpty: 0,
    skippedLong: 0,
    labelCounts: Object.fromEntries(LABELS.map((label) => [label, 0])),
    fingerprints: 0,
    frameIds: new Map<string, number>(),
    frameNames: [],
    frameCounts: [],
    actionableCounts: { 0: 0, 1: 0 },
    populationCounts: { R: 0, H: 0, E: 0 },
  };
}

function appendRecord(state: SplitState, record: CorpusRecord): void {
  const tokens = tokenize(record.text);

  if (tokens.length === 0) {
    state.skippedEmpty++;

    return;
  }

  if (tokens.length > MAX_TOKENS) {
    state.skippedLong++;

    return;
  }

  const shard: PerRecord = featurizeRecord(record, tokens);

  state.writers.rows.append(shard.rows);
  state.writers.labels.append(shard.labels);
  state.writers.boundaries.append(shard.boundaries);
  state.writers.kinds.append(shard.kinds);
  state.writers.neighbors.append(shard.neighbors);
  state.writers.speech.append(shard.speech);
  state.writers.actionable.append(shard.actionable);

  const population = classifyPopulation({
    intent: record.intent,
    speechAct: SPEECH_ACTS[shard.speech[0]!]!,
    spans: record.spans,
  });

  state.writers.populations.append(Uint8Array.of(POPULATION_IDS[population]));
  state.populationCounts[population]++;
  state.metadata.append(sequenceMetadata(record, shard, state.sequences, population));

  for (let index = 0; index < shard.tokenCount; index++) {
    if (shard.kinds[index] === 3) continue;
    const bit = shard.actionable[index]! === 1 ? 1 : 0;

    state.actionableCounts[bit] += 1;
  }

  const rawFrame = record.template ?? `${record.family}/unspecified`;
  const frameName = rawFrame.split("/")[0] ?? rawFrame;
  let frameId = state.frameIds.get(frameName);

  if (frameId === undefined) {
    frameId = state.frameNames.length;
    state.frameIds.set(frameName, frameId);
    state.frameNames.push(frameName);
    state.frameCounts.push(0);
  }

  state.writers.frames.append(Uint32Array.of(frameId));
  state.frameCounts[frameId]!++;
  state.sequences++;
  state.tokens += shard.tokenCount;

  if (state.tokens > 0xffff_ffff) {
    throw new RangeError(`${state.split} token offsets exceed uint32 capacity`);
  }

  state.writers.offsets.append(Uint32Array.of(state.tokens));
  state.fingerprints += 1;

  for (const label of LABELS) {
    state.labelCounts[label]! += shard.labelCounts[label] ?? 0;
  }
}

function parseBackwardRecord(line: string, lineNumber: number): CorpusRecord {
  let value: unknown;

  try {
    value = JSON.parse(line);
  } catch {
    throw new Error(`line ${lineNumber}: invalid JSON`);
  }

  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`line ${lineNumber}: expected an object`);
  }

  const normalized = { ...(value as Record<string, unknown>) };

  delete normalized.seed;
  normalized.source ??= "rule";
  const parsed = corpusRecordSchema.safeParse(normalized);

  if (!parsed.success) {
    throw new Error(
      `line ${lineNumber}: invalid corpus record: ${parsed.error.issues
        .map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`)
        .join("; ")}`,
    );
  }

  return parsed.data as CorpusRecord;
}

function finishState(state: SplitState): FeaturizeManifest {
  for (const writer of Object.values(state.writers)) writer.flush();

  return {
    sequenceMetadata: state.metadata.finish(),
    fingerprints: [],
    roleClasses: LABEL_COUNT,
    rowsPerToken: FEATURE_COUNT,
    featureRows: EMBEDDING_ROWS,
    labelCounts: state.labelCounts,
    frameNames: state.frameNames,
    frameCounts: state.frameCounts,
    speechActClasses: SPEECH_ACT_COUNT,
    speechActNames: SPEECH_ACTS,
    actionableCounts: state.actionableCounts,
    populationCounts: state.populationCounts,
    format: SHARD_FORMAT,
  };
}

export async function featurizeKindsStreaming(
  inputValue: string,
  outputBaseValue: string,
): Promise<StreamingFeaturizeSummary> {
  const input = resolve(repo, inputValue);
  const outputBase = resolve(repo, outputBaseValue);
  const states = {} as Record<Split, SplitState>;
  const files: number[] = [];
  const failures: unknown[] = [];
  const manifests: Array<readonly [string, FeaturizeManifest]> = [];
  let records = 0;

  try {
    for (const split of SPLITS) states[split] = createState(outputBase, split, files);

    {
      await using reader = jsonlReader(jsonlByteChunks(input));

      for await (const line of reader.lines) {
        if (line.trim().length === 0) continue;
        records++;
        const record = parseBackwardRecord(line, records);

        appendRecord(states[record.split], record);

        if (records % 100_000 === 0) {
          // Force collection of short-lived tokenizer/feature arrays. Bun's
          // automatic collection can lag behind multi-hour allocation-heavy streams.
          Bun.gc(true);
          const rssGiB = process.memoryUsage.rss() / 1024 ** 3;

          process.stdout.write(
            `featurized ${records.toLocaleString()} records (${rssGiB.toFixed(2)} GiB RSS)\r`,
          );
        }
      }
    }

    // Split disjointness is enforced by the generator. Do not retain all
    // fingerprints here: streaming corpora can contain tens of millions of rows.
    for (const state of Object.values(states)) manifests.push([state.prefix, finishState(state)]);
  } catch (error) {
    failures.push(error);
  } finally {
    for (const fd of files) {
      try {
        closeSync(fd);
      } catch (error) {
        failures.push(error);
      }
    }
  }

  if (failures.length === 1) throw failures[0];

  if (failures.length > 1) {
    throw new AggregateError(failures, "Shard preparation and cleanup failed");
  }

  for (const [prefix, manifest] of manifests) {
    writeFileSync(`${prefix}.json`, `${JSON.stringify(manifest, null, 2)}\n`);
  }

  process.stdout.write("\n");

  return {
    input,
    outputBase,
    records,
    splits: Object.fromEntries(
      SPLITS.map((split) => {
        const state = states[split];

        return [
          split,
          {
            split,
            prefix: state.prefix,
            sequences: state.sequences,
            tokens: state.tokens,
            skippedEmpty: state.skippedEmpty,
            skippedLong: state.skippedLong,
            fingerprints: state.fingerprints,
          },
        ];
      }),
    ) as Record<Split, StreamingSplitSummary>,
  };
}
