/**
 * Convert aligned corpus records to binary features and supervision.
 *
 * The corpus records recorded spans with UTF-16 offsets; the tokenizer turns
 * the same text into UTF-16-offset tokens. This module maps one onto the other
 * and rejects a span that does not cover a complete token. It never guesses
 * a label or falls back to a parser: misalignment is a generator bug.
 *
 * Every corpus record takes the same path. The shard contract is owned
 * here and is authoritative for the Python reader: `uint16` sparse feature rows
 * per token, `uint8` labels/boundaries/kinds/speech/actionable, `int16` neighbour
 * pairs, a `uint32` sequence-offset table of length `N + 1`, and one `uint32`
 * wording frame id per sequence. Speech-act is one uint8 per sequence;
 * actionable is one uint8 per token.
 */
import { isScheduleLabel, LABEL_COUNT, labelId, LABELS } from "../../src/model/labels.js";
import type { Label } from "../../src/model/labels.js";
import { SPEECH_ACTS, speechActId } from "../../src/model/speech-acts.js";
import type { SpeechAct } from "../../src/model/speech-acts.js";
import { FEATURE_COUNT, featureRows, PADDING_ROW, tokenize } from "../../src/model/tokenizer.js";
import type { CorpusRecord } from "./corpus/merge.js";
import type { SequenceMetadataManifest } from "./sequence-metadata.js";

/** Longest sequence the model will see; longer examples are skipped. */
export const MAX_TOKENS = 128;

/**
 * Whitespace has no role, so it carries an out-of-range sentinel rather than a
 * real label id. Every real id is in `[0, LABEL_COUNT)`; Python masks the rest.
 */
export const PADDING_LABEL = LABEL_COUNT;

/** Shard files, each named by suffix and storage dtype. */
export const SHARD_FORMAT = {
  rows: { suffix: ".rows.bin", dtype: "uint16" },
  labels: { suffix: ".labels.bin", dtype: "uint8" },
  boundaries: { suffix: ".boundaries.bin", dtype: "uint8" },
  kinds: { suffix: ".kinds.bin", dtype: "uint8" },
  neighbors: { suffix: ".neighbors.bin", dtype: "int16" },
  offsets: { suffix: ".offsets.bin", dtype: "uint32" },
  frames: { suffix: ".frames.bin", dtype: "uint32" },
  speech: { suffix: ".speech.bin", dtype: "uint8" },
  actionable: { suffix: ".actionable.bin", dtype: "uint8" },
  populations: { suffix: ".populations.bin", dtype: "uint8" },
} as const;

/** Sequence-level corpus population. Must match training/cron/dataset.py. */
export const POPULATION_IDS = { R: 0, H: 1, E: 2 } as const;

/** The manifest layout contract, including sequence-level wording frames. */
export interface FeaturizeManifest {
  sequenceMetadata: SequenceMetadataManifest;
  fingerprints: string[];
  roleClasses: number;
  rowsPerToken: number;
  featureRows: number;
  labelCounts: Record<string, number>;
  frameNames: string[];
  frameCounts: number[];
  speechActClasses: number;
  speechActNames: readonly SpeechAct[];
  actionableCounts: Record<0 | 1, number>;
  populationCounts: Record<"R" | "H" | "E", number>;
  format: typeof SHARD_FORMAT;
}

export interface PerRecord {
  rows: Uint16Array;
  labels: Uint8Array;
  boundaries: Uint8Array;
  kinds: Uint8Array;
  neighbors: Int16Array;
  tokenCount: number;
  labelCounts: Record<string, number>;
  speech: Uint8Array;
  actionable: Uint8Array;
}

/**
 * Featurize a record's already-tokenized text. The tokenizer and the recorded
 * spans share UTF-16 offsets, so alignment is a comparison of half-open
 * intervals.
 */
function isSpeechAct(value: string): value is SpeechAct {
  return (SPEECH_ACTS as readonly string[]).includes(value);
}

function speechActForRecord(record: CorpusRecord): SpeechAct {
  const name = record.speechAct ?? (record.intent !== null ? "request" : "none");

  if (!isSpeechAct(name)) throw new RangeError(`unknown speechAct: ${name}`);

  return name;
}

function defaultActionableBit(label: Label, speechAct: SpeechAct): boolean {
  if (speechAct !== "request") return false;

  return isScheduleLabel(label);
}

export function featurizeRecord(
  record: CorpusRecord,
  tokens: ReturnType<typeof tokenize>,
): PerRecord {
  const rows = new Uint16Array(tokens.length * FEATURE_COUNT).fill(PADDING_ROW);
  const labels = new Uint8Array(tokens.length).fill(PADDING_LABEL);
  const boundaries = new Uint8Array(tokens.length);
  const kinds = new Uint8Array(tokens.length);
  const actionable = new Uint8Array(tokens.length);
  const neighbors = new Int16Array(tokens.length * 2).fill(-1);
  const labelCounts: Record<string, number> = {};

  for (const label of LABELS) labelCounts[label] = 0;
  const speechAct = speechActForRecord(record);

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

      labels[index] = labelId[span.label];
      labelCounts[span.label] = (labelCounts[span.label] ?? 0) + 1;
      boundaries[index] = Number(span.clauseStart && token.start === span.start);
      actionable[index] = Number(span.actionable ?? defaultActionableBit(span.label, speechAct));
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
    speech: Uint8Array.of(speechActId[speechAct]),
    actionable,
  };
}
