/** Independent pure per-record feature pass over the actual published-candidate JSONL. */
import { createHash, type Hash } from "node:crypto";
import { StringDecoder } from "node:string_decoder";

import { LABELS } from "../../../../src/model/labels";
import { SCHEDULE_KINDS, type ScheduleKind } from "../../../../src/model/schedule-kinds";
import { SPEECH_ACTS } from "../../../../src/model/speech-acts";
import { EMBEDDING_ROWS, FEATURE_COUNT, tokenize } from "../../../../src/model/tokenizer";
import { featurizeRecord, MAX_TOKENS, POPULATION_IDS } from "../../featurize";
import { classifyPopulation } from "../../mix/quotas";
import { METADATA_SUFFIX, sequenceMetadata } from "../../sequence-metadata";
import { jsonlByteChunks } from "../jsonl-io";
import { type CorpusRecord, corpusRecordSchema, SPLITS } from "../merge";

const KIND_FRAMES: Record<ScheduleKind, readonly string[]> = {
  none: ["easy-no", "negative"],
  interval: [
    "unit-interval",
    "n-interval",
    "interval-at-time",
    "frequency-idiom",
    "minute-interval",
    "hour-interval",
    "day-interval",
    "week-interval",
    "month-interval",
    "correction-interval",
  ],
  daily: ["time-of-day", "day-at-time", "daily-clock", "correction-clock"],
  weekly: [
    "daygroup",
    "daygroup-at-time",
    "single-weekday",
    "weekday-list",
    "weekday-range",
    "weekday-at-time",
    "every-other",
    "every-other-at-time",
    "day-group",
    "correction-weekday",
  ],
  monthly: [
    "day-of-month",
    "day-of-month-at-time",
    "day-rule",
    "day-rule-at-time",
    "last-day-of-month",
    "last-business-day",
  ],
  ordinal: ["ordinal-weekday", "ordinal-weekday-at-time"],
  bounded: [
    "bounds",
    "bounds-at-time",
    "start-bound",
    "exclusive-end-bound",
    "inclusive-end-bound",
    "date-window",
  ],
  exception: [
    "exception-group",
    "exception-days",
    "selector-exception",
    "exception-date",
    "exception-date-at-time",
    "exception-at-time",
    "selector-exception-at-time",
    "weekday-exception",
    "date-exception",
    "selector-with-exception",
  ],
  ambiguous: [],
};

export function productionFrameTarget(frame: string): number {
  for (const [kind, names] of Object.entries(KIND_FRAMES)) {
    if (names.includes(frame)) return SCHEDULE_KINDS.indexOf(kind as ScheduleKind);
  }

  throw Error(`Unknown production frame taxonomy: ${frame}`);
}

const FIELDS = [
  "rows",
  "labels",
  "boundaries",
  "kinds",
  "neighbors",
  "offsets",
  "frames",
  "speech",
  "actionable",
  "populations",
] as const;

type Field = (typeof FIELDS)[number];

interface Digest {
  hash: Hash;
  bytes: number;
}

const make = (): Digest => ({ hash: createHash("sha256"), bytes: 0 });

function update(d: Digest, value: ArrayBufferView) {
  const b = Buffer.from(value.buffer, value.byteOffset, value.byteLength);

  d.hash.update(b);
  d.bytes += b.length;
}

function finish(d: Digest) {
  return { bytes: d.bytes, sha256: d.hash.digest("hex") };
}

export async function sourceExpectations(input: string, visit?: (row: CorpusRecord) => void) {
  const states = Object.fromEntries(
    SPLITS.map((split) => [
      split,
      {
        sequences: 0,
        tokens: 0,
        frames: new Map<string, number>(),
        frameCounts: [] as number[],
        labelCounts: Object.fromEntries(LABELS.map((label) => [label, 0])),
        actionableCounts: { "0": 0, "1": 0 },
        populationCounts: { R: 0, H: 0, E: 0 },
        fields: Object.fromEntries(FIELDS.map((key) => [key, make()])) as Record<Field, Digest>,
        scheduleKinds: make(),
        metadata: make(),
      },
    ]),
  ) as Record<
    (typeof SPLITS)[number],
    {
      sequences: number;
      tokens: number;
      frames: Map<string, number>;
      frameCounts: number[];
      labelCounts: Record<string, number>;
      actionableCounts: Record<"0" | "1", number>;
      populationCounts: Record<"R" | "H" | "E", number>;
      fields: Record<Field, Digest>;
      scheduleKinds: Digest;
      metadata: Digest;
    }
  >;

  for (const state of Object.values(states)) update(state.fields.offsets, Uint32Array.of(0));
  let records = 0;
  const sourceHash = createHash("sha256");

  const line = (text: string) => {
    if (!text.trim()) return;
    const row = corpusRecordSchema.parse(JSON.parse(text)) as CorpusRecord;

    visit?.(row);
    const tokens = tokenize(row.text);

    if (!tokens.length || tokens.length > MAX_TOKENS) {
      throw Error(`Unfeaturizable source record: ${row.id}`);
    }

    const f = featurizeRecord(row, tokens);
    const state = states[row.split];

    for (const key of [
      "rows",
      "labels",
      "boundaries",
      "kinds",
      "neighbors",
      "speech",
      "actionable",
    ] as const) {
      update(state.fields[key], f[key]);
    }

    const population = classifyPopulation({
      intent: row.intent,
      speechAct: SPEECH_ACTS[f.speech[0]!]!,
      spans: row.spans,
    });

    update(state.fields.populations, Uint8Array.of(POPULATION_IDS[population]));
    state.populationCounts[population]++;

    update(
      state.metadata,
      Buffer.from(`${JSON.stringify(sequenceMetadata(row, f, state.sequences, population))}\n`),
    );

    const frame = (row.template ?? `${row.family}/unspecified`).split("/")[0]!;

    if (!state.frames.has(frame)) {
      state.frames.set(frame, state.frames.size);
      state.frameCounts.push(0);
    }

    state.frameCounts[state.frames.get(frame)!]!++;
    for (const label of LABELS) state.labelCounts[label]! += f.labelCounts[label] ?? 0;

    for (let index = 0; index < f.tokenCount; index++) {
      if (f.kinds[index] !== 3) state.actionableCounts[f.actionable[index] === 1 ? "1" : "0"]++;
    }

    update(state.fields.frames, Uint32Array.of(state.frames.get(frame)!));
    update(state.scheduleKinds, Uint8Array.of(productionFrameTarget(frame)));
    state.tokens += f.tokenCount;
    if (state.tokens > 0xffffffff) throw Error("Token offset overflow");
    update(state.fields.offsets, Uint32Array.of(state.tokens));
    state.sequences++;
    records++;
  };

  const decoder = new StringDecoder("utf8");
  let pending = "";

  for await (const chunk of jsonlByteChunks(input, (artifactChunk) =>
    sourceHash.update(artifactChunk),
  )) {
    pending += decoder.write(chunk);

    let start = 0;
    let end: number;

    while ((end = pending.indexOf("\n", start)) !== -1) {
      line(pending.slice(start, end));
      start = end + 1;
    }

    pending = pending.slice(start);
  }

  pending += decoder.end();
  if (pending) line(pending);

  return {
    version: 1,
    scope: "source-derived-shard-expectations",
    rowsPerToken: FEATURE_COUNT,
    featureRows: EMBEDDING_ROWS,
    corpus: { records, sha256: sourceHash.digest("hex") },
    splits: Object.fromEntries(
      Object.entries(states).map(([split, state]) => [
        split,
        {
          sequences: state.sequences,
          tokens: state.tokens,
          frameNames: [...state.frames.keys()],
          frameCounts: state.frameCounts,
          labelCounts: state.labelCounts,
          actionableCounts: state.actionableCounts,
          populationCounts: state.populationCounts,
          fields: Object.fromEntries(FIELDS.map((key) => [key, finish(state.fields[key])])),
          scheduleKinds: finish(state.scheduleKinds),
          sequenceMetadata: {
            version: 1,
            suffix: METADATA_SUFFIX,
            rows: state.sequences,
            ...finish(state.metadata),
          },
        },
      ]),
    ),
  };
}
