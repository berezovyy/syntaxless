import { afterAll, expect, test } from "bun:test";
import {
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  readlinkSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, relative, resolve } from "node:path";

import { LABEL_COUNT, labelId } from "../../src/model/labels";
import { SPEECH_ACTS, speechActId } from "../../src/model/speech-acts";
import { EMBEDDING_ROWS, FEATURE_COUNT, tokenize } from "../../src/model/tokenizer";
import type { ScheduleIntent } from "../../src/types/intent";
import type { CorpusRecord } from "./corpus/merge";
import { featurizeRecord, PADDING_LABEL, SHARD_FORMAT } from "./featurize";
import type { FeaturizeManifest } from "./featurize";
import { featurizeKindsStreaming } from "./featurize-kinds";
import { Sentence } from "./sentence";

const workdir = mkdtempSync(join(tmpdir(), "syntaxless-featurize-"));

afterAll(() => {
  rmSync(workdir, { recursive: true, force: true });
});

const MONDAY: ScheduleIntent = {
  frequency: { unit: "week", interval: 1 },
  weekdays: ["monday"],
};

test("context-number supervision is retained without actionability or a clause boundary", () => {
  const sentence = new Sentence();

  sentence.clause();
  sentence.add("account");
  sentence.add("17", "CONTEXT_NUMBER");
  sentence.add("daily", "UNIT_DAY");

  const row: CorpusRecord = {
    id: "context-number",
    fingerprint: "fixture-context-number",
    split: "train",
    family: "interval",
    text: sentence.text,
    spans: sentence.spans,
    intent: { frequency: { unit: "day", interval: 1 } },
    source: "rule",
  };

  const tokens = tokenize(row.text);
  const data = featurizeRecord(row, tokens);
  const number = tokens.findIndex((token) => token.text === "17");
  const daily = tokens.findIndex((token) => token.text === "daily");

  expect(data.labels[number]).toBe(labelId.CONTEXT_NUMBER);
  expect(data.actionable[number]).toBe(0);
  expect(data.boundaries[number]).toBe(0);
  expect(data.boundaries[daily]).toBe(1);
  expect(data.actionable[daily]).toBe(1);
});

/** Inspect real kernel handles; a disappearing descriptor is normal during enumeration. */
function openShardFiles(directory: string): string[] {
  const paths: string[] = [];

  for (const name of readdirSync("/proc/self/fd")) {
    let path: string;

    try {
      path = readlinkSync(`/proc/self/fd/${name}`);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") continue;
      throw error;
    }

    if (
      path.startsWith(`${directory}/out-`) ||
      path.startsWith(`${directory}/input`) ||
      path === "/dev/full"
    ) {
      paths.push(path);
    }
  }

  return paths.sort();
}

test.skipIf(process.platform !== "linux")(
  "a real shard flush failure releases all files and publishes no manifests",
  async () => {
    const directory = mkdtempSync(join(workdir, "flush-failure-"));
    const input = join(directory, "input.jsonl");

    writeFileSync(
      input,
      JSON.stringify(
        makeRecord({
          id: "disk-full",
          text: "every Monday",
          spans: [
            { start: 0, end: 5, label: "RECUR", clauseStart: false },
            { start: 6, end: 12, label: "WEEKDAY_MON", clauseStart: true },
          ],
        }),
      ) + "\n",
    );

    symlinkSync("/dev/full", join(directory, "out-train.rows.bin"));
    expect(openShardFiles(directory)).toEqual([]);

    await expect(featurizeKindsStreaming(input, join(directory, "out"))).rejects.toThrow(
      /ENOSPC|no space/i,
    );

    expect(openShardFiles(directory)).toEqual([]);

    for (const split of ["train", "validation", "heldout"]) {
      expect(existsSync(join(directory, `out-${split}.json`))).toBe(false);
    }
  },
);

interface RecordOptions {
  id: string;
  text: string;
  spans: CorpusRecord["spans"];
  fingerprint?: string;
  split?: CorpusRecord["split"];
  family?: CorpusRecord["family"];
  intent?: CorpusRecord["intent"];
  speechAct?: CorpusRecord["speechAct"];
}

function makeRecord(options: RecordOptions): CorpusRecord {
  return {
    id: options.id,
    split: options.split ?? "train",
    family: options.family ?? "weekday-list",
    fingerprint: options.fingerprint ?? `fp-${options.id}`,
    text: options.text,
    intent: options.intent === undefined ? MONDAY : options.intent,
    ...(options.speechAct === undefined ? {} : { speechAct: options.speechAct }),
    spans: options.spans,
    source: "rule",
  };
}

test("a fragment starting mid-token throws Unaligned supervision", () => {
  const record = makeRecord({
    id: "misaligned-1",
    text: "every Monday",
    spans: [
      { start: 1, end: 5, label: "RECUR", clauseStart: false },
      { start: 6, end: 12, label: "WEEKDAY_MON", clauseStart: true },
    ],
  });

  expect(() => featurizeRecord(record, tokenize(record.text))).toThrow(
    /Unaligned supervision in misaligned-1/,
  );
});

test("streaming shards preserve exact supervision and sequence boundaries", async () => {
  const records = [
    makeRecord({
      id: "feat-3",
      text: "every Monday",
      speechAct: "request",
      spans: [
        { start: 0, end: 5, label: "RECUR", clauseStart: false },
        { start: 6, end: 12, label: "WEEKDAY_MON", clauseStart: true },
      ],
    }),
    makeRecord({
      id: "feat-4",
      text: "every 2 hours",
      intent: null,
      family: "negative",
      speechAct: "quoted",
      spans: [
        { start: 0, end: 5, label: "RECUR", clauseStart: false },
        { start: 6, end: 7, label: "INTERVAL_2", clauseStart: true },
        { start: 8, end: 13, label: "UNIT_HOUR", clauseStart: false },
      ],
    }),
  ];

  const input = join(workdir, "corpus.jsonl");

  writeFileSync(input, `${records.map((r) => JSON.stringify(r)).join("\n")}\n`);

  const repository = resolve(import.meta.dir, "../../../..");

  const summary = await featurizeKindsStreaming(
    relative(repository, input),
    relative(repository, join(workdir, "shard")),
  );

  const split = summary.splits.train;

  expect(split.prefix).toBe(join(workdir, "shard-train"));
  const manifest: FeaturizeManifest = JSON.parse(readFileSync(`${split.prefix}.json`, "utf8"));
  const file = (field: keyof typeof SHARD_FORMAT) => `${split.prefix}${SHARD_FORMAT[field].suffix}`;
  const rowsBytes = statSync(file("rows")).size;
  const labelBytes = statSync(file("labels")).size;

  expect(labelBytes).toBe(rowsBytes / 2 / manifest.rowsPerToken);

  expect(manifest).toMatchObject({
    roleClasses: LABEL_COUNT,
    rowsPerToken: FEATURE_COUNT,
    featureRows: EMBEDDING_ROWS,
    format: {
      rows: { suffix: ".rows.bin", dtype: "uint16" },
      labels: { suffix: ".labels.bin", dtype: "uint8" },
      boundaries: { suffix: ".boundaries.bin", dtype: "uint8" },
      kinds: { suffix: ".kinds.bin", dtype: "uint8" },
      neighbors: { suffix: ".neighbors.bin", dtype: "int16" },
      offsets: { suffix: ".offsets.bin", dtype: "uint32" },
    },
  });

  const offsetBuffer = readFileSync(file("offsets"));

  const offsets = new Uint32Array(
    offsetBuffer.buffer,
    offsetBuffer.byteOffset,
    offsetBuffer.byteLength / 4,
  );

  expect([...offsets]).toEqual([0, 3, 8]);
  expect(split.tokens).toBe(8);
  expect(split.sequences).toBe(2);
  const bytes = (path: string) => [...readFileSync(path)];

  expect(bytes(file("labels"))).toEqual([
    labelId.RECUR,
    PADDING_LABEL,
    labelId.WEEKDAY_MON,
    labelId.RECUR,
    PADDING_LABEL,
    labelId.INTERVAL_2,
    PADDING_LABEL,
    labelId.UNIT_HOUR,
  ]);

  expect(bytes(file("boundaries"))).toEqual([0, 0, 1, 0, 0, 1, 0, 0]);
  expect(bytes(file("kinds"))).toEqual([0, 3, 0, 0, 3, 1, 3, 0]);
  expect(bytes(file("speech"))).toEqual([speechActId.request, speechActId.quoted]);
  expect(bytes(file("actionable"))).toEqual([1, 0, 1, 0, 0, 0, 0, 0]);
  expect(bytes(file("populations"))).toEqual([0, 1]);
  expect(manifest.actionableCounts).toEqual({ 0: 3, 1: 2 });
  const neighborBuffer = readFileSync(file("neighbors"));

  const neighbors = new Int16Array(
    neighborBuffer.buffer,
    neighborBuffer.byteOffset,
    neighborBuffer.byteLength / 2,
  );

  // Neighbor indices restart at each sequence and skip whitespace.
  expect([...neighbors]).toEqual([-1, 2, 0, 2, 0, -1, -1, 2, 0, 2, 0, 4, 2, 4, 2, -1]);
  expect(manifest.speechActClasses).toBe(SPEECH_ACTS.length);
  expect(manifest.speechActNames).toEqual([...SPEECH_ACTS]);

  expect(manifest.format.speech).toEqual({
    suffix: ".speech.bin",
    dtype: "uint8",
  });

  expect(manifest.format.actionable).toEqual({
    suffix: ".actionable.bin",
    dtype: "uint8",
  });
});

test("a correction sequence marks only B tokens actionable", () => {
  const record = makeRecord({
    id: "feat-correction",
    text: "i meant friday rather than thursday",
    speechAct: "correction",
    intent: { frequency: { unit: "week", interval: 1 }, weekdays: ["friday"] },
    family: "terse",
    spans: [
      { start: 0, end: 7, label: "O", clauseStart: false, actionable: false },
      { start: 8, end: 14, label: "WEEKDAY_FRI", clauseStart: true, actionable: true },
      { start: 15, end: 26, label: "O", clauseStart: false, actionable: false },
      {
        start: 27,
        end: 35,
        label: "WEEKDAY_THU",
        clauseStart: false,
        actionable: false,
      },
    ],
  });

  const shard = featurizeRecord(record, tokenize(record.text));

  expect([...shard.speech]).toEqual([speechActId.correction]);
  const tokens = ["i", " ", "meant", " ", "friday", " ", "rather", " ", "than", " ", "thursday"];

  expect(shard.tokenCount).toBe(tokens.length);
  const friday = [...shard.labels].findIndex((id) => id === labelId.WEEKDAY_FRI);
  const thursday = [...shard.labels].lastIndexOf(labelId.WEEKDAY_THU);

  expect(friday).toBeGreaterThanOrEqual(0);
  expect(thursday).toBeGreaterThan(friday);
  expect(shard.actionable[friday]).toBe(1);
  expect(shard.actionable[thursday]).toBe(0);
  expect(shard.actionable[[...shard.kinds].indexOf(3)]).toBe(0);
});
