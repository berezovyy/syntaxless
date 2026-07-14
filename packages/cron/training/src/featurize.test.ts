import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { LABEL_COUNT, labelId, LABELS } from "../../src/model/labels";
import { SPEECH_ACTS, speechActId } from "../../src/model/speech-acts";
import { EMBEDDING_ROWS, FEATURE_COUNT } from "../../src/model/tokenizer";
import type { ScheduleIntent } from "../../src/types/intent";
import type { CorpusRecord } from "./corpus/merge";
import { featurizeFiles, featurizeRecords, PADDING_LABEL } from "./featurize";
import { Sentence } from "./sentence";

const workdir = mkdtempSync(join(tmpdir(), "syntaxless-featurize-"));

afterAll(() => {
  rmSync(workdir, { recursive: true, force: true });
});

const MONDAY: ScheduleIntent = {
  frequency: { unit: "week", interval: 1 },
  weekdays: ["monday"],
};

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

function zeroCounts(): Record<string, number> {
  return Object.fromEntries(LABELS.map((label) => [label, 0]));
}

test("a well-formed example yields the expected labelCounts", () => {
  const record = makeRecord({
    id: "feat-1",
    text: "every Monday",
    spans: [
      { start: 0, end: 5, label: "RECUR", clauseStart: false },
      { start: 6, end: 12, label: "WEEKDAY_MON", clauseStart: true },
    ],
  });

  const shard = featurizeRecords([record]);

  const expected = zeroCounts();

  expected.RECUR = 1;
  expected.WEEKDAY_MON = 1;
  expect(shard.labelCounts).toEqual(expected);
  expect(shard.sequences).toBe(1);
  expect(shard.tokens).toBe(3);
  expect([...shard.labels]).toEqual([labelId.RECUR, PADDING_LABEL, labelId.WEEKDAY_MON]);
  expect([...shard.boundaries]).toEqual([0, 0, 1]);
  expect([...shard.kinds]).toEqual([0, 3, 0]);
});

test("a fragment starting mid-token throws Unaligned supervision", () => {
  const record = makeRecord({
    id: "misaligned-1",
    text: "every Monday",
    spans: [
      { start: 1, end: 5, label: "RECUR", clauseStart: false },
      { start: 6, end: 12, label: "WEEKDAY_MON", clauseStart: true },
    ],
  });

  expect(() => featurizeRecords([record])).toThrow(/Unaligned supervision in misaligned-1/);
});

test("serialized shards preserve exact supervision and sequence boundaries", () => {
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

  const summary = featurizeFiles({ input, out: join(workdir, "shard") });
  const rowsBytes = statSync(summary.files.rows).size;
  const labelBytes = statSync(summary.files.labels).size;

  expect(labelBytes).toBe(rowsBytes / 2 / summary.manifest.rowsPerToken);

  expect(summary.manifest).toMatchObject({
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

  const offsetBuffer = readFileSync(summary.files.offsets);

  const offsets = new Uint32Array(
    offsetBuffer.buffer,
    offsetBuffer.byteOffset,
    offsetBuffer.byteLength / 4,
  );

  expect([...offsets]).toEqual([0, 3, 8]);
  expect(summary.tokens).toBe(8);
  expect(summary.sequences).toBe(2);
  const bytes = (path: string) => [...readFileSync(path)];

  expect(bytes(summary.files.labels)).toEqual([
    labelId.RECUR,
    PADDING_LABEL,
    labelId.WEEKDAY_MON,
    labelId.RECUR,
    PADDING_LABEL,
    labelId.INTERVAL_2,
    PADDING_LABEL,
    labelId.UNIT_HOUR,
  ]);

  expect(bytes(summary.files.boundaries)).toEqual([0, 0, 1, 0, 0, 1, 0, 0]);
  expect(bytes(summary.files.kinds)).toEqual([0, 3, 0, 0, 3, 1, 3, 0]);
  expect(bytes(summary.files.speech)).toEqual([speechActId.request, speechActId.quoted]);
  expect(bytes(summary.files.actionable)).toEqual([1, 0, 1, 0, 0, 0, 0, 0]);
  const neighborBuffer = readFileSync(summary.files.neighbors);

  const neighbors = new Int16Array(
    neighborBuffer.buffer,
    neighborBuffer.byteOffset,
    neighborBuffer.byteLength / 2,
  );

  // Neighbor indices restart at each sequence and skip whitespace.
  expect([...neighbors]).toEqual([-1, 2, 0, 2, 0, -1, -1, 2, 0, 2, 0, 4, 2, 4, 2, -1]);
  expect(summary.manifest.speechActClasses).toBe(SPEECH_ACTS.length);
  expect(summary.manifest.speechActNames).toEqual([...SPEECH_ACTS]);

  expect(summary.manifest.format.speech).toEqual({
    suffix: ".speech.bin",
    dtype: "uint8",
  });

  expect(summary.manifest.format.actionable).toEqual({
    suffix: ".actionable.bin",
    dtype: "uint8",
  });
});

test("a request defaults structural tokens to actionable when the span omits the bit", () => {
  const record = makeRecord({
    id: "feat-request-structural",
    text: "every Monday",
    speechAct: "request",
    spans: [
      { start: 0, end: 5, label: "RECUR", clauseStart: false },
      { start: 6, end: 12, label: "WEEKDAY_MON", clauseStart: true },
    ],
  });

  const shard = featurizeRecords([record]);

  expect([...shard.actionable]).toEqual([1, 0, 1]);
});

test("a request writes speech-act id and content-token actionable bits", () => {
  const record = makeRecord({
    id: "feat-request",
    text: "every Monday",
    speechAct: "request",
    spans: [
      { start: 0, end: 5, label: "RECUR", clauseStart: false, actionable: false },
      { start: 6, end: 12, label: "WEEKDAY_MON", clauseStart: true, actionable: true },
    ],
  });

  const shard = featurizeRecords([record]);

  expect([...shard.speech]).toEqual([speechActId.request]);
  expect(shard.actionable.length).toBe(shard.tokens);
  expect([...shard.actionable]).toEqual([0, 0, 1]);
  expect(shard.actionableCounts).toEqual({ 0: 1, 1: 1 });
});

test("a descriptive sequence has descriptive speech-act and no actionable tokens", () => {
  const sentence = new Sentence();

  sentence.add("the shop is open", "O", " ", false);
  sentence.clause();
  sentence.add("every", "RECUR", " ", false);
  sentence.add("weekday", "DAYGROUP_WEEKDAY", " ", false);
  sentence.add("at", "GLUE", " ", false);
  sentence.add("9", "HOUR_9", " ", false);

  const record = makeRecord({
    id: "feat-descriptive",
    text: sentence.text,
    intent: null,
    speechAct: "descriptive",
    family: "negative",
    spans: sentence.spans,
  });

  const shard = featurizeRecords([record]);

  expect([...shard.speech]).toEqual([speechActId.descriptive]);
  expect(shard.actionable.length).toBe(shard.tokens);
  expect([...shard.actionable].every((bit) => bit === 0)).toBe(true);

  expect(
    shard.labels.includes(labelId.DAYGROUP_WEEKDAY) && shard.labels.includes(labelId.HOUR_9),
  ).toBe(true);
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

  const shard = featurizeRecords([record]);

  expect([...shard.speech]).toEqual([speechActId.correction]);
  const tokens = ["i", " ", "meant", " ", "friday", " ", "rather", " ", "than", " ", "thursday"];

  expect(shard.tokens).toBe(tokens.length);
  const friday = [...shard.labels].findIndex((id) => id === labelId.WEEKDAY_FRI);
  const thursday = [...shard.labels].lastIndexOf(labelId.WEEKDAY_THU);

  expect(friday).toBeGreaterThanOrEqual(0);
  expect(thursday).toBeGreaterThan(friday);
  expect(shard.actionable[friday]).toBe(1);
  expect(shard.actionable[thursday]).toBe(0);
  expect(shard.actionable[[...shard.kinds].indexOf(3)]).toBe(0);
});

test("an unknown speechAct is a hard error, never request", () => {
  const record = makeRecord({
    id: "feat-unknown-act",
    text: "every Monday",
    speechAct: "please" as CorpusRecord["speechAct"],
    spans: [
      { start: 0, end: 5, label: "RECUR", clauseStart: false },
      { start: 6, end: 12, label: "WEEKDAY_MON", clauseStart: true },
    ],
  });

  expect(() => featurizeRecords([record])).toThrow(/unknown speechAct/);
});
