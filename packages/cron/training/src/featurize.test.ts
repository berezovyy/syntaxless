import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { LABEL_COUNT, labelId, LABELS } from "../../src/model/labels";
import { EMBEDDING_ROWS, FEATURE_COUNT } from "../../src/model/tokenizer";
import type { ScheduleIntent } from "../../src/types/intent";
import type { CorpusRecord } from "./corpus/merge";
import { featurizeFiles, featurizeRecords, PADDING_LABEL } from "./featurize";

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
}

function makeRecord(options: RecordOptions): CorpusRecord {
  return {
    id: options.id,
    split: options.split ?? "train",
    family: options.family ?? "weekday-list",
    fingerprint: options.fingerprint ?? `fp-${options.id}`,
    text: options.text,
    intent: MONDAY,
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
      { start: 6, end: 12, label: "WEEKDAY", clauseStart: true },
    ],
  });

  const shard = featurizeRecords([record]);

  const expected = zeroCounts();

  expected.RECUR = 1;
  expected.WEEKDAY = 1;
  expect(shard.labelCounts).toEqual(expected);
  expect(shard.sequences).toBe(1);
  expect(shard.tokens).toBe(3);
  expect([...shard.labels]).toEqual([labelId.RECUR, PADDING_LABEL, labelId.WEEKDAY]);
  expect([...shard.boundaries]).toEqual([0, 0, 1]);
  expect([...shard.kinds]).toEqual([0, 3, 0]);
});

test("whitespace tokens carry no label and no boundary", () => {
  const record = makeRecord({
    id: "feat-2",
    text: "every Monday",
    spans: [
      { start: 0, end: 5, label: "RECUR", clauseStart: false },
      { start: 6, end: 12, label: "WEEKDAY", clauseStart: true },
    ],
  });

  const shard = featurizeRecords([record]);
  const whitespace = [...shard.kinds].indexOf(3);

  expect(whitespace).toBeGreaterThanOrEqual(0);
  expect(shard.labels[whitespace]).toBe(PADDING_LABEL);
  expect(shard.boundaries[whitespace]).toBe(0);
});

test("a fragment starting mid-token throws Unaligned supervision", () => {
  const record = makeRecord({
    id: "misaligned-1",
    text: "every Monday",
    spans: [
      { start: 1, end: 5, label: "RECUR", clauseStart: false },
      { start: 6, end: 12, label: "WEEKDAY", clauseStart: true },
    ],
  });

  expect(() => featurizeRecords([record])).toThrow(/Unaligned supervision in misaligned-1/);
});

test("labels.bin length equals rows.bin bytes / 2 / rowsPerToken", () => {
  const records = [
    makeRecord({
      id: "feat-3",
      text: "every Monday",
      spans: [
        { start: 0, end: 5, label: "RECUR", clauseStart: false },
        { start: 6, end: 12, label: "WEEKDAY", clauseStart: true },
      ],
    }),
    makeRecord({
      id: "feat-4",
      text: "every 2 hours",
      spans: [
        { start: 0, end: 5, label: "RECUR", clauseStart: false },
        { start: 6, end: 7, label: "INTERVAL", clauseStart: true },
        { start: 8, end: 13, label: "GLUE", clauseStart: false },
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

  expect(offsets.length - 1).toBe(summary.sequences);
  expect(offsets.at(-1)).toBe(summary.tokens);
  expect(summary.sequences).toBe(2);
});
