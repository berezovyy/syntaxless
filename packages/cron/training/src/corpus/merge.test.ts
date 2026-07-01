import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ScheduleIntent } from "../../../src/types/intent";
import type { CorpusRecord, CorpusSource } from "./merge";
import {
  mergeCorpus,
  mergeFiles,
  mergerSha256,
  parseArgs,
  readLlmRecords,
  readRuleRecords,
} from "./merge";
import { normalizeForKey } from "./normalize";

const workdir = mkdtempSync(join(tmpdir(), "syntaxless-merge-"));

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
  fingerprint: string;
  split?: CorpusRecord["split"];
  family?: CorpusRecord["family"];
  intent?: ScheduleIntent | null;
  spans?: CorpusRecord["spans"];
  source?: CorpusSource;
  template?: string;
  angleId?: string;
  model?: string;
  promptHash?: string;
  runId?: string;
}

function makeRecord(options: RecordOptions): CorpusRecord {
  return {
    id: options.id,
    split: options.split ?? "train",
    family: options.family ?? "weekday-list",
    fingerprint: options.fingerprint,
    text: options.text,
    intent: options.intent === undefined ? MONDAY : options.intent,
    spans: options.spans ?? [
      { start: 0, end: options.text.length, label: "O", clauseStart: false },
    ],
    source: options.source ?? "rule",
    ...(options.template === undefined ? {} : { template: options.template }),
    ...(options.angleId === undefined ? {} : { angleId: options.angleId }),
    ...(options.model === undefined ? {} : { model: options.model }),
    ...(options.promptHash === undefined ? {} : { promptHash: options.promptHash }),
    ...(options.runId === undefined ? {} : { runId: options.runId }),
  };
}

test("normalizeForKey folds case, whitespace, punctuation, and digit words", () => {
  expect(normalizeForKey("Every   Monday!")).toBe("every monday");
  expect(normalizeForKey("  EVERY monday,  ")).toBe("every monday");
  expect(normalizeForKey("every twenty one minutes")).toBe("every 21 minutes");
  expect(normalizeForKey("every 21 minutes")).toBe("every 21 minutes");
  expect(normalizeForKey("every twenty minutes")).toBe("every 20 minutes");
});

test("exact duplicates collapse to one, keeping the first by id", () => {
  const first = makeRecord({ id: "rule-1", text: "every Monday", fingerprint: "fp-a" });

  const second = makeRecord({
    id: "rule-2",
    text: "Every   monday!",
    fingerprint: "fp-b",
  });

  const merged = mergeCorpus([{ source: "rule", records: [second, first] }]);

  expect(merged.records).toHaveLength(1);
  expect(merged.records[0]!.id).toBe("rule-1");
  expect(merged.records[0]!.text).toBe("every Monday");
  expect(merged.dedup.exact.dropped).toBe(1);

  expect(merged.dedup.exact.samples[0]).toMatchObject({
    keptId: "rule-1",
    droppedId: "rule-2",
  });
});

test("two sources sharing a fingerprint dedupe, and rule wins", () => {
  const rule = makeRecord({
    id: "rule-1",
    text: "every Monday",
    fingerprint: "fp-shared",
    source: "rule",
  });

  const llm = makeRecord({
    id: "llm-1",
    text: "on Mondays",
    fingerprint: "fp-shared",
    source: "llm",
    angleId: "ops/weekday-list/terse",
  });

  const merged = mergeCorpus([
    { source: "llm", records: [llm] },
    { source: "rule", records: [rule] },
  ]);

  expect(merged.records).toHaveLength(1);
  expect(merged.records[0]!.source).toBe("rule");
  expect(merged.dedup.exact.dropped).toBe(0);
  expect(merged.dedup.structural.dropped).toBe(1);
});

test("a heldout fingerprint planted in train fails the merge", () => {
  const span = { start: 0, end: 5, label: "RECUR" as const, clauseStart: false };

  const heldout = makeRecord({
    id: "rule-heldout",
    split: "heldout",
    text: "the train leaves at five",
    fingerprint: "fp-heldout",
    spans: [span],
  });

  const train = makeRecord({
    id: "rule-train",
    split: "train",
    text: "an ordinary reminder",
    fingerprint: "fp-heldout",
    spans: [span],
  });

  expect(() => mergeCorpus([{ source: "rule", records: [train, heldout] }])).toThrow(
    /held-out fingerprint/,
  );
});

test("provenance survives the merge, additively", () => {
  const llm = makeRecord({
    id: "llm-1",
    text: "every Monday",
    fingerprint: "fp-llm",
    source: "llm",
    angleId: "ops/weekday-list/terse",
    model: "fixture-model",
    promptHash: "p".repeat(64),
    runId: "fixture-run",
  });

  const rule = makeRecord({
    id: "rule-1",
    text: "every Tuesday",
    fingerprint: "fp-rule",
    template: "weekday-list/basic",
  });

  const merged = mergeCorpus([
    { source: "rule", records: [rule] },
    { source: "llm", records: [llm] },
  ]);

  const byId = new Map(merged.records.map((record) => [record.id, record]));

  expect(byId.get("llm-1")).toMatchObject({
    source: "llm",
    family: "weekday-list",
    angleId: "ops/weekday-list/terse",
    model: "fixture-model",
    promptHash: "p".repeat(64),
    runId: "fixture-run",
  });

  expect(byId.get("rule-1")).toMatchObject({
    source: "rule",
    template: "weekday-list/basic",
  });
});

test("the output is byte-stable under input reordering", () => {
  const a = makeRecord({ id: "rule-a", text: "every Monday", fingerprint: "fp-a" });
  const b = makeRecord({ id: "rule-b", text: "every Tuesday", fingerprint: "fp-b" });
  const c = makeRecord({ id: "rule-c", text: "every Wednesday", fingerprint: "fp-c" });

  const first = mergeCorpus([{ source: "rule", records: [a, b, c] }]);
  const second = mergeCorpus([{ source: "rule", records: [c, a, b] }]);

  expect(JSON.stringify(first.records)).toBe(JSON.stringify(second.records));
  expect(first.dedup).toEqual(second.dedup);
});

test("source order does not affect the merged bytes", () => {
  const rule = makeRecord({ id: "rule-1", text: "every Monday", fingerprint: "fp-r" });

  const llm = makeRecord({
    id: "llm-1",
    text: "every Tuesday",
    fingerprint: "fp-l",
    source: "llm",
    angleId: "ops/weekday-list/terse",
  });

  const ruleFirst = mergeCorpus([
    { source: "rule", records: [rule] },
    { source: "llm", records: [llm] },
  ]);

  const llmFirst = mergeCorpus([
    { source: "llm", records: [llm] },
    { source: "rule", records: [rule] },
  ]);

  expect(JSON.stringify(ruleFirst.records)).toBe(JSON.stringify(llmFirst.records));
});

test("a gold record is never dropped for an llm duplicate", () => {
  const llm = makeRecord({
    id: "llm-1",
    text: "every monday",
    fingerprint: "fp-llm",
    source: "llm",
    angleId: "ops/weekday-list/terse",
  });

  const gold = makeRecord({
    id: "rule-9",
    text: "Every Monday!",
    fingerprint: "fp-gold",
    source: "rule",
  });

  const merged = mergeCorpus([
    { source: "llm", records: [llm] },
    { source: "rule", records: [gold] },
  ]);

  expect(merged.records).toHaveLength(1);

  expect(merged.records[0]).toMatchObject({
    id: "rule-9",
    source: "rule",
    text: "Every Monday!",
  });

  expect(merged.dedup.exact.samples[0]).toMatchObject({
    keptId: "rule-9",
    droppedId: "llm-1",
  });
});

test("a missing source and unknown fields are rejected loudly", () => {
  const rulePath = join(workdir, "rule-invalid.jsonl");

  writeFileSync(
    rulePath,
    `${JSON.stringify({
      id: "x",
      split: "train",
      family: "weekday-list",
      fingerprint: "f",
      text: "t",
      intent: null,
      spans: [],
    })}\n`,
  );

  expect(() => readRuleRecords(rulePath)).toThrow(/invalid rule record/);

  const llmPath = join(workdir, "llm-invalid.jsonl");

  writeFileSync(
    llmPath,
    `${JSON.stringify({
      text: "every Monday",
      intent: MONDAY,
      output: "schedule",
      spans: [{ start: 0, end: 12, label: "WEEKDAY", clauseStart: true }],
      fingerprint: "f",
      source: "llm",
      angleId: "ops/weekday-list/terse",
      model: "m",
      promptHash: "p",
      runId: "r",
      slots: [],
    })}\n`,
  );

  expect(() => readLlmRecords(llmPath)).toThrow(/invalid llm record/);
});

test("the manifest accounts for every input record", () => {
  const ruleRecords = [
    makeRecord({ id: "rule-1", text: "every Monday", fingerprint: "fp-1" }),
    makeRecord({ id: "rule-2", text: "every Monday!", fingerprint: "fp-2" }),
  ];

  const rulePath = join(workdir, "rule.jsonl");

  writeFileSync(rulePath, `${ruleRecords.map((record) => JSON.stringify(record)).join("\n")}\n`);

  const llmRecord = {
    text: "on Mondays",
    intent: MONDAY,
    output: "schedule" as const,
    spans: [{ start: 0, end: 12, label: "WEEKDAY" as const, clauseStart: true }],
    fingerprint: "fp-llm",
    source: "llm" as const,
    angleId: "ops/weekday-list/terse",
    model: "fixture-model",
    promptHash: "a".repeat(64),
    runId: "fixture-run",
  };

  const llmPath = join(workdir, "llm.jsonl");

  writeFileSync(llmPath, `${JSON.stringify(llmRecord)}\n`);

  const summary = mergeFiles({
    rule: [rulePath],
    llm: [llmPath],
    out: join(workdir, "merged"),
  });

  const manifest = JSON.parse(readFileSync(summary.files.manifest, "utf8")) as {
    inputRecords: number;
    mergedRecords: number;
    mergerSha256: string;
  };

  expect(manifest.inputRecords).toBe(3);
  expect(manifest.mergedRecords).toBe(2);
  expect(manifest.mergedRecords).toBe(summary.records);
  expect(manifest.mergerSha256).toBe(mergerSha256());
});

test("parseArgs accepts repeated sources and rejects an empty merge", () => {
  expect(parseArgs(["--rule", "a", "--rule", "b", "--llm", "c", "--out", "o"])).toEqual({
    rule: ["a", "b"],
    llm: ["c"],
    out: "o",
  });

  expect(() => parseArgs(["--out", "o"])).toThrow(/at least one/);
  expect(() => parseArgs(["--rule", "a"])).toThrow(/--out is required/);
});
