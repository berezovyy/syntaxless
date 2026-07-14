import { describe, expect, test } from "bun:test";
import { writeFileSync } from "node:fs";
import { join } from "node:path";

import { labelId } from "../../src/model/labels";
import { tempDir } from "../tests/helpers/temp-dir";
import {
  emptyOracleReport,
  groupedOracleFailures,
  labeledTokensFromGold,
  loadGoldOracleRecords,
  loadKindSeedOracleRecords,
  oracleMatches,
  type OracleRecord,
  parseOracleArgs,
  recordOracleOutcome,
  reservoirSampleJsonl,
  summarizeOracleReport,
} from "./oracle";

const workdir = tempDir("syntaxless-oracle-");

const intervalGlue: OracleRecord = {
  id: "glue-drop",
  source: "gold",
  family: "interval",
  kind: "n-interval",
  template: "every-n",
  text: "every 5 minutes",
  spans: [
    { start: 0, end: 5, label: "RECUR", clauseStart: false, actionable: false },
    { start: 6, end: 7, label: "INTERVAL_5", clauseStart: true, actionable: true },
    { start: 8, end: 15, label: "UNIT_MIN", clauseStart: false, actionable: true },
  ],
  intent: { frequency: { unit: "minute", interval: 5 } },
  speechAct: "request",
};

const hourly: OracleRecord = {
  id: "hourly",
  source: "kind-seed",
  family: "interval",
  kind: "unit-interval",
  template: "adverb",
  text: "hourly",
  spans: [{ start: 0, end: 6, label: "UNIT_HOUR", clauseStart: true, actionable: true }],
  intent: { frequency: { unit: "hour", interval: 1 } },
  speechAct: "request",
};

const twelveHour: OracleRecord = {
  id: "clock-130",
  source: "gold",
  family: "time-of-day",
  kind: "time-of-day",
  template: "digits",
  text: "1:30 daily",
  spans: [
    { start: 0, end: 1, label: "HOUR_LIT", clauseStart: true, actionable: true },
    { start: 1, end: 2, label: "GLUE", clauseStart: false, actionable: false },
    { start: 2, end: 4, label: "MINUTE_30", clauseStart: false, actionable: true },
    { start: 5, end: 10, label: "UNIT_DAY", clauseStart: false, actionable: true },
  ],
  intent: {
    frequency: { unit: "day", interval: 1 },
    time: { hour: 13, minute: 30 },
  },
  speechAct: "request",
};

test("gold alignment preserves labels, clause boundaries, and actionability while skipping spaces", () => {
  const labeled = labeledTokensFromGold(intervalGlue.text, intervalGlue.spans);

  expect(
    labeled.map(({ token, role, boundary, actionable }) => ({
      text: token.text,
      role,
      boundary,
      actionable,
    })),
  ).toEqual([
    { text: "every", role: labelId.RECUR, boundary: false, actionable: false },
    { text: "5", role: labelId.INTERVAL_5, boundary: true, actionable: true },
    { text: "minutes", role: labelId.UNIT_MIN, boundary: false, actionable: true },
  ]);
});

describe("oracle misses", () => {
  test("structural bits false still round-trip when the clause is actionable", () => {
    expect(oracleMatches(intervalGlue)).toBe(true);
  });

  test("uninvertible 12h clock without meridiem is a miss", () => {
    expect(oracleMatches(twelveHour)).toBe(false);
  });

  test("groups that miss under kind/template", () => {
    const report = emptyOracleReport();

    recordOracleOutcome(report, twelveHour);
    const grouped = groupedOracleFailures(report);

    expect(grouped["time-of-day\tdigits"]?.count).toBe(1);
    expect(grouped["time-of-day\tdigits"]?.kind).toBe("time-of-day");
    expect(grouped["time-of-day\tdigits"]?.template).toBe("digits");
  });
});

describe("reservoirSampleJsonl", () => {
  test("sampling returns unique original rows and is reproducible for a seed", async () => {
    const path = join(workdir, "oracle-sample-fixture.jsonl");
    const lines = Array.from({ length: 40 }, (_, index) => JSON.stringify({ id: `row-${index}` }));

    writeFileSync(path, `${lines.join("\n")}\n`);
    const sample = await reservoirSampleJsonl(path, 7, 1);

    expect(sample).toHaveLength(7);
    expect(new Set(sample).size).toBe(7);
    expect(sample.every((row) => lines.includes(row))).toBe(true);
    expect(await reservoirSampleJsonl(path, 7, 1)).toEqual(sample);
    expect(await reservoirSampleJsonl(path, 100, 1)).toEqual(lines);
  });
});

describe("parseOracleArgs", () => {
  test("defaults kinds-sample to 0 so bun test does not stream 21G", () => {
    expect(parseOracleArgs([])).toEqual({
      seeds: 50,
      kindsSample: 0,
      out: "packages/cron/training/results/oracle-red.json",
      gate: false,
    });

    expect(parseOracleArgs(["--gate"]).kindsSample).toBe(200000);
    expect(parseOracleArgs(["--gate", "--kinds-sample", "0"]).kindsSample).toBe(0);

    expect(parseOracleArgs(["--gate", "--seeds", "2", "--kinds-sample", "10"])).toMatchObject({
      gate: true,
      kindsSample: 10,
      seeds: 2,
    });
  });
});

test("gold and seeded generator outputs round-trip without silently changing the schedule", async () => {
  const report = emptyOracleReport();

  for (const record of await loadGoldOracleRecords()) recordOracleOutcome(report, record);
  for (const record of loadKindSeedOracleRecords(3)) recordOracleOutcome(report, record);
  expect(report.sources.gold.total).toBeGreaterThan(0);
  expect(report.sources["kind-seed"].total).toBeGreaterThan(0);
  expect(groupedOracleFailures(report)).toEqual({});
  expect(report.failureCount).toBe(0);
}, 120_000);

test("oracle summaries count successes and mismatches by source and template", () => {
  const report = emptyOracleReport();

  for (const record of [hourly, intervalGlue, twelveHour]) recordOracleOutcome(report, record);

  expect(summarizeOracleReport(report)).toMatchObject({
    total: 3,
    correct: 2,
    failureCount: 1,
    sources: {
      gold: { total: 2, failures: 1 },
      "kind-seed": { total: 1, failures: 0 },
    },
    byKind: {
      "unit-interval": { total: 1, failures: 0 },
      "n-interval": { total: 1, failures: 0 },
      "time-of-day": { total: 1, failures: 1 },
    },
    byKindTemplate: {
      "time-of-day\tdigits": { kind: "time-of-day", template: "digits", count: 1 },
    },
    examples: [expect.objectContaining({ id: "clock-130" })],
  });
});
