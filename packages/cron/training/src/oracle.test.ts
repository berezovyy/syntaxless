import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

import { decodeIntent } from "../../src/parser/decode.js";
import { resolveRepo } from "./evaluate";
import {
  emptyOracleReport,
  groupedOracleFailures,
  labeledTokensFromGold,
  loadGoldOracleRecords,
  loadKindSeedOracleRecords,
  oracleDecode,
  oracleMatches,
  type OracleRecord,
  type OracleReport,
  parseOracleArgs,
  recordOracleOutcome,
  reservoirSampleJsonl,
  summarizeOracleReport,
} from "./oracle";

const RESULTS = resolveRepo("packages/cron/training/results/oracle-red-test.json");

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

describe("labeledTokensFromGold", () => {
  test("feeds shipped decodeIntent rather than a copy", () => {
    const labeled = labeledTokensFromGold(hourly.text, hourly.spans);
    const shipped = decodeIntent(labeled, undefined, hourly.text, "off", "request");
    const outcome = oracleDecode(hourly);

    expect(outcome.ok).toBe(true);
    if (outcome.ok) expect(outcome.intent).toEqual(shipped.intent);
  });
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
  test("streams a sample without loading the whole file as one string", async () => {
    const dir = resolveRepo("packages/cron/training/results");

    mkdirSync(dir, { recursive: true });
    const path = join(dir, "oracle-sample-fixture.jsonl");
    const lines = Array.from({ length: 40 }, (_, index) => JSON.stringify({ id: `row-${index}` }));

    writeFileSync(path, `${lines.join("\n")}\n`);
    const sample = await reservoirSampleJsonl(path, 7, 1);

    expect(sample).toHaveLength(7);
    expect(new Set(sample).size).toBe(7);
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

describe("oracle round-trip suite", () => {
  let report: OracleReport;

  beforeAll(async () => {
    report = emptyOracleReport();
    for (const record of await loadGoldOracleRecords()) recordOracleOutcome(report, record);
    for (const record of loadKindSeedOracleRecords(3)) recordOracleOutcome(report, record);
    mkdirSync(dirname(RESULTS), { recursive: true });
    writeFileSync(RESULTS, `${JSON.stringify(summarizeOracleReport(report), null, 2)}\n`);
  }, 120_000);

  test("covers gold JSONL and KindDef template × 3 seeds", () => {
    expect(report.sources.gold.total).toBeGreaterThan(0);
    expect(report.sources["kind-seed"].total).toBeGreaterThan(0);
  });

  test("writes a red-list summary even while the suite is not 100%", () => {
    const summary = summarizeOracleReport(report);

    expect(summary.total).toBe(report.total);
    expect(summary.failureCount).toBe(report.failureCount);
  });

  test("oracle round-trip is 100% on gold JSONL and kind-seeds", () => {
    expect(report.failureCount).toBe(0);
  });
});

afterAll(() => {
  // Keep RESULTS on disk for the Phase 0 scratch copy.
});
