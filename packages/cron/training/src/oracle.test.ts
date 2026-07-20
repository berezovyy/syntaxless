import { expect, test } from "bun:test";

import { labelId } from "../../src/model/labels";
import {
  emptyOracleReport,
  labeledTokensFromGold,
  type OracleRecord,
  recordOracleOutcome,
  summarizeOracleReport,
} from "./oracle";

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
