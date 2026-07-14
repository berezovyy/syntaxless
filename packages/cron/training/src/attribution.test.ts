import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { tempDir } from "../tests/helpers/temp-dir";
import {
  addAttribution,
  attributeMiss,
  type AttributionBucket,
  type AttributionInput,
  emptyAttributionReport,
  goldViolatesSemantics,
  writeAttributionFile,
} from "./attribution";

const hourly = { frequency: { unit: "hour" as const, interval: 1 } };

const missed: AttributionInput = {
  text: "hourly",
  expectedIntent: hourly,
  actualIntent: null,
  accepted: false,
  goldSpans: [{ start: 0, end: 6, label: "UNIT_HOUR", clauseStart: true, actionable: true }],
  goldLabels: ["UNIT_HOUR"],
  predictedLabels: ["UNIT_HOUR"],
  goldSpeechAct: "request",
  predictedSpeechAct: "request",
  goldActionable: [true],
  predictedActionable: [true],
  goldOracle: { ok: true, intent: hourly },
  predictedOracle: { ok: true, intent: hourly },
};

test.each([
  {
    name: "matching heads and decodable intent were rejected",
    overrides: {},
    bucket: "CONF",
  },
  { name: "wrong role", overrides: { predictedLabels: ["O"] }, bucket: "ROLE" },
  {
    name: "missing token prediction",
    overrides: { predictedLabels: [] },
    bucket: "ROLE",
  },
  {
    name: "wrong speech act",
    overrides: { predictedSpeechAct: "descriptive" },
    bucket: "ACT",
  },
  {
    name: "wrong actionable bit",
    overrides: { predictedActionable: [false] },
    bucket: "ACT",
  },
  {
    name: "missing actionable prediction",
    overrides: { predictedActionable: [] },
    bucket: "ACT",
  },
  {
    name: "role disagreement takes priority over actionability",
    overrides: { predictedLabels: ["O"], predictedActionable: [false] },
    bucket: "ROLE",
  },
  {
    name: "unusable gold decoding takes priority over wrong predictions",
    overrides: {
      goldOracle: { ok: false, error: "unrecognized value" },
      predictedLabels: ["O"],
    },
    bucket: "DECODE",
  },
  {
    name: "accepted output changed meaning despite matching heads",
    overrides: {
      accepted: true,
      actualIntent: { frequency: { unit: "day", interval: 1 } },
    },
    bucket: "DECODE",
  },
] satisfies Array<{
  name: string;
  overrides: Partial<AttributionInput>;
  bucket: AttributionBucket;
}>)("attribution: $name", ({ overrides, bucket }) => {
  expect(attributeMiss({ ...missed, ...overrides })).toBe(bucket);
});

test("an unqualified 1:30 labeled as 13:30 is a gold error, not a model error", () => {
  const input: AttributionInput = {
    text: "1:30 daily",
    expectedIntent: {
      frequency: { unit: "day", interval: 1 },
      time: { hour: 13, minute: 30 },
    },
    actualIntent: null,
    accepted: false,
    goldSpans: [
      { start: 0, end: 1, label: "HOUR_LIT", clauseStart: true },
      { start: 1, end: 2, label: "GLUE", clauseStart: false },
      { start: 2, end: 4, label: "MINUTE_30", clauseStart: false },
      { start: 5, end: 10, label: "UNIT_DAY", clauseStart: false },
    ],
    goldLabels: ["HOUR_LIT", "GLUE", "MINUTE_30", "UNIT_DAY"],
    predictedLabels: ["O", "O", "O", "O"],
  };

  expect(goldViolatesSemantics(input)).toBe(true);
  expect(attributeMiss(input)).toBe("GOLD");

  expect(
    goldViolatesSemantics({
      ...input,
      expectedIntent: {
        frequency: { unit: "day", interval: 1 },
        time: { hour: 1, minute: 30 },
      },
    }),
  ).toBe(false);
});

test.each([
  {
    name: "accepted non-schedule",
    expectedIntent: null,
    actualIntent: hourly,
    accepted: true,
    bucket: "ACT",
  },
  {
    name: "refused request",
    expectedIntent: hourly,
    actualIntent: null,
    accepted: false,
    predictedSpeechAct: "question",
    bucket: "ACT",
  },
  {
    name: "decodable request rejected by confidence",
    expectedIntent: hourly,
    actualIntent: null,
    accepted: false,
    predictedOracle: { ok: true, intent: hourly },
    bucket: "CONF",
  },
  {
    name: "wrong schedule without gold token labels",
    expectedIntent: hourly,
    actualIntent: { frequency: { unit: "day", interval: 1 } },
    accepted: true,
    bucket: "ROLE",
  },
] satisfies Array<Omit<AttributionInput, "text"> & { name: string; bucket: AttributionBucket }>)(
  "unlabeled evaluation: $name",
  ({ name: _name, bucket, ...input }) => {
    expect(attributeMiss({ text: "evaluation example", ...input })).toBe(bucket);
  },
);

test("writing attribution replaces a rerun's source counts without losing other sources", async () => {
  const path = join(tempDir("syntaxless-attribution-"), "nested", "report.json");
  const first = emptyAttributionReport();

  addAttribution(first, { id: "a", source: "gold", bucket: "ROLE" });
  addAttribution(first, { id: "b", source: "gold", bucket: "ROLE" });
  await writeAttributionFile(path, "gold", first);
  const other = emptyAttributionReport();

  addAttribution(other, { id: "c", source: "human", bucket: "ACT" });
  await writeAttributionFile(path, "human", other);
  const rerun = emptyAttributionReport();

  addAttribution(rerun, { id: "d", source: "gold", bucket: "CONF" });
  await writeAttributionFile(path, "gold", rerun);

  expect(JSON.parse(readFileSync(path, "utf8"))).toEqual({
    ROLE: 0,
    ACT: 1,
    DECODE: 0,
    CONF: 1,
    GOLD: 0,
    sources: {
      gold: { ROLE: 0, ACT: 0, DECODE: 0, CONF: 1, GOLD: 0 },
      human: { ROLE: 0, ACT: 1, DECODE: 0, CONF: 0, GOLD: 0 },
    },
    records: [{ id: "d", source: "gold", bucket: "CONF" }],
  });
});
