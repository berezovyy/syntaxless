import { describe, expect, test } from "bun:test";

import { tokenize } from "../../src/model/tokenizer";
import type { EvaluationResult, GoldRecord } from "./evaluate";
import { canonicalizeIntent, lintGoldRecords, parseCorpus, riskCoverage } from "./evaluate";

describe("canonicalizeIntent", () => {
  test("normalizes semantic sets without mutating the input", () => {
    const intent = {
      frequency: { unit: "week" as const, interval: 1 },
      weekdays: ["friday" as const, "monday" as const],
      exceptions: {
        weekdays: ["sunday" as const, "tuesday" as const],
        dates: ["2026-12-02", "2026-01-01", "2026-12-02"],
      },
    };

    expect(canonicalizeIntent(intent)).toEqual({
      frequency: { unit: "week", interval: 1 },
      weekdays: ["monday", "friday"],
      exceptions: {
        weekdays: ["tuesday", "sunday"],
        dates: ["2026-01-01", "2026-12-02"],
      },
    });

    expect(intent.weekdays).toEqual(["friday", "monday"]);
  });
});

describe("parseCorpus does not patch gold semantics", () => {
  test("leaves 1:30 as 1:30 without inserting a meridiem", () => {
    const item: GoldRecord = {
      id: "clock",
      family: "time-of-day",
      text: "remind me 1:30 daily",
      spans: [
        { start: 0, end: 9, label: "O", clauseStart: false },
        { start: 10, end: 11, label: "HOUR", clauseStart: true },
        { start: 11, end: 12, label: "GLUE", clauseStart: false },
        { start: 12, end: 14, label: "MINUTE", clauseStart: false },
        { start: 15, end: 20, label: "RECUR", clauseStart: false },
      ],
      intent: {
        frequency: { unit: "day", interval: 1 },
        time: { hour: 13, minute: 30 },
      },
      expected: { supported: true, expression: "30 13 * * *" },
    };

    const [parsed] = parseCorpus(`${JSON.stringify(item)}\n`);

    expect(parsed?.text).toBe("remind me 1:30 daily");
    expect(parsed?.spans.some((span) => span.label === "MERIDIEM")).toBe(false);
    expect(parsed?.intent?.time).toEqual({ hour: 13, minute: 30 });

    for (const token of tokenize(item.text).filter((value) => value.kind !== 3)) {
      expect(item.spans.some((span) => token.start >= span.start && token.end <= span.end)).toBe(
        true,
      );
    }
  });

  test("does not rewrite a through date into an exclusive end", () => {
    const item: GoldRecord = {
      id: "bounds",
      family: "bounds",
      text: "from 2026-01-01 through 2026-01-31, every day",
      spans: [],
      intent: {
        frequency: { unit: "day", interval: 1 },
        bounds: { start: "2026-01-01", end: "2026-01-31" },
      },
      expected: { supported: false, expression: null },
    };

    const [parsed] = parseCorpus(`${JSON.stringify(item)}\n`);

    expect(parsed?.intent?.bounds).toEqual({
      start: "2026-01-01",
      end: "2026-01-31",
    });
  });
});

describe("lintGoldRecords", () => {
  test("rejects duplicate ids and overlapping spans", () => {
    const record: GoldRecord = {
      id: "same",
      family: "negative",
      text: "ordinary prose",
      spans: [{ start: 0, end: 8, label: "O", clauseStart: false }],
      intent: null,
      expected: { supported: false, expression: null },
    };

    expect(() => lintGoldRecords([record, record])).toThrow(/duplicate/);

    expect(() =>
      lintGoldRecords([
        {
          ...record,
          spans: [
            { start: 0, end: 8, label: "O", clauseStart: false },
            { start: 7, end: 13, label: "O", clauseStart: false },
          ],
        },
      ]),
    ).toThrow(/span/);
  });
});

describe("riskCoverage", () => {
  test("reports acceptance risk, silent changes, and negative false accepts", () => {
    const [point] = riskCoverage(
      [
        { accepted: true, correct: true, confidence: 0.9, negative: false },
        { accepted: true, correct: false, confidence: 0.8, negative: false },
        { accepted: true, correct: false, confidence: 0.7, negative: true },
        { accepted: false, correct: true, confidence: 0.1, negative: true },
      ],
      [0.5],
    );

    expect(point).toEqual({
      threshold: 0.5,
      total: 4,
      accepted: 3,
      acceptedWrong: 2,
      coverage: 0.75,
      risk: 2 / 3,
      silentMeaningChangeRate: 0.5,
      negativeFalseAcceptanceRate: 0.5,
    });
  });
});

describe("overlay-off report column", () => {
  test("is a sibling of overlay-on contracts and never a fifth summed metric", () => {
    const metrics: Array<EvaluationResult["contracts"][number]["metric"]> = [
      "tokenRoles",
      "rawDecode",
      "exactAst",
      "compiledExpression",
    ];

    expect(metrics).toHaveLength(4);

    const overlayOff = {
      contracts: [],
      selectivePrediction: [],
      rawSelectivePrediction: [],
      typoInvariance: {
        total: 0,
        cleanCorrect: 0,
        noisyCorrect: 0,
        pairedCorrect: 0,
        noisySilentWrong: 0,
        noisyAccuracy: 0,
        pairedAccuracy: 0,
        byFamily: {},
        byNoise: {},
        failureCounts: {},
        failures: [],
      },
    };

    const result: EvaluationResult = {
      artifactSha256: "0".repeat(64),
      corpus: "unused.jsonl",
      corpusSha256: "0".repeat(64),
      familyDistribution: {},
      scope: "unit",
      contracts: [],
      selectivePrediction: [],
      rawSelectivePrediction: [],
      typoInvariance: {
        total: 0,
        cleanCorrect: 0,
        noisyCorrect: 0,
        pairedCorrect: 0,
        noisySilentWrong: 0,
        noisyAccuracy: 0,
        pairedAccuracy: 0,
        byFamily: {},
        byNoise: {},
        failureCounts: {},
        failures: [],
      },
      overlayOff,
      attribution: {
        ROLE: 0,
        ACT: 0,
        DECODE: 0,
        CONF: 0,
        GOLD: 0,
        records: [],
      },
    };

    expect(result.contracts).toEqual([]);
    expect(result.overlayOff).toBe(overlayOff);
  });
});
