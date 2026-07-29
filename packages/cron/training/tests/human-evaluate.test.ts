import { describe, expect, test } from "bun:test";

import { resolveRepo } from "../src/evaluate";
import {
  assertAcceptanceFresh,
  assertHumanIsolation,
  decideHumanPromotion,
  parseHumanCorpus,
  selectCalibrationThreshold,
} from "../src/human-evaluate";
import type { HumanEvaluationRecord, HumanRates } from "../src/human-evaluate";

const valid = {
  version: 1,
  id: "dev-example",
  partition: "development",
  family: "interval",
  capabilities: ["conversational"],
  critical: false,
  trainingEligible: false,
  text: "Please refresh this every eleven minutes.",
  intent: { frequency: { unit: "minute", interval: 11 } },
  expected: { supported: true, expression: "*/11 * * * *" },
} satisfies HumanEvaluationRecord;

describe("acceptance consumption ledger", () => {
  test("blocks a consumed corpus hash even when the file path changes", async () => {
    await expect(
      assertAcceptanceFresh(
        "acceptance",
        "d6a589e7b626e9689b2341ac19d7668263b53fd67746d6291c743ea0253d2e81",
      ),
    ).rejects.toThrow(/already been consumed/);
  });

  test("does not block development or a genuinely new acceptance hash", async () => {
    const fresh = "0".repeat(64);

    await expect(assertAcceptanceFresh("development", fresh)).resolves.toBeUndefined();
    await expect(assertAcceptanceFresh("acceptance", fresh)).resolves.toBeUndefined();
  });
});

describe("parseHumanCorpus", () => {
  test("rejects fields that make a row training-shaped", () => {
    expect(() => parseHumanCorpus(`${JSON.stringify({ ...valid, spans: [] })}\n`)).toThrow(
      /training field spans/,
    );
  });

  test("requires an explicit non-training marker", () => {
    expect(() =>
      parseHumanCorpus(`${JSON.stringify({ ...valid, trainingEligible: true })}\n`),
    ).toThrow(/invalid schema/);
  });
});

describe("assertHumanIsolation", () => {
  test("fails closed for any corpus under training data", async () => {
    const records = parseHumanCorpus(`${JSON.stringify(valid)}\n`);

    await expect(
      assertHumanIsolation(resolveRepo("packages/cron/training/data/human.jsonl"), records),
    ).rejects.toThrow(/outside training\/data/);
  });
});

describe("human evaluation policy", () => {
  test("calibration reports bar failures but is never promotion-eligible", () => {
    const passing: HumanRates = {
      accuracy: 1,
      silentMeaningChangeRate: 0,
      negativeFalseAcceptanceRate: 0,
      byFamily: { interval: { total: 10, correct: 10 } },
      criticalFailures: [],
    };

    expect(decideHumanPromotion("calibration", passing)).toEqual({
      eligible: false,
      passed: false,
      failures: [],
    });

    expect(decideHumanPromotion("calibration", { ...passing, accuracy: 0.5 }).failures).toEqual([
      "overall accuracy below 0.95",
    ]);
  });

  test("regression is never promotion-eligible", () => {
    const passing: HumanRates = {
      accuracy: 1,
      silentMeaningChangeRate: 0,
      negativeFalseAcceptanceRate: 0,
      byFamily: { interval: { total: 10, correct: 10 } },
      criticalFailures: [],
    };

    const decision = decideHumanPromotion("regression", passing);

    expect(decision.eligible).toBe(false);
    expect(decision.passed).toBe(false);
  });

  test("selects maximum coverage that satisfies both safety limits", () => {
    const base = {
      total: 100,
      acceptedWrong: 0,
      silentMeaningChangeRate: 0,
    };

    const selected = selectCalibrationThreshold([
      {
        ...base,
        threshold: 0.4,
        accepted: 90,
        coverage: 0.9,
        risk: 0.01,
        negativeFalseAcceptanceRate: 0,
      },
      {
        ...base,
        threshold: 0.5,
        accepted: 80,
        coverage: 0.8,
        risk: 0,
        negativeFalseAcceptanceRate: 0,
      },
      {
        ...base,
        threshold: 0.6,
        accepted: 70,
        coverage: 0.7,
        risk: 0,
        negativeFalseAcceptanceRate: 0,
      },
    ]);

    expect(selected?.threshold).toBe(0.5);
  });
});

const passing: HumanRates = {
  accuracy: 1,
  silentMeaningChangeRate: 0,
  negativeFalseAcceptanceRate: 0,
  byFamily: { interval: { total: 100, correct: 100 } },
  criticalFailures: [],
};

test.each([
  { name: "overall accuracy at floor", rates: { accuracy: 0.95 }, passed: true },
  { name: "overall accuracy below floor", rates: { accuracy: 0.949 }, passed: false },
  {
    name: "family accuracy at floor",
    rates: { byFamily: { interval: { total: 100, correct: 90 } } },
    passed: true,
  },
  {
    name: "family accuracy below floor",
    rates: { byFamily: { interval: { total: 100, correct: 89 } } },
    passed: false,
  },
  {
    name: "silent changes at limit",
    rates: { silentMeaningChangeRate: 0.005 },
    passed: true,
  },
  {
    name: "silent changes over limit",
    rates: { silentMeaningChangeRate: 0.00501 },
    passed: false,
  },
  {
    name: "false accepts at limit",
    rates: { negativeFalseAcceptanceRate: 0.01 },
    passed: true,
  },
  {
    name: "false accepts over limit",
    rates: { negativeFalseAcceptanceRate: 0.01001 },
    passed: false,
  },
  {
    name: "critical failure despite high averages",
    rates: { criticalFailures: ["wrong-weekday"] },
    passed: false,
  },
])("promotion: $name", ({ rates, passed }) => {
  const scores = { ...passing, ...rates };

  expect(decideHumanPromotion("acceptance", scores).passed).toBe(passed);
});
