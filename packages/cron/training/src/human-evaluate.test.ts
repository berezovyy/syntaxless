import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";

import { resolveRepo } from "./evaluate";
import { coarseText } from "./fingerprint";
import {
  assertAcceptanceFresh,
  assertHumanIsolation,
  parseArgs,
  parseHumanCorpus,
  selectCalibrationThreshold,
} from "./human-evaluate";

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
} as const;

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
  test("accepts the isolated evaluation schema", () => {
    expect(parseHumanCorpus(`${JSON.stringify(valid)}\n`)).toHaveLength(1);
  });

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

  test("rejects duplicate ids", () => {
    expect(() => parseHumanCorpus(`${JSON.stringify(valid)}\n${JSON.stringify(valid)}\n`)).toThrow(
      /duplicate human id/,
    );
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

describe("human evaluation CLI", () => {
  test("defaults to development and validates partitions", () => {
    expect(parseArgs([]).partition).toBe("development");
    expect(parseArgs(["--weights", "/tmp/candidate.ts"]).weights).toBe("/tmp/candidate.ts");
    expect(() => parseArgs(["--partition", "mystery"])).toThrow(/invalid human partition/);
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

describe("committed human-v1 corpus", () => {
  test("keeps partitions disjoint, supported, and generator-isolated", async () => {
    const path = resolveRepo("packages/cron/evaluation/human-v1.jsonl");
    const records = parseHumanCorpus(await readFile(path, "utf8"));

    await expect(assertHumanIsolation(path, records)).resolves.toEqual({
      checked: true,
      outsideTrainingData: true,
      incompatibleWithTrainingSchema: true,
      absentFromGeneratorSources: true,
    });

    const seen = new Set<string>();

    for (const record of records) {
      const fingerprint = coarseText(record.text);

      expect(seen.has(fingerprint)).toBe(false);
      seen.add(fingerprint);
    }

    for (const partition of ["development", "calibration", "acceptance"] as const) {
      expect(records.some((record) => record.partition === partition)).toBe(true);
    }

    const acceptance = records.filter((record) => record.partition === "acceptance");
    const familyCounts = new Map<string, number>();

    for (const record of acceptance) {
      familyCounts.set(record.family, (familyCounts.get(record.family) ?? 0) + 1);
    }

    for (const support of familyCounts.values()) expect(support).toBeGreaterThanOrEqual(5);

    for (const capability of [
      "long-message",
      "correction",
      "ambiguity",
      "question",
      "ordinary-statement",
      "multiple-schedules",
      "quoted-schedule",
      "typo",
    ]) {
      expect(acceptance.some((record) => record.capabilities.includes(capability))).toBe(true);
    }
  });
});

describe("consumed human-v2 behavioral corpus", () => {
  test("remains isolated, broad, and permanently unavailable as fresh acceptance", async () => {
    const path = resolveRepo("packages/cron/evaluation/human-v2.jsonl");
    const records = parseHumanCorpus(await readFile(path, "utf8"));

    expect(records).toHaveLength(150);
    expect(records.every((record) => record.partition === "acceptance")).toBe(true);

    await expect(assertHumanIsolation(path, records)).resolves.toEqual({
      checked: true,
      outsideTrainingData: true,
      incompatibleWithTrainingSchema: true,
      absentFromGeneratorSources: true,
    });

    await expect(
      assertAcceptanceFresh(
        "acceptance",
        "ed3914bb0cd6fbf591e0c49fcd22a67afdf9d04c1c65936d1258b2832efe2ad8",
      ),
    ).rejects.toThrow(/already been consumed/);

    const fingerprints = new Set(records.map((record) => coarseText(record.text)));

    expect(fingerprints.size).toBe(records.length);
    const familyCounts = new Map<string, number>();

    for (const record of records) {
      familyCounts.set(record.family, (familyCounts.get(record.family) ?? 0) + 1);
    }

    expect(familyCounts.size).toBeGreaterThanOrEqual(15);
    for (const support of familyCounts.values()) expect(support).toBeGreaterThanOrEqual(5);
  });
});

describe("consumed human-v3 behavioral corpus", () => {
  test("remains isolated and permanently unavailable as fresh acceptance", async () => {
    const path = resolveRepo("packages/cron/evaluation/human-v3.jsonl");
    const records = parseHumanCorpus(await readFile(path, "utf8"));

    expect(records).toHaveLength(100);
    expect(records.every((record) => record.partition === "acceptance")).toBe(true);

    await expect(assertHumanIsolation(path, records)).resolves.toEqual({
      checked: true,
      outsideTrainingData: true,
      incompatibleWithTrainingSchema: true,
      absentFromGeneratorSources: true,
    });

    await expect(
      assertAcceptanceFresh(
        "acceptance",
        "b1247ed94271c743ab1f633a3d0c670872b65f44426412f09f42b284eddae93b",
      ),
    ).rejects.toThrow(/already been consumed/);
  });
});

describe("consumed human-v4 behavioral corpus", () => {
  test("remains isolated and permanently unavailable as fresh acceptance", async () => {
    const path = resolveRepo("packages/cron/evaluation/human-v4.jsonl");
    const records = parseHumanCorpus(await readFile(path, "utf8"));

    expect(records).toHaveLength(100);
    expect(records.every((record) => record.partition === "acceptance")).toBe(true);

    await expect(assertHumanIsolation(path, records)).resolves.toEqual({
      checked: true,
      outsideTrainingData: true,
      incompatibleWithTrainingSchema: true,
      absentFromGeneratorSources: true,
    });

    await expect(
      assertAcceptanceFresh(
        "acceptance",
        "5af1c32e7ff713b3a090f61fbf5abd7ed5cc16622953706a3423e2f0a3ef6a23",
      ),
    ).rejects.toThrow(/already been consumed/);
  });
});

describe("consumed human-v5 behavioral corpus", () => {
  test("remains isolated and permanently unavailable as fresh acceptance", async () => {
    const path = resolveRepo("packages/cron/evaluation/human-v5.jsonl");
    const records = parseHumanCorpus(await readFile(path, "utf8"));

    expect(records).toHaveLength(100);
    expect(records.every((record) => record.partition === "acceptance")).toBe(true);

    await expect(assertHumanIsolation(path, records)).resolves.toEqual({
      checked: true,
      outsideTrainingData: true,
      incompatibleWithTrainingSchema: true,
      absentFromGeneratorSources: true,
    });

    await expect(
      assertAcceptanceFresh(
        "acceptance",
        "989896ccaa1191fa95532281569a562cde630e7317c3a2765a8a0b3820bfd07a",
      ),
    ).rejects.toThrow(/already been consumed/);
  });
});
