import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { coarseText } from "./fingerprint";
import { assertHumanIsolation, parseArgs, parseHumanCorpus } from "./human-evaluate";

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
      assertHumanIsolation(resolve("packages/cron/training/data/human.jsonl"), records),
    ).rejects.toThrow(/outside training\/data/);
  });
});

describe("human evaluation CLI", () => {
  test("defaults to development and validates partitions", () => {
    expect(parseArgs([]).partition).toBe("development");
    expect(() => parseArgs(["--partition", "mystery"])).toThrow(/invalid human partition/);
  });
});

describe("committed human-v1 corpus", () => {
  test("keeps partitions disjoint, supported, and generator-isolated", async () => {
    const path = resolve("packages/cron/evaluation/human-v1.jsonl");
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
