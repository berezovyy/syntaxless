import { expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { compileExpectation } from "../src/compile-expectation.js";
import { resolveRepo, sha256Hex } from "../src/evaluate";
import { loadRegressionRecords, parseHumanCorpus } from "../src/human-evaluate";
import type { HumanEvaluationRecord } from "../src/human-evaluate.js";

const directory = resolveRepo("packages/cron/evaluation");

const corpora = readdirSync(directory)
  .filter((name) => /^human-.*\.jsonl$/.test(name))
  .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
  .map((name) => ({
    name,
    records: parseHumanCorpus(readFileSync(join(directory, name), "utf8")),
  }));

test("compiler corrections preserve exact historical fixture and consumption identities", () => {
  const manifest = JSON.parse(
    readFileSync(join(directory, "compiler-contract-v2.json"), "utf8"),
  ) as {
    sourceSha256: Record<string, string>;
    corrections: {
      id: string;
      recordSha256: string;
      originalExpected: HumanEvaluationRecord["expected"];
      expected: HumanEvaluationRecord["expected"];
    }[];
  };

  for (const [name, digest] of Object.entries(manifest.sourceSha256)) {
    expect(sha256Hex(readFileSync(join(directory, name), "utf8"))).toBe(digest);
  }

  const records = new Map(
    corpora.flatMap(({ records }) => records).map((row) => [sha256Hex(JSON.stringify(row)), row]),
  );

  expect(new Set(manifest.corrections.map((row) => row.recordSha256)).size).toBe(
    manifest.corrections.length,
  );

  for (const correction of manifest.corrections) {
    const row = records.get(correction.recordSha256);

    if (row === undefined || row.intent === null) {
      throw new Error("missing reviewed historical row");
    }

    const before = JSON.stringify(row);
    const { unit, interval } = row.intent.frequency;
    const base = unit === "minute" ? 60 : unit === "hour" ? 24 : undefined;

    expect(base).toBeDefined();
    expect(base! % interval).not.toBe(0);
    expect(row.id).toBe(correction.id);
    expect(row.expected).toEqual(correction.originalExpected);

    expect(compileExpectation(row)).toEqual({
      supported: false,
      code: "non-unit-interval",
    });

    expect(correction.expected).toEqual(compileExpectation(row));
    expect(JSON.stringify(row)).toBe(before);
    const newRow = { ...row, id: "new-unreviewed-row" };

    expect(compileExpectation(newRow)).toBe(newRow.expected);
  }
});

test(
  "regression loading includes all versioned " + "records and excludes calibration-only corpora",
  async () => {
    const expected = corpora
      .filter(({ name }) => /^human-v\d+\.jsonl$/.test(name))
      .flatMap(({ records }) => records);

    const loaded = await loadRegressionRecords();

    expect(loaded.records).toEqual(expected);
    expect(loaded.corpusSha256).toBe(sha256Hex(loaded.contents));
  },
);
