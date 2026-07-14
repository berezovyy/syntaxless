import { expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { compile } from "../../src/compiler/cron";
import { corpusRecordSchema } from "../src/corpus/merge";
import { normalizeForKey } from "../src/corpus/normalize";
import { resolveRepo, sha256Hex } from "../src/evaluate";
import {
  assertAcceptanceFresh,
  assertHumanIsolation,
  loadRegressionRecords,
  parseHumanCorpus,
} from "../src/human-evaluate";

const directory = resolveRepo("packages/cron/evaluation");

const corpora = readdirSync(directory)
  .filter((name) => /^human-.*\.jsonl$/.test(name))
  .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
  .map((name) => ({
    name,
    records: parseHumanCorpus(readFileSync(join(directory, name), "utf8")),
  }));

test("evaluation corpus discovery cannot silently skip all fixtures", () => {
  expect(corpora.length).toBeGreaterThan(0);
});

test.each(corpora)(
  "$name remains isolated and its gold compiler expectations are valid",
  async ({ name, records }) => {
    await assertHumanIsolation(join(directory, name), records);
    const seen = new Set<string>();

    for (const record of records) {
      expect(record.trainingEligible, record.id).toBe(false);
      expect(corpusRecordSchema.safeParse(record).success, record.id).toBe(false);
      const key = normalizeForKey(record.text);

      expect(seen.has(key), `${name}: duplicate wording ${record.id}`).toBe(false);
      seen.add(key);

      if (record.intent === null) {
        expect(record.expected, record.id).toMatchObject({ parse: "reject" });
      } else {
        const actual = compile(record.intent);

        expect(actual, record.id).toMatchObject(record.expected);
      }
    }
  },
);

// Versions 1–15 have already been consumed. New acceptance corpora must remain
// fresh until their one permitted run; discovery must not force consumption.
test.each(Array.from({ length: 15 }, (_, index) => `human-v${index + 1}.jsonl`))(
  "%s cannot be reused as fresh acceptance under a different filename",
  async (name) => {
    const corpus = corpora.find((corpus) => corpus.name === name);

    if (!corpus) throw new Error(`Missing consumed regression corpus: ${name}`);
    const { records } = corpus;
    const acceptance = records.filter((row) => row.partition === "acceptance");
    const support = new Map<string, number>();

    for (const row of acceptance) support.set(row.family, (support.get(row.family) ?? 0) + 1);

    for (const [family, count] of support) {
      expect(count, `${family} acceptance support`).toBeGreaterThanOrEqual(5);
    }

    const digest = sha256Hex(`${acceptance.map((row) => JSON.stringify(row)).join("\n")}\n`);

    await expect(assertAcceptanceFresh("acceptance", digest)).rejects.toThrow(
      /already been consumed/,
    );
  },
);

// The post-v10 corpora were collected as new wording, unlike legacy revisions.
// Preserve that independence without copying every corpus's row/family counts.
test.each(
  corpora.filter(
    ({ name }) => /^human-v\d+\.jsonl$/.test(name) && Number(name.match(/\d+/)![0]) >= 11,
  ),
)(
  "$name has no normalized wording from earlier evaluation or calibration corpora",
  ({ name, records }) => {
    const version = Number(name.match(/\d+/)![0]);

    const prior = new Set(
      corpora
        .filter(
          (corpus) =>
            corpus.name.includes("calibration") || Number(corpus.name.match(/\d+/)![0]) < version,
        )
        .flatMap((corpus) => corpus.records.map((row) => normalizeForKey(row.text))),
    );

    expect(records.filter((row) => prior.has(normalizeForKey(row.text)))).toEqual([]);
  },
);

test("regression loading includes all versioned records and excludes calibration-only corpora", async () => {
  const expected = corpora
    .filter(({ name }) => /^human-v\d+\.jsonl$/.test(name))
    .flatMap(({ records }) => records);

  const loaded = await loadRegressionRecords();

  expect(loaded.records).toEqual(expected);
  expect(loaded.corpusSha256).toBe(sha256Hex(loaded.contents));
});
