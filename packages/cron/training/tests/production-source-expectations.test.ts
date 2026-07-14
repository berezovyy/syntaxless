import { expect, test } from "bun:test";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { CanonicalGzipWriter } from "../src/corpus/jsonl-io";
import { publicCorpusRecord } from "../src/corpus/production/public-source";
import { hashArtifact } from "../src/corpus/production/publication";
import {
  productionFrameTarget,
  sourceExpectations,
} from "../src/corpus/production/source-expectations";
import { featurizeKindsStreaming } from "../src/featurize-kinds";
import { KINDS } from "../src/kinds";
import { tempDir } from "./helpers/temp-dir";

test("source-derived expectations cover every binary field and split, independently of the shard writer", async () => {
  const directory = tempDir("cron-source-expectations-");

  const input = join(directory, "corpus.jsonl");
  const prefix = join(directory, "source");

  const splits = ["train", "validation", "heldout"] as const;

  const rows = splits.map((split, n) =>
    publicCorpusRecord(
      { sourceId: "fixture", license: "CC-BY-2.0-FR" },
      {
        id: n + 1,
        text: "several bronze gears filled the crate",
        owner: null,
        group_id: split,
        split,
      },
    ),
  );

  writeFileSync(input, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  const seen: string[] = [];
  const expected = await sourceExpectations(input, (row) => seen.push(row.id));

  expect(seen).toEqual(rows.map((r) => r.id));
  expect(expected.corpus).toEqual({ records: 3, sha256: await hashArtifact(input) });
  await featurizeKindsStreaming(input, prefix);

  for (const split of splits) {
    expect(expected.splits[split]!.sequences).toBe(1);
    expect(expected.splits[split]!.frameNames).toEqual(["easy-no"]);
    const manifest = JSON.parse(readFileSync(`${prefix}-${split}.json`, "utf8"));

    expect(expected.splits[split]!.frameCounts).toEqual(manifest.frameCounts);
    expect(expected.splits[split]!.labelCounts).toEqual(manifest.labelCounts);
    expect(expected.splits[split]!.actionableCounts).toEqual(manifest.actionableCounts);
    expect(expected.splits[split]!.populationCounts).toEqual(manifest.populationCounts);

    for (const [field, value] of Object.entries(expected.splits[split]!.fields)) {
      const path = `${prefix}-${split}${manifest.format[field].suffix}`;

      expect({
        bytes: readFileSync(path).length,
        sha256: await hashArtifact(path),
      }).toEqual(value);
    }

    expect(Object.keys(expected.splits[split]!.fields)).toHaveLength(10);
  }
});

test("production frame taxonomy covers all systematic kinds and rejects unknown names instead of silently defaulting to weekly", () => {
  for (const kind of KINDS.filter((k) => k.intents.some((i) => i !== null))) {
    expect(productionFrameTarget(kind.id)).toBeGreaterThan(0);
  }

  expect(productionFrameTarget("easy-no")).toBe(0);
  expect(productionFrameTarget("daily-clock")).toBe(2);
  expect(productionFrameTarget("inclusive-end-bound")).toBe(6);
  expect(productionFrameTarget("correction-interval")).toBe(1);

  expect(() => productionFrameTarget("unregistered-new-family")).toThrow(
    "Unknown production frame taxonomy",
  );
});

test("gzip is a deterministic canonical corpus artifact consumed without an uncompressed copy", async () => {
  const directory = tempDir("cron-source-gzip-");
  const plain = join(directory, "corpus.jsonl");
  const first = join(directory, "corpus-a.jsonl.gz");
  const second = join(directory, "corpus-b.jsonl.gz");

  const rows = (["train", "validation", "heldout"] as const).map((split, index) =>
    publicCorpusRecord(
      { sourceId: "gzip-fixture", license: "CC-BY-2.0-FR" },
      {
        id: index + 1,
        text: `several bronze gears filled crate ${index}`,
        owner: null,
        group_id: split,
        split,
      },
    ),
  );

  const payload = rows.map((row) => `${JSON.stringify(row)}\n`);

  writeFileSync(plain, payload.join(""));

  for (const path of [first, second]) {
    const writer = new CanonicalGzipWriter(path);

    for (const line of payload) await writer.append(line);
    await writer.close();
  }

  expect(readFileSync(first)).toEqual(readFileSync(second));

  const plainExpected = await sourceExpectations(plain);
  const gzipExpected = await sourceExpectations(first);

  expect(gzipExpected.splits).toEqual(plainExpected.splits);

  expect(gzipExpected.corpus).toEqual({
    records: rows.length,
    sha256: await hashArtifact(first),
  });

  expect(gzipExpected.corpus.sha256).not.toBe(plainExpected.corpus.sha256);

  const plainPrefix = join(directory, "plain");
  const gzipPrefix = join(directory, "gzip");

  await featurizeKindsStreaming(plain, plainPrefix);
  await featurizeKindsStreaming(first, gzipPrefix);

  for (const split of ["train", "validation", "heldout"] as const) {
    const manifest = JSON.parse(readFileSync(`${plainPrefix}-${split}.json`, "utf8"));

    expect(readFileSync(`${gzipPrefix}-${split}.json`)).toEqual(
      readFileSync(`${plainPrefix}-${split}.json`),
    );

    for (const field of Object.values(manifest.format) as { suffix: string }[]) {
      expect(readFileSync(`${gzipPrefix}-${split}${field.suffix}`)).toEqual(
        readFileSync(`${plainPrefix}-${split}${field.suffix}`),
      );
    }
  }

  const truncated = join(directory, "truncated.jsonl.gz");
  const compressed = readFileSync(first);

  writeFileSync(truncated, compressed.subarray(0, compressed.length - 4));
  expect(sourceExpectations(truncated)).rejects.toThrow();
});
