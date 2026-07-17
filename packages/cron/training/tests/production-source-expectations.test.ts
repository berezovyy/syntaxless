import { expect, test } from "bun:test";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type { CorpusRecord, Split } from "../src/corpus/merge";
import { hashArtifact } from "../src/corpus/production/publication";
import { sourceExpectations } from "../src/corpus/production/source-expectations";
import { featurizeKindsStreaming } from "../src/featurize-kinds";
import { sha256Hex } from "../src/fingerprint";
import { tempDir } from "./helpers/temp-dir";

function publicRecord(sourceId: string, split: Split, id: number, text: string): CorpusRecord {
  return {
    id: `${sourceId}/${id}`,
    split,
    family: "negative",
    text,
    spans: [
      {
        start: 0,
        end: text.length,
        label: "O",
        actionable: false,
        clauseStart: false,
      },
    ],
    intent: null,
    speechAct: "none",
    source: "rule",
    template: "easy-no/tatoeba-v2",
    fingerprint: sha256Hex(`negative\0${text}`),
    provenance: {
      authorship: "public-source",
      sourceId,
      sourceRecordId: String(id),
      groupId: split,
      constructionId: "ordinary-english",
      license: "CC-BY-2.0-FR",
      attribution: `Tatoeba contributors; no listed owner; https://tatoeba.org/en/sentences/show/${id}; runtime-normalized`,
    },
  };
}

test("source-derived expectations cover every binary field and split, independently of the shard writer", async () => {
  const directory = tempDir("cron-source-expectations-");

  const input = join(directory, "corpus.jsonl");
  const prefix = join(directory, "source");

  const splits = ["train", "validation", "heldout"] as const;

  const rows = splits.map((split, n) =>
    publicRecord("fixture", split, n + 1, "several bronze gears filled the crate"),
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
