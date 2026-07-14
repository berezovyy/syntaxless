import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { normalize } from "../../src/parser/normalize";
import { writeBackward } from "../src/backward";
import { loadProtectedText, textKey } from "../src/corpus/production/protected-text";
import type { CorpusRecord } from "../src/generate";
import { bootstrapEasySentences } from "../src/mix/bootstrap";
import { classifyPopulation, SHARE_BAND, sharesInBand, sharesOf } from "../src/mix/quotas";
import type { MixReport } from "../src/mix/report";
import { loadEasySource } from "../src/mix/tatoeba";
import { humanEvalTexts } from "../src/slots/leak";
import { tempDir } from "./helpers/temp-dir";

function readJsonl(path: string): CorpusRecord[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as CorpusRecord);
}

test("bootstrap easy-no source has at least 2000 unique sentences", () => {
  const sentences = bootstrapEasySentences();

  expect(new Set(sentences).size).toBeGreaterThanOrEqual(2000);
});

test("writeBackward --limit 2000 still emits E in band (catalog order cannot starve none)", () => {
  const dir = tempDir("mix-quota-");
  const out = join(dir, "kinds.jsonl");

  writeBackward({ out, limit: 2000, split: "train", mix: true });
  const records = readJsonl(out);

  expect(records.length).toBe(2000);
  expect(records.every((record) => record.split === "train")).toBe(true);
  const counts = { R: 0, H: 0, E: 0 };

  for (const record of records) counts[classifyPopulation(record)] += 1;
  const shares = sharesOf(counts);

  expect(sharesInBand(shares)).toBe(true);
  expect(counts.E / records.length).toBeGreaterThanOrEqual(SHARE_BAND.E[0]);
  expect(counts.E).toBeGreaterThan(0);
  const report = JSON.parse(readFileSync(join(dir, "mix-report.json"), "utf8")) as MixReport;

  expect(report.inBand).toBe(true);
  expect(report.counts.E).toBe(counts.E);
});

test("mixed all-split fingerprints are disjoint across train/validation/heldout", () => {
  const dir = tempDir("mix-disjoint-");
  const out = join(dir, "kinds.jsonl");

  writeBackward({ out, limit: 2000, mix: true });
  const records = readJsonl(out);
  const texts = new Set<string>();
  const protectedText = loadProtectedText();

  const bySplit: Record<string, Set<string>> = {
    train: new Set(),
    validation: new Set(),
    heldout: new Set(),
  };

  for (const record of records) {
    bySplit[record.split]!.add(record.fingerprint);
    expect(record.text).toBe(normalize(record.text));
    expect(protectedText.matches(record.text)).toBe(false);
    // At this size E has not exhausted its pool, so every row must be unique.
    expect(texts.has(textKey(record.text))).toBe(false);
    texts.add(textKey(record.text));
  }

  for (const left of Object.keys(bySplit)) {
    for (const right of Object.keys(bySplit)) {
      if (left >= right) continue;

      for (const fingerprint of bySplit[left]!) {
        expect(bySplit[right]!.has(fingerprint)).toBe(false);
      }
    }
  }
});

test("a 25k train mix exhausts its E pool before repeating, with at least 2000 unique texts", () => {
  const dir = tempDir("mix-unique-e-");
  const out = join(dir, "kinds.jsonl");

  writeBackward({ out, limit: 25_000, split: "train", mix: true });
  const records = readJsonl(out);
  const easy = records.filter((record) => record.speechAct === "none" && record.intent === null);
  const unique = new Set(easy.map((record) => record.text));

  expect(easy.length / records.length).toBeGreaterThanOrEqual(SHARE_BAND.E[0]);
  expect(unique.size).toBeGreaterThanOrEqual(2000);
  const source = loadEasySource({ blocked: new Set(humanEvalTexts()) });
  const trainPoolSize = Math.floor(source.sentences.length * 0.75);

  expect(unique.size).toBe(Math.min(easy.length, trainPoolSize));
});
