import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { structuralPattern } from "../src/fingerprint";
import type { CorpusRecord } from "../src/generate";
import { generate, generatorSha256 } from "../src/generate";

const workdir = mkdtempSync(join(tmpdir(), "syntaxless-generate-"));

afterAll(() => {
  rmSync(workdir, { recursive: true, force: true });
});

function frames(records: readonly CorpusRecord[]): Set<string> {
  return new Set(
    records
      .filter((record) => record.family !== "negative")
      .map((record) => JSON.stringify(structuralPattern(record.text, record.spans))),
  );
}

test("heldout and excluded train fingerprints are disjoint", () => {
  const heldout = generate({
    split: "heldout",
    count: 80,
    seed: 314159,
    out: join(workdir, "heldout.jsonl"),
  });

  const train = generate({
    split: "train",
    count: 400,
    seed: 271828,
    out: join(workdir, "train.jsonl"),
    exclude: heldout.fingerprintsPath,
  });

  const trainFingerprints = new Set(train.fingerprints);
  const overlap = heldout.fingerprints.filter((fingerprint) => trainFingerprints.has(fingerprint));

  expect(overlap).toEqual([]);

  // Leak-proof check: no training example shares a structural frame with a
  // held-out example, carriers and values erased.
  const heldoutFrames = frames(heldout.records);
  const leaked = [...frames(train.records)].filter((frame) => heldoutFrames.has(frame));

  expect(leaked).toEqual([]);
});

test("the heldout artifact holds only fingerprints and generatorSha256", () => {
  const heldout = generate({
    split: "heldout",
    count: 20,
    seed: 7,
    out: join(workdir, "artifact.jsonl"),
  });

  const artifact = JSON.parse(readFileSync(heldout.fingerprintsPath, "utf8")) as Record<
    string,
    unknown
  >;

  expect(Object.keys(artifact).sort()).toEqual(["fingerprints", "generatorSha256"]);
  expect(artifact.fingerprints).toEqual([...heldout.fingerprints].sort());
  expect(artifact.generatorSha256).toBe(generatorSha256());
});

test("the same seed and count produce identical text and fingerprints", () => {
  const options = {
    split: "train" as const,
    count: 150,
    seed: 123456,
    out: join(workdir, "det-a.jsonl"),
  };

  const first = generate(options);
  const second = generate({ ...options, out: join(workdir, "det-b.jsonl") });

  expect(first.records.map((record) => record.text)).toEqual(
    second.records.map((record) => record.text),
  );

  expect(first.fingerprints).toEqual(second.fingerprints);

  expect(first.records.map((record) => record.fingerprint)).toEqual(
    second.records.map((record) => record.fingerprint),
  );
});

test("an exclusion file actually removes rows that the same seeded draw would emit", () => {
  const options = {
    split: "train" as const,
    count: 40,
    seed: 1947,
    out: join(workdir, "exclude-control.jsonl"),
  };

  const control = generate(options);

  const filtered = generate({
    ...options,
    out: join(workdir, "exclude-filtered.jsonl"),
    exclude: control.fingerprintsPath,
  });

  expect(filtered.excluded).toBeGreaterThan(0);
  expect(filtered.records).toHaveLength(40);
  const excluded = new Set(control.fingerprints);

  expect(filtered.records.filter((row) => excluded.has(row.fingerprint))).toEqual([]);
});
