import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { readJsonl, writeJsonl } from "../../../shared/training/jsonl.js";
import { reservedHashes } from "../../../shared/training/reserved.js";
import { generateCompact, writeCompact } from "../src/compact.js";
import { featurizeFiles } from "../src/featurize.js";
import { structuralPattern } from "../src/fingerprint.js";

function temporary(run: (directory: string) => void) {
  const directory = mkdtempSync(join(tmpdir(), "syntaxless-validation-preparation-"));

  try {
    run(directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

test(
  "streamed JSONL preserves split UTF-8 characters " + "and cannot overwrite an existing corpus",
  () =>
    temporary((directory) => {
      const path = join(directory, "rows.jsonl");
      const rows = [{ text: "a".repeat(65524) + "😀é" }, { text: "final record" }];
      const result = writeJsonl(path, rows);

      expect([...readJsonl(path)]).toEqual(rows);
      const bytes = readFileSync(path);

      expect(result).toEqual({
        rows: 2,
        sha256: createHash("sha256").update(bytes).digest("hex"),
      });

      expect(() => writeJsonl(path, [{ text: "replacement" }])).toThrow();
      expect(readFileSync(path)).toEqual(bytes);
      writeFileSync(join(directory, "ending.jsonl"), '\r\n{"text":"last"}');
      expect([...readJsonl(join(directory, "ending.jsonl"))]).toEqual([{ text: "last" }]);
    }),
);

test(
  "compact question variants keep the statement frame " +
    "and streaming generation preserves the seeded rows",
  () =>
    temporary((directory) => {
      const rows = generateCompact(800);
      const questions = rows.filter((row) => row.template === "question");

      expect(questions.length).toBeGreaterThan(0);

      const questionGroups = questions.map((row) => {
        const pattern = structuralPattern(
          row.text,
          row.spans.filter((span) => span.label !== "UNSUPPORTED"),
        ).filter((part) => part !== "O");

        return createHash("sha256").update(JSON.stringify(pattern)).digest("hex");
      });

      expect(questionGroups).toEqual(questions.map((row) => row.fingerprint));
      const path = join(directory, "corpus.jsonl");

      writeCompact(path, 800);
      expect([...readJsonl(path)]).toEqual(rows);
      expect(() => writeCompact(path, 800)).toThrow();
    }),
);

test("file preparation rejects a split leak even when only one split is requested", () =>
  temporary((directory) => {
    const row = generateCompact(1)[0]!;
    const path = join(directory, "valid.jsonl");

    writeJsonl(path, [{ ...row, split: "train" }]);

    const positive = featurizeFiles({
      input: path,
      out: join(directory, "valid"),
      split: "train",
    });

    expect(positive.sequences).toBe(1);
    const leaked = join(directory, "leaked.jsonl");

    writeJsonl(leaked, [
      { ...row, split: "train" },
      { ...row, split: "heldout" },
    ]);

    expect(() =>
      featurizeFiles({ input: leaked, out: join(directory, "leaked"), split: "train" }),
    ).toThrow("structural split leak");

    const invalid = join(directory, "invalid.jsonl");

    writeJsonl(invalid, [
      {
        ...row,
        split: "train",
        spans: row.spans.map((span, i) => (i ? span : { ...span, label: "UNKNOWN_ROLE" })),
      },
    ]);

    expect(() => featurizeFiles({ input: invalid, out: join(directory, "invalid") })).toThrow(
      "unknown training role",
    );
  }));

test(
  "reserved exclusions require complete wording and " +
    "preserve verified archives after retiring full rows",
  () =>
    temporary((directory) => {
      const metadataDirectory = join(directory, "evaluation");
      const consumedDirectory = join(metadataDirectory, "consumed");

      mkdirSync(consumedDirectory, { recursive: true });

      const path = join(metadataDirectory, "reserved.jsonl");

      const row = {
        text: "  Please preserve the declared intent  ",
        trainingEligible: false,
      };

      const dataset = writeJsonl(path, [row]);

      writeFileSync(path + ".meta.json", JSON.stringify({ ...dataset, trainingEligible: false }));

      // A retired duplicate may precede its intact copy in the metadata inventory.
      writeFileSync(
        join(metadataDirectory, "a-retired.jsonl.meta.json"),
        JSON.stringify({ ...dataset, trainingEligible: false }),
      );

      const textHash = createHash("sha256").update(row.text.trim().toLowerCase()).digest("hex");

      expect(reservedHashes(metadataDirectory, consumedDirectory)).toEqual(new Set([textHash]));
      unlinkSync(path);

      expect(() => reservedHashes(metadataDirectory, consumedDirectory)).toThrow(
        "unconsumed reserved dataset is missing",
      );

      const archive = JSON.stringify({ sha256: dataset.sha256, textHashes: [textHash] }) + "\n";
      const archivePath = join(consumedDirectory, dataset.sha256 + ".archive.json");

      writeFileSync(archivePath, archive);

      writeFileSync(
        join(consumedDirectory, dataset.sha256 + ".json"),
        JSON.stringify({
          sha256: dataset.sha256,
          archiveSha256: createHash("sha256").update(archive).digest("hex"),
        }),
      );

      expect(reservedHashes(metadataDirectory, consumedDirectory)).toEqual(new Set([textHash]));
      writeFileSync(archivePath, archive + "\n");

      expect(() => reservedHashes(metadataDirectory, consumedDirectory)).toThrow(
        "consumed archive changed",
      );
    }),
);
