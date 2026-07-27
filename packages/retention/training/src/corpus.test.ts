import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { LABELS } from "../../src/labels.js";
import { prepare } from "../../src/prepare.js";
import { generate, hash, writeCorpus } from "./corpus.js";
import { featurize } from "./featurize.js";

test("numeric substitutions retain a frame and questions recover the unchanged base", () => {
  const a = prepare("keep daily backups for 7 days");
  const b = prepare("keep daily backups for 3 days");

  expect(a.text).toBe(b.text);

  expect(a.lexemes.filter((lexeme) => lexeme.numeric).map((lexeme) => lexeme.numeric)).not.toEqual(
    b.lexemes.filter((lexeme) => lexeme.numeric).map((lexeme) => lexeme.numeric),
  );

  const rows = generate(64);
  const questions = rows.filter((row) => row.family === "question");

  expect(questions.length).toBeGreaterThan(0);

  expect(
    questions.map((row) =>
      hash(
        JSON.stringify([
          prepare(row.text)
            .lexemes.slice(0, -1)
            .map((lexeme) => lexeme.text),
          row.roles.slice(0, -1),
        ]),
      ),
    ),
  ).toEqual(questions.map((row) => row.fingerprint));

  const owners = new Map<string, string>();

  expect(
    rows.every((row) => {
      const owner = owners.get(row.fingerprint);

      owners.set(row.fingerprint, row.split);

      return owner === undefined || owner === row.split;
    }),
  ).toBe(true);
});

test(
  "immutable preparation preserves semantic boundaries " +
    "and rejects missing labels or split leaks",
  () => {
    const directory = mkdtempSync(join(tmpdir(), "syntaxless-retention-preparation-"));

    try {
      const path = join(directory, "corpus.jsonl");
      const rows = writeCorpus(path, 8);
      const bytes = readFileSync(path);
      const row = rows[0]!;

      expect(() => writeCorpus(path, 8)).toThrow();
      expect(readFileSync(path)).toEqual(bytes);
      const prefix = join(directory, "valid");

      featurize([row], prefix);
      const labels = [...readFileSync(prefix + ".labels.bin")];

      expect([...readFileSync(prefix + ".boundaries.bin")]).toEqual(
        labels.map((label) =>
          Number(label !== 255 && LABELS[label] !== "O" && LABELS[label] !== "GLUE"),
        ),
      );

      expect(
        labels.some((label) => label !== 255 && LABELS[label] !== "O" && LABELS[label] !== "GLUE"),
      ).toBe(true);

      expect(() =>
        featurize(
          [row, { ...row, split: row.split === "train" ? "heldout" : "train" }],
          join(directory, "leak"),
          "train",
        ),
      ).toThrow("crosses corpus splits");

      expect(() =>
        featurize([{ ...row, trainingEligible: false }], join(directory, "reserved")),
      ).toThrow("evaluation-only");

      expect(() => featurize([{ ...row, roles: [] }], join(directory, "unaligned"))).toThrow(
        "unaligned role supervision",
      );

      expect(() =>
        featurize(
          [{ ...row, roles: row.roles.map(() => "UNKNOWN_ROLE" as never) }],
          join(directory, "unknown"),
        ),
      ).toThrow("unknown training role");
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  },
);
