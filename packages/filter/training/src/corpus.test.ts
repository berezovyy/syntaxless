import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { prepare } from "../../src/prepare.js";
import { generate, hash, writeCorpus } from "./corpus.js";
import { featurize } from "./featurize.js";

test(
  "caller field and value substitutions share a " +
    "language frame and questions retain their base split",
  () => {
    const a = prepare("spending over 100", {
      fields: [{ id: "a", aliases: ["spending"], type: "number" }],
    });

    const b = prepare("cost over 3.25", {
      fields: [{ id: "b", aliases: ["cost"], type: "number" }],
    });

    expect(a.text).toBe(b.text);
    expect(a.lexemes[0]!.field).not.toBe(b.lexemes[0]!.field);
    expect(a.lexemes[2]!.value).not.toBe(b.lexemes[2]!.value);
    const questions = generate(64).filter((row) => row.family === "question");

    expect(questions.length).toBeGreaterThan(0);

    const fingerprints = questions.map((row) => {
      const lexemes = prepare(row.text, row.context).lexemes;
      let start = 0;

      while (row.roles[start] === "UNSUPPORTED") start++;

      return hash(
        JSON.stringify(
          lexemes.slice(start, -1).map((lexeme, i) => [lexeme.text, row.roles[start + i]]),
        ),
      );
    });

    expect(fingerprints).toEqual(questions.map((row) => row.fingerprint));
  },
);

test(
  "corpus output is immutable and missing or unknown " +
    "supervision cannot be written as valid training roles",
  () => {
    const directory = mkdtempSync(join(tmpdir(), "syntaxless-filter-preparation-"));

    try {
      const path = join(directory, "corpus.jsonl");
      const rows = writeCorpus(path, 8);
      const bytes = readFileSync(path);

      expect(rows.length).toBe(8);
      expect(() => writeCorpus(path, 8)).toThrow();
      expect(readFileSync(path)).toEqual(bytes);
      const row = rows[0]!;

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
