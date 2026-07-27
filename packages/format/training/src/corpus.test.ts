import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { Role } from "../../src/labels.js";
import { LABELS } from "../../src/labels.js";
import { FEATURE_COUNT, featureRows, PADDING_ROW, tokenize } from "../../src/model/tokenizer.js";
import { prepare } from "../../src/prepare.js";
import { constructionFingerprint, generate } from "./corpus.js";
import type { Row } from "./corpus.js";
import { featurize } from "./featurize.js";

test("literal substitutions and paired questions cannot move a construction between splits", () => {
  const roles: Role[] = ["CURRENCY", "GLUE", "EXACT", "PRECISION", "GLUE", "GLUE", "LOCALE"];
  const a = constructionFingerprint("euros with two decimals in locale en-US", roles);
  const b = constructionFingerprint("Canadian dollars with 7 decimals in locale fr-FR", roles);

  expect(a).toBe(b);

  expect(
    constructionFingerprint(
      "euros with two decimals in locale en-US",
      roles.map((r) => (r === "EXACT" ? "MAX" : r)),
    ),
  ).not.toBe(a);

  const rows = generate(96);
  const questions = rows.filter((r) => r.family === "question");

  expect(questions.length).toBeGreaterThan(0);

  for (const row of questions) {
    const base = row.text.slice(0, -1);

    expect(row.fingerprint).toBe(constructionFingerprint(base, row.roles.slice(0, -1)));
  }

  const assignments = new Map<string, Row["split"]>();

  for (const row of rows) {
    const previous = assignments.get(row.fingerprint);

    if (previous) expect(row.split).toBe(previous);
    assignments.set(row.fingerprint, row.split);
  }
});

test(
  "buffered binary writing preserves offsets, " + "local neighbors and role labels across a flush",
  () => {
    const directory = mkdtempSync(join(tmpdir(), "syntaxless-format-features-"));

    try {
      const text = "number with 2 decimals";
      const roles: Role[] = ["NUMBER", "GLUE", "EXACT", "PRECISION"];

      const row: Row = {
        text,
        context: {},
        intent: {
          kind: "number",
          negative: "minus",
          precision: { minimum: 2, maximum: 2 },
        },
        roles,
        family: "number",
        split: "train",
        fingerprint: constructionFingerprint(text, roles),
      };

      const tokens = tokenize(prepare(text).text);
      const count = Math.floor(8192 / tokens.length) + 2;

      expect(tokens.length * count).toBeGreaterThan(8192);
      const prefix = join(directory, "features");

      featurize(
        Array.from({ length: count }, () => row),
        prefix,
      );

      const binary = (name: string) => readFileSync(prefix + `.${name}.bin`);

      const offsets = binary("offsets");
      const features = binary("rows");
      const labels = binary("labels");
      const neighbors = binary("neighbors");

      expect(offsets.length).toBe((count + 1) * 4);
      expect(features.length).toBe(count * tokens.length * FEATURE_COUNT * 2);
      expect(labels.length).toBe(count * tokens.length);

      const expectedFeatures = tokens.flatMap((token) => {
        const active = featureRows(token);

        return [...active, ...Array(FEATURE_COUNT - active.length).fill(PADDING_ROW)];
      });

      const lexemes = prepare(text).lexemes;
      const nonspace = tokens.map((t, i) => (t.kind === 3 ? -1 : i)).filter((i) => i >= 0);

      const expectedLabels = tokens.map((token) =>
        token.kind === 3
          ? 255
          : LABELS.indexOf(roles[lexemes.findIndex((l) => l.start === token.start)]!),
      );

      const expectedNeighbors = tokens.flatMap((_, i) => [
        nonspace.findLast((j) => j < i) ?? -1,
        nonspace.find((j) => j > i) ?? -1,
      ]);

      for (const sequence of [0, count - 2, count - 1]) {
        expect(offsets.readUInt32LE(sequence * 4)).toBe(sequence * tokens.length);

        expect(
          Array.from({ length: expectedFeatures.length }, (_, i) =>
            features.readUInt16LE((sequence * expectedFeatures.length + i) * 2),
          ),
        ).toEqual(expectedFeatures);

        expect([
          ...labels.subarray(sequence * tokens.length, (sequence + 1) * tokens.length),
        ]).toEqual(expectedLabels);

        expect(
          Array.from({ length: expectedNeighbors.length }, (_, i) =>
            neighbors.readInt16LE((sequence * expectedNeighbors.length + i) * 2),
          ),
        ).toEqual(expectedNeighbors);
      }

      expect(offsets.readUInt32LE(count * 4)).toBe(count * tokens.length);

      expect(() =>
        featurize([{ ...row, trainingEligible: false }], join(directory, "reserved")),
      ).toThrow("evaluation-only");
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  },
);
