import { expect, test } from "bun:test";

import {
  EMBEDDING_ROWS,
  FEATURE_COUNT,
  featureRows,
  PADDING_ROW,
  tokenize,
} from "../src/model/tokenizer.ts";

const SAMPLES = [
  "every 2 weeks at 9:30 am on Mondays",
  "first Monday of every month",
  "on the 15th of every month",
  "from 2026-01-01 to 2027-03-04",
  "weekdays excluding Saturday and Sunday",
  "I'd like to book a meeting, please.",
  "café — 3 items",
  "",
  "   ",
];

test("offsets round-trip through UTF-16 slices for every token", () => {
  for (const text of SAMPLES) {
    for (const token of tokenize(text)) {
      expect(text.slice(token.start, token.end)).toBe(token.text);
    }
  }
});

test("featureRows returns at most FEATURE_COUNT disjoint valid rows", () => {
  expect(FEATURE_COUNT).toBe(32);
  expect(PADDING_ROW).toBe(EMBEDDING_ROWS);

  for (const text of SAMPLES) {
    for (const token of tokenize(text)) {
      const rows = featureRows(token);

      expect(rows.length).toBeLessThanOrEqual(FEATURE_COUNT);
      expect(new Set(rows).size).toBe(rows.length);

      for (const row of rows) {
        expect(Number.isInteger(row)).toBe(true);
        expect(row).toBeGreaterThanOrEqual(0);
        expect(row).toBeLessThan(EMBEDDING_ROWS);
      }
    }
  }
});

test("character n-grams preserve evidence across common misspellings", () => {
  const rows = (word: string) =>
    new Set(featureRows(tokenize(word)[0]!).filter((row) => row >= 580));

  const overlap = (left: Set<number>, right: Set<number>) =>
    [...left].filter((row) => right.has(row)).length;

  expect(overlap(rows("monday"), rows("mnday"))).toBeGreaterThan(0);
  expect(overlap(rows("friday"), rows("firday"))).toBeGreaterThan(0);
  expect(overlap(rows("every"), rows("evry"))).toBeGreaterThan(0);
});

test("legacy weight layouts omit character n-grams", () => {
  const token = tokenize("monday")[0]!;
  const rows = featureRows(token, { featureRows: 580, rowsPerToken: 17 });

  expect(rows.length).toBeLessThanOrEqual(17);
  expect(rows.every((row) => row < 580)).toBe(true);
});

test("token kinds and multi-token whitespace are preserved", () => {
  const tokens = tokenize("a  b");

  expect(tokens.map((token) => token.kind)).toEqual([0, 3, 0]);
  expect(tokens[1]!.text).toBe("  ");
});
