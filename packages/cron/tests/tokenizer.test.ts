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
  expect(FEATURE_COUNT).toBe(17);
  expect(PADDING_ROW).toBe(EMBEDDING_ROWS);

  for (const text of SAMPLES) {
    for (const token of tokenize(text)) {
      const rows = featureRows(token.features);

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

test("token kinds and multi-token whitespace are preserved", () => {
  const tokens = tokenize("a  b");

  expect(tokens.map((token) => token.kind)).toEqual([0, 3, 0]);
  expect(tokens[1]!.text).toBe("  ");
});
