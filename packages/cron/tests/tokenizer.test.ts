import { expect, test } from "bun:test";

import { tokenize } from "../src/model/tokenizer.ts";

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
