import { expect, test } from "bun:test";

import { parseComparison } from "../src/comparison/parse.ts";
import type { ComparisonParseFailureCode } from "../src/comparison/parse.ts";
import type { ComparisonIntent } from "../src/index.ts";

const positives: Array<{
  phrase: string;
  comparison: ComparisonIntent;
  unit?: "character";
}> = [
  {
    phrase: "more than one hundred",
    comparison: { min: { value: 100, inclusive: false } },
  },
];

for (const { phrase, comparison, unit } of positives) {
  test(`parseComparison(${JSON.stringify(phrase)})`, () => {
    const result = parseComparison(phrase);

    expect(result.ok).toBe(true);

    if (result.ok) {
      expect(result.comparison).toEqual(comparison);
      expect(result.unit).toBe(unit);
    }
  });
}

const negatives: Array<{ phrase: string; code: ComparisonParseFailureCode }> = [
  { phrase: "between 5", code: "incomplete-range" },
];

for (const { phrase, code } of negatives) {
  test(`parseComparison(${JSON.stringify(phrase)}) fails as ${code}`, () => {
    const result = parseComparison(phrase);

    expect(result).toEqual({ ok: false, code });
  });
}
