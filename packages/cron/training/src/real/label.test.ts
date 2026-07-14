import { expect, test } from "bun:test";

import { assertLabelK, LABEL_K, LABEL_TEMPERATURES, parseConstrainedIntent } from "./label";

test("label k=3 constrained JSON accepts a schedule and null", () => {
  expect(LABEL_K).toBe(3);
  expect(LABEL_TEMPERATURES).toEqual([0, 0.4, 0.8]);
  expect(parseConstrainedIntent(null)).toBeNull();

  expect(parseConstrainedIntent({ frequency: { unit: "day", interval: 1 } })).toEqual({
    frequency: { unit: "day", interval: 1 },
  });

  expect(() => parseConstrainedIntent({ frequency: { unit: "day" } })).toThrow(/schema-invalid/);
  expect(() => assertLabelK([{ temperature: 0, intent: null }])).toThrow(/k=3/);
});
