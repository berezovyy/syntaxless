import { expect, test } from "bun:test";

import { faAtCoverage, faAtCovMeetsTarget } from "./fa-at-cov";

test("FA@cov is accepted-negatives/negatives at accepted-positives/positives", () => {
  const scored = faAtCoverage({
    negativeTotal: 100,
    negativeAccepted: 2,
    positiveTotal: 50,
    positiveAccepted: 45,
  });

  expect(scored.fa).toBeCloseTo(0.02);
  expect(scored.coverage).toBeCloseTo(0.9);
  expect(scored.formatted).toBe(`${scored.fa}@${scored.coverage}`);
  expect(faAtCovMeetsTarget("0.01@0.9")).toBe(true);
  expect(faAtCovMeetsTarget("0.02@0.9")).toBe(false);
  expect(faAtCovMeetsTarget("0.01@0.8")).toBe(false);
});
