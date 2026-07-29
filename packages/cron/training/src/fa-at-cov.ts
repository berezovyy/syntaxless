/**
 * FA@cov: false-accept rate on negatives at the coverage the abstention
 * threshold yields on positives. Coverage is H-pos accepted / H-pos, not
 * overall accuracy.
 */
export function faAtCoverage(input: {
  negativeTotal: number;
  negativeAccepted: number;
  positiveTotal: number;
  positiveAccepted: number;
}): { fa: number; coverage: number; formatted: string } {
  const fa = input.negativeAccepted / Math.max(input.negativeTotal, 1);
  const coverage = input.positiveAccepted / Math.max(input.positiveTotal, 1);

  return { fa, coverage, formatted: `${fa}@${coverage}` };
}
