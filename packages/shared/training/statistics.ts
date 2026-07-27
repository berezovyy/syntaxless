/** 95% Wilson score interval for `n` successes out of `d`; zero observed is not zero risk. */
export function interval(n: number, d: number): [number, number] {
  if (!d) return [0, 1];

  const z = 1.96;
  const p = n / d;
  const denominator = 1 + (z * z) / d;

  const middle = (p + (z * z) / (2 * d)) / denominator;
  const radius = (z * Math.sqrt((p * (1 - p)) / d + (z * z) / (4 * d * d))) / denominator;

  return [Math.max(0, middle - radius), Math.min(1, middle + radius)];
}
