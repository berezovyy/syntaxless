/**
 * Deterministic pseudo-random number source for the corpus generator.
 *
 * The training pipeline must be reproducible from a seed, so it never touches
 * `Math.random`. `mulberry32` is a small, fast 32-bit generator with enough
 * distribution for sampling corpus structure.
 */

export interface Rng {
  /** Uniform float in [0, 1). */
  next(): number;
  /** Uniform integer in [min, max], inclusive. */
  int(min: number, max: number): number;
  /** Uniform choice from a non-empty list. */
  pick<T>(items: readonly T[]): T;
  /** Weighted choice from a non-empty list of [item, weight] pairs. */
  weighted<T>(items: readonly (readonly [T, number])[]): T;
  /** True with the given probability (default 0.5). */
  bool(probability?: number): boolean;
}

/** Wrap a seeded mulberry32 stream behind the shared {@link Rng} interface. */
export function createRng(seed: number): Rng {
  const next = mulberry32(seed);

  return {
    next,
    int(min, max) {
      if (max < min) throw new RangeError(`int(${min}, ${max}): max must be >= min`);

      return min + Math.floor(next() * (max - min + 1));
    },
    pick(items) {
      if (items.length === 0) throw new RangeError("pick() expects a non-empty list");

      return items[Math.floor(next() * items.length)]!;
    },
    weighted(items) {
      if (items.length === 0) throw new RangeError("weighted() expects a non-empty list");
      let total = 0;

      for (const [, weight] of items) {
        if (!Number.isFinite(weight) || weight < 0) {
          throw new RangeError("weighted() weights must be finite and non-negative");
        }

        total += weight;
      }

      if (total <= 0) throw new RangeError("weighted() total weight must be positive");
      let target = next() * total;

      for (const [item, weight] of items) {
        target -= weight;
        if (target < 0) return item;
      }

      return items[items.length - 1]![0];
    },
    bool(probability = 0.5) {
      if (probability < 0 || probability > 1) {
        throw new RangeError(`bool(${probability}): probability must be in [0, 1]`);
      }

      return next() < probability;
    },
  };
}

/** Mulberry32: a compact seeded 32-bit PRNG returning floats in [0, 1). */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;

  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);

    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
