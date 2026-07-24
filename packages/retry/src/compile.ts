import { ceil, numberDecimal } from "./exact.js";
import { InputError } from "./types.js";
import type { Jitter, Result, RetryIntent } from "./types.js";
import { validate } from "./validate.js";

export interface CompileOptions {
  seed?: number;
}

function bounds(delay: number, jitter: Jitter, cap?: number): [number, number] {
  let low = 0n;
  let high = BigInt(delay);

  if (jitter.kind === "equal") low = ceil(high, 2n);

  if (jitter.kind === "proportional") {
    const f = numberDecimal(jitter.fraction);
    const base = BigInt(delay);

    low = ceil(base * (f.denominator - f.numerator), f.denominator);
    high = (base * (f.denominator + f.numerator)) / f.denominator;
    // Cap the final jitter range as well as the base schedule.
    if (cap !== undefined && high > BigInt(cap)) high = BigInt(cap);
  }

  if (high > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new InputError("overflow", "jitter range exceeds safe whole milliseconds");
  }

  return [Number(low), Number(high)];
}

export function compile(
  intent: RetryIntent,
  options: CompileOptions = {},
): Result<{ delaysMs: readonly number[] }> {
  try {
    validate(intent);

    if (
      options.seed !== undefined &&
      (!Number.isInteger(options.seed) || options.seed < 0 || options.seed > 0xffffffff)
    ) {
      throw new InputError("invalid-input", "seed must be an unsigned 32-bit integer");
    }

    if (intent.jitter && intent.jitter.kind !== "none" && options.seed === undefined) {
      throw new InputError("missing-seed", "provide a seed for reproducible jitter");
    }

    let state = options.seed ?? 0;

    const next = (): bigint => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;

      return BigInt(state);
    };

    const delays: number[] = [];
    const factor = numberDecimal(intent.multiplier ?? 1);

    let numerator = BigInt(intent.initialDelayMs);
    let denominator = 1n;

    const cap = intent.maxDelayMs === undefined ? undefined : BigInt(intent.maxDelayMs);

    for (let index = 0; index < intent.retries; index++) {
      if (cap !== undefined && numerator >= cap * denominator) {
        numerator = cap;
        denominator = 1n;
      }

      const base = ceil(numerator, denominator);

      if (base > BigInt(Number.MAX_SAFE_INTEGER)) {
        throw new InputError("overflow", "uncapped schedule exceeds safe whole milliseconds");
      }

      let delay = Number(base);

      if (intent.jitter && intent.jitter.kind !== "none") {
        const [low, high] = bounds(delay, intent.jitter, intent.maxDelayMs);
        // Combine two LCG outputs into 53 bits; use integer arithmetic for large ranges.
        const sample = (next() << 21n) + (next() >> 11n);

        delay = low + Number((sample * (BigInt(high) - BigInt(low) + 1n)) / (1n << 53n));
      }

      delays.push(delay);

      if (intent.strategy === "exponential" && numerator !== 0n) {
        numerator *= factor.numerator;
        denominator *= factor.denominator;
      }
    }

    return { supported: true, delaysMs: delays };
  } catch (error) {
    if (error instanceof InputError) {
      return { supported: false, code: error.code, reason: error.message };
    }

    throw error;
  }
}
