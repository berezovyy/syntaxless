/** Exact decimal arithmetic; round only when the schedule contract requires it. */
import { InputError } from "./types.js";

export interface Fraction {
  numerator: bigint;
  denominator: bigint;
}

export function decimal(text: string): Fraction {
  const match = /^([+-]?)(\d*)(?:\.(\d+))?$/.exec(text);

  if (!match || !(match[2] || match[3]) || text.length > 128) {
    throw new InputError("invalid-input", "provide an ordinary decimal literal");
  }

  const denominator = 10n ** BigInt(match[3]?.length ?? 0);

  return {
    numerator:
      (BigInt(match[2] || "0") * denominator + BigInt(match[3] || "0")) *
      (match[1] === "-" ? -1n : 1n),
    denominator,
  };
}

export function exactNumber(text: string): number {
  const original = decimal(text);
  const value = Number(text);

  if (!Number.isFinite(value) || Math.abs(value) > Number.MAX_SAFE_INTEGER) {
    throw new InputError("invalid-input", "number is outside the finite safe range");
  }

  const rendered = numberDecimal(value);

  if (rendered.numerator * original.denominator !== original.numerator * rendered.denominator) {
    throw new InputError(
      "invalid-input",
      "number cannot be preserved by the public number representation",
    );
  }

  return value;
}

export function integer(value: Fraction): number {
  if (value.numerator % value.denominator !== 0n) {
    throw new InputError("invalid-input", "counts and durations must resolve to whole numbers");
  }

  const result = value.numerator / value.denominator;

  if (result < 0n || result > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new InputError("invalid-input", "value is outside the nonnegative safe integer range");
  }

  return Number(result);
}

export const ceil = (n: bigint, d: bigint): bigint => (n + d - 1n) / d;

/** Decode the decimal representation of a finite public number, including exponent notation. */
export function numberDecimal(value: number): Fraction {
  const text = String(value);
  const match = /^(.+)e([+-]?\d+)$/i.exec(text);

  if (!match) return decimal(text);

  const base = decimal(match[1]!);
  const exponent = Number(match[2]);

  return exponent >= 0
    ? {
        numerator: base.numerator * 10n ** BigInt(exponent),
        denominator: base.denominator,
      }
    : {
        numerator: base.numerator,
        denominator: base.denominator * 10n ** BigInt(-exponent),
      };
}
