import { matchLongestAlias, NUMBER_PHRASES } from "../parser/lexicon.js";

/** Resolve an exact literal or a supported word number without rounding it. */
export function readNumber(
  tokens: readonly string[],
  index: number,
): { value: number; tokenCount: number } | null {
  const phrase = matchLongestAlias(tokens, index, NUMBER_PHRASES);

  if (phrase) return { value: phrase.value, tokenCount: phrase.tokenCount };
  const literal = tokens[index];

  if (literal === undefined || !/^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$/.test(literal)) return null;
  const value = Number(literal);

  if (!Number.isFinite(value) || Math.abs(value) > Number.MAX_SAFE_INTEGER) return null;
  if (Number.isInteger(value) && !Number.isSafeInteger(value)) return null;

  return { value, tokenCount: 1 };
}
