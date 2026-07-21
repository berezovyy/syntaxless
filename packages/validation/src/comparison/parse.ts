import { validateComparison } from "../compiler/validate.js";
import {
  COMPARISON_PHRASES,
  type ComparisonMeaning,
  matchLongestAlias,
  type Unit,
  UNIT_PHRASES,
} from "../parser/lexicon.js";
import { normalize } from "../parser/normalize.js";
import type { ComparisonIntent } from "../types/intent.js";
import { readNumber } from "./number.js";

export type ComparisonParseFailureCode =
  "empty" | "unrecognized" | "incomplete-range" | "conflicting-bounds" | "inverted-range";

export type ComparisonParseResult =
  | { ok: true; comparison: ComparisonIntent; unit?: Unit }
  | { ok: false; code: ComparisonParseFailureCode };

const GLUE = new Set(["and", "but"]);
const DIGITS = /^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$/;

const SYMBOL_OPERATORS = COMPARISON_PHRASES.map((phrase) => phrase.surface).filter((surface) =>
  /^[<>=]+$/.test(surface),
);

function fail(code: ComparisonParseFailureCode): ComparisonParseResult {
  return { ok: false, code };
}

function tokenize(normalized: string): string[] {
  const raw = normalized.split(/[\s,]+/).filter((token) => token.length > 0);
  const tokens: string[] = [];

  for (const token of raw) {
    let split = false;

    for (const symbol of SYMBOL_OPERATORS) {
      if (!token.startsWith(symbol)) continue;
      const rest = token.slice(symbol.length);

      if (DIGITS.test(rest)) {
        tokens.push(symbol, rest);
        split = true;
        break;
      }
    }

    if (!split) tokens.push(token);
  }

  return tokens;
}

function mergeComparison(into: ComparisonIntent, next: ComparisonIntent): boolean {
  if (next.min !== undefined) {
    if (into.min !== undefined) return false;
    into.min = next.min;
  }

  if (next.max !== undefined) {
    if (into.max !== undefined) return false;
    into.max = next.max;
  }

  return true;
}

function clauseFromOperator(
  op: Exclude<ComparisonMeaning, { side: "between" }>,
  value: number,
): ComparisonIntent {
  if (op.side === "exact") {
    return {
      min: { value, inclusive: true },
      max: { value, inclusive: true },
    };
  }

  if (op.side === "min") {
    return { min: { value, inclusive: op.inclusive } };
  }

  return { max: { value, inclusive: op.inclusive } };
}

export function parseComparison(text: string): ComparisonParseResult {
  if (typeof text !== "string") {
    return fail("unrecognized");
  }

  const normalized = normalize(text);

  if (normalized.length === 0) return fail("empty");

  const tokens = tokenize(normalized);
  let index = 0;
  let unit: Unit | undefined;
  const comparison: ComparisonIntent = {};
  let sawClause = false;

  while (index < tokens.length) {
    const start = index;

    if (sawClause && tokens[index] !== undefined && GLUE.has(tokens[index]!)) {
      index += 1;
    }

    const op = matchLongestAlias(tokens, index, COMPARISON_PHRASES);

    if (!op) {
      index = start;
      break;
    }

    index += op.tokenCount;

    if (op.value.side === "between") {
      const startNumber = readNumber(tokens, index);

      if (!startNumber) return fail("incomplete-range");
      index += startNumber.tokenCount;
      if (tokens[index] !== "and") return fail("incomplete-range");
      index += 1;
      const endNumber = readNumber(tokens, index);

      if (!endNumber) return fail("incomplete-range");
      index += endNumber.tokenCount;
      if (startNumber.value > endNumber.value) return fail("inverted-range");

      const next: ComparisonIntent = {
        min: { value: startNumber.value, inclusive: true },
        max: { value: endNumber.value, inclusive: true },
      };

      if (!mergeComparison(comparison, next)) return fail("conflicting-bounds");
      sawClause = true;
      continue;
    }

    const number = readNumber(tokens, index);

    if (!number) return fail("unrecognized");
    index += number.tokenCount;
    const next = clauseFromOperator(op.value, number.value);

    if (!mergeComparison(comparison, next)) return fail("conflicting-bounds");
    sawClause = true;
  }

  while (index < tokens.length) {
    const unitMatch = matchLongestAlias(tokens, index, UNIT_PHRASES);

    if (!unitMatch) break;
    unit = unitMatch.value;
    index += unitMatch.tokenCount;
  }

  if (!sawClause) return fail("unrecognized");
  if (index !== tokens.length) return fail("unrecognized");

  const invalid = validateComparison(comparison);

  if (invalid) {
    return fail(invalid.code === "conflicting-bounds" ? "conflicting-bounds" : "unrecognized");
  }

  return unit === undefined ? { ok: true, comparison } : { ok: true, comparison, unit };
}
