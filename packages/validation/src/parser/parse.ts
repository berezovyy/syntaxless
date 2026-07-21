import { parseComparison } from "../comparison/parse.js";
import { validateField } from "../compiler/validate.js";
import type { Bound, ComparisonIntent, FieldIntent, FieldType } from "../types/intent.js";
import {
  matchLongestAlias,
  NULLABLE_PHRASES,
  PRESENCE_PHRASES,
  TYPE_PHRASES,
  type Unit,
} from "./lexicon.js";
import type { ModelParserOptions } from "./model.js";
import { normalize } from "./normalize.js";

export type ParseFailureCode =
  | "empty"
  | "unrecognized"
  | "missing-type"
  | "missing-presence"
  | "unrecognized-slot"
  | "conflicting-bounds"
  | "conflicting-presence"
  | "inverted-range"
  | "low-confidence"
  | "model-error"
  | "incompatible-model";

export type ParseSource = "rules" | "model";

export type ParseOptions = ModelParserOptions;

export interface Diagnostic {
  code: string;
  severity: "error" | "warning" | "suggestion";
  message: string;
}

export type ParseResult =
  | {
      ok: true;
      intent: FieldIntent;
      confidence: number;
      diagnostics: Diagnostic[];
      source: ParseSource;
    }
  | {
      ok: false;
      reason: string;
      code: ParseFailureCode;
      confidence: number;
      diagnostics: Diagnostic[];
      source: ParseSource;
    };

const STRING_LIKE = new Set<FieldType>(["string", "email", "url", "uuid"]);
const NUMBER_LIKE = new Set<FieldType>(["number", "integer"]);
const FILLER = new Set(["with"]);

function fail(code: ParseFailureCode, reason: string): ParseResult {
  return { ok: false, reason, code, confidence: 0, diagnostics: [], source: "rules" };
}

function succeed(intent: FieldIntent): ParseResult {
  return { ok: true, intent, confidence: 1, diagnostics: [], source: "rules" };
}

function tokenize(normalized: string): string[] {
  return normalized.split(/[\s,]+/).filter((token) => token.length > 0);
}

function markUsed(used: boolean[], index: number, count: number): void {
  for (let offset = 0; offset < count; offset++) used[index + offset] = true;
}

function inclusiveLength(bound: Bound, side: "min" | "max"): number {
  if (bound.inclusive) return bound.value;

  return side === "min" ? bound.value + 1 : bound.value - 1;
}

function placeComparison(
  type: FieldType,
  comparison: ComparisonIntent,
  unit: Unit | undefined,
): { fields: Pick<FieldIntent, "minLength" | "maxLength" | "minimum" | "maximum"> } | ParseResult {
  if (STRING_LIKE.has(type)) {
    if (unit !== "character") {
      return fail("unrecognized-slot", "string-like comparisons require a character unit");
    }

    const fields: Pick<FieldIntent, "minLength" | "maxLength" | "minimum" | "maximum"> = {};

    if (comparison.min !== undefined) fields.minLength = inclusiveLength(comparison.min, "min");
    if (comparison.max !== undefined) fields.maxLength = inclusiveLength(comparison.max, "max");

    return { fields };
  }

  if (NUMBER_LIKE.has(type)) {
    if (unit !== undefined) {
      return fail("unrecognized-slot", "number-like comparisons do not take a unit");
    }

    const fields: Pick<FieldIntent, "minLength" | "maxLength" | "minimum" | "maximum"> = {};

    if (comparison.min !== undefined) fields.minimum = comparison.min;
    if (comparison.max !== undefined) fields.maximum = comparison.max;

    return { fields };
  }

  return fail("unrecognized-slot", "this type does not take a comparison");
}

function parseWithRules(text: string): ParseResult {
  if (typeof text !== "string") {
    return fail("unrecognized", "unrecognized input");
  }

  const normalized = normalize(text);

  if (normalized.length === 0) return fail("empty", "empty input");

  const tokens = tokenize(normalized);
  const used = new Array<boolean>(tokens.length).fill(false);
  const types: FieldType[] = [];
  const presence: boolean[] = [];
  let nullable = false;

  for (let index = 0; index < tokens.length; index++) {
    if (used[index]) continue;

    const presenceMatch = matchLongestAlias(tokens, index, PRESENCE_PHRASES);

    if (presenceMatch) {
      presence.push(presenceMatch.value);
      markUsed(used, index, presenceMatch.tokenCount);
      index += presenceMatch.tokenCount - 1;
      continue;
    }

    const typeMatch = matchLongestAlias(tokens, index, TYPE_PHRASES);

    if (typeMatch) {
      types.push(typeMatch.value);
      markUsed(used, index, typeMatch.tokenCount);
      index += typeMatch.tokenCount - 1;
      continue;
    }

    const nullableMatch = matchLongestAlias(tokens, index, NULLABLE_PHRASES);

    if (nullableMatch) {
      nullable = true;
      markUsed(used, index, nullableMatch.tokenCount);
      index += nullableMatch.tokenCount - 1;
      continue;
    }

    if (tokens[index] !== undefined && FILLER.has(tokens[index]!)) {
      used[index] = true;
    }
  }

  if (types.length === 0 && presence.length === 0) {
    return fail("unrecognized", "unrecognized input");
  }

  if (types.length === 0) return fail("missing-type", "missing field type");
  if (types.length > 1) return fail("unrecognized", "multiple field types");
  if (presence.length === 0) return fail("missing-presence", "missing required or optional");
  if (presence.length > 1) return fail("conflicting-presence", "conflicting presence");

  const leftover = tokens.filter((_, index) => !used[index]);
  const intent: FieldIntent = { type: types[0]!, required: presence[0]! };

  if (nullable) intent.nullable = true;

  if (leftover.length > 0) {
    const compared = parseComparison(leftover.join(" "));

    if (!compared.ok) {
      if (compared.code === "conflicting-bounds" || compared.code === "inverted-range") {
        return fail(compared.code, compared.code);
      }

      return fail("unrecognized", "unrecognized comparison");
    }

    const placed = placeComparison(intent.type, compared.comparison, compared.unit);

    if ("ok" in placed) return placed;
    if (placed.fields.minLength !== undefined) intent.minLength = placed.fields.minLength;
    if (placed.fields.maxLength !== undefined) intent.maxLength = placed.fields.maxLength;
    if (placed.fields.minimum !== undefined) intent.minimum = placed.fields.minimum;
    if (placed.fields.maximum !== undefined) intent.maximum = placed.fields.maximum;
  }

  const invalid = validateField(intent);

  if (invalid) {
    if (invalid.code === "conflicting-bounds") {
      return fail("conflicting-bounds", invalid.reason);
    }

    if (invalid.code === "wrong-slot") {
      return fail("unrecognized-slot", invalid.reason);
    }

    return fail("unrecognized", invalid.reason);
  }

  return succeed(intent);
}

export async function parse(text: string, options?: ParseOptions): Promise<ParseResult> {
  if (options?.source !== "rules") {
    const { parseWithModel } = await import("./model.js");

    return parseWithModel(text, options);
  }

  return parseWithRules(text);
}
