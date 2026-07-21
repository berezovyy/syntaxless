import type { Bound, ComparisonIntent, FieldIntent, FieldType } from "../types/intent.js";

export type ValidateFailureCode =
  "invalid-type" | "invalid-length" | "invalid-bound" | "conflicting-bounds" | "wrong-slot";

export type Failure = {
  supported: false;
  reason: string;
  code: ValidateFailureCode;
};

const FIELD_TYPES = new Set<FieldType>([
  "string",
  "email",
  "url",
  "uuid",
  "number",
  "integer",
  "boolean",
]);

const STRING_LIKE = new Set<FieldType>(["string", "email", "url", "uuid"]);
const NUMBER_LIKE = new Set<FieldType>(["number", "integer"]);

export function unsupported(code: ValidateFailureCode, reason: string): Failure {
  return { supported: false, reason, code };
}

function isFiniteNumber(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value) &&
    Math.abs(value) <= Number.MAX_SAFE_INTEGER
  );
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function validateBound(bound: Bound, side: "min" | "max"): Failure | null {
  if (typeof bound !== "object" || bound === null) {
    return unsupported("invalid-bound", `${side} must be a Bound object`);
  }

  if (!isFiniteNumber(bound.value)) {
    return unsupported("invalid-bound", `${side}.value must be a finite number`);
  }

  if (typeof bound.inclusive !== "boolean") {
    return unsupported("invalid-bound", `${side}.inclusive must be a boolean`);
  }

  return null;
}

function validateLength(value: number, slot: "minLength" | "maxLength"): Failure | null {
  if (!isNonNegativeInteger(value)) {
    return unsupported("invalid-length", `${slot} must be an integer >= 0`);
  }

  return null;
}

/**
 * Interval emptiness is evaluated on the AST as written. Exclusive bounds are
 * not widened here (that is compile math). An exclusive bound still occupies
 * its value, so min 10 exclusive and max 10 inclusive is empty.
 */
export function validateComparison(intent: ComparisonIntent): Failure | null {
  if (typeof intent !== "object" || intent === null) {
    return unsupported("invalid-bound", "comparison must be an object");
  }

  const min = intent.min;
  const max = intent.max;

  if (min !== undefined) {
    const failure = validateBound(min, "min");

    if (failure) return failure;
  }

  if (max !== undefined) {
    const failure = validateBound(max, "max");

    if (failure) return failure;
  }

  if (min !== undefined && max !== undefined) {
    if (min.value > max.value) {
      return unsupported("conflicting-bounds", "min is greater than max");
    }

    if (min.value === max.value && (!min.inclusive || !max.inclusive)) {
      return unsupported("conflicting-bounds", "exclusive bounds leave an empty interval");
    }
  }

  return null;
}

export function validateField(intent: FieldIntent): Failure | null {
  if (typeof intent !== "object" || intent === null) {
    return unsupported("invalid-type", "intent must be an object");
  }

  const knownKeys = new Set([
    "type",
    "required",
    "nullable",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
  ]);

  if (Object.keys(intent).some((key) => !knownKeys.has(key))) {
    return unsupported("invalid-type", "unknown intent property");
  }

  if (typeof intent.type !== "string" || !FIELD_TYPES.has(intent.type)) {
    return unsupported(
      "invalid-type",
      `unknown type: ${String((intent as { type?: unknown }).type)}`,
    );
  }

  if (typeof intent.required !== "boolean") {
    return unsupported("invalid-type", "required must be a boolean");
  }

  if (intent.nullable !== undefined && typeof intent.nullable !== "boolean") {
    return unsupported("invalid-type", "nullable must be a boolean");
  }

  const hasLength = intent.minLength !== undefined || intent.maxLength !== undefined;
  const hasValue = intent.minimum !== undefined || intent.maximum !== undefined;

  if (hasLength && !STRING_LIKE.has(intent.type)) {
    return unsupported("wrong-slot", "length constraints are only valid on string-like types");
  }

  if (hasValue && !NUMBER_LIKE.has(intent.type)) {
    return unsupported("wrong-slot", "value constraints are only valid on number-like types");
  }

  if (intent.minLength !== undefined) {
    const failure = validateLength(intent.minLength, "minLength");

    if (failure) return failure;
  }

  if (intent.maxLength !== undefined) {
    const failure = validateLength(intent.maxLength, "maxLength");

    if (failure) return failure;
  }

  if (
    intent.minLength !== undefined &&
    intent.maxLength !== undefined &&
    intent.minLength > intent.maxLength
  ) {
    return unsupported("conflicting-bounds", "minLength is greater than maxLength");
  }

  if (intent.required && STRING_LIKE.has(intent.type) && intent.maxLength === 0) {
    return unsupported(
      "conflicting-bounds",
      "a required string must contain at least one character",
    );
  }

  const comparison: ComparisonIntent = {};

  if (intent.minimum !== undefined) comparison.min = intent.minimum;
  if (intent.maximum !== undefined) comparison.max = intent.maximum;

  return validateComparison(comparison);
}
