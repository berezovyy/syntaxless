import type { Bound, FieldIntent, FieldType } from "../types/intent.js";
import { validateField } from "./validate.js";

/**
 * JSON Schema is the portable compile target (the cron-string equivalent).
 * Zod/HTML5 are later compilers against the same AST.
 * Never invent `pattern` or `format` beyond the closed FieldType map.
 * `required` on a nameless field: string-like required ⇒ `minLength: 1`
 * (and format for email/url/uuid). Object-level `required` arrays wait for
 * multi-field schemas. v0 compile refuses `nullable`.
 */
export type JsonSchema = {
  type: "string" | "number" | "integer" | "boolean";
  format?: "email" | "uri" | "uuid";
  minLength?: number;
  maxLength?: number;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
};

export type CompileFailureCode =
  | "invalid-type"
  | "invalid-length"
  | "invalid-bound"
  | "conflicting-bounds"
  | "wrong-slot"
  | "nullable";

export type CompileResult =
  | { supported: true; schema: JsonSchema }
  | { supported: false; reason: string; code: CompileFailureCode };

const FORMAT_BY_TYPE: Partial<Record<FieldType, JsonSchema["format"]>> = {
  email: "email",
  url: "uri",
  uuid: "uuid",
};

const JSON_TYPE_BY_FIELD: Record<FieldType, JsonSchema["type"]> = {
  string: "string",
  email: "string",
  url: "string",
  uuid: "string",
  number: "number",
  integer: "integer",
  boolean: "boolean",
};

const STRING_LIKE = new Set<FieldType>(["string", "email", "url", "uuid"]);

function exclusiveIntegerMinimum(bound: Bound): number {
  return Math.floor(bound.value) + 1;
}

function exclusiveIntegerMaximum(bound: Bound): number {
  return Math.ceil(bound.value) - 1;
}

export function compile(intent: FieldIntent): CompileResult {
  const invalid = validateField(intent);

  if (invalid) return invalid;

  if (intent.nullable === true) {
    return {
      supported: false,
      reason: "nullable is not a v0 compile target",
      code: "nullable",
    };
  }

  const schema: JsonSchema = { type: JSON_TYPE_BY_FIELD[intent.type] };
  const format = FORMAT_BY_TYPE[intent.type];

  if (format !== undefined) schema.format = format;

  if (STRING_LIKE.has(intent.type)) {
    let minLength = intent.minLength;

    if (intent.required) {
      minLength = minLength === undefined ? 1 : Math.max(minLength, 1);
    }

    if (minLength !== undefined) schema.minLength = minLength;
    if (intent.maxLength !== undefined) schema.maxLength = intent.maxLength;
  }

  if (intent.minimum !== undefined) {
    if (intent.minimum.inclusive) {
      schema.minimum =
        intent.type === "integer" ? Math.ceil(intent.minimum.value) : intent.minimum.value;
    } else if (intent.type === "integer") {
      schema.minimum = exclusiveIntegerMinimum(intent.minimum);
    } else {
      schema.exclusiveMinimum = intent.minimum.value;
    }
  }

  if (intent.maximum !== undefined) {
    if (intent.maximum.inclusive) {
      schema.maximum =
        intent.type === "integer" ? Math.floor(intent.maximum.value) : intent.maximum.value;
    } else if (intent.type === "integer") {
      schema.maximum = exclusiveIntegerMaximum(intent.maximum);
    } else {
      schema.exclusiveMaximum = intent.maximum.value;
    }
  }

  if (
    intent.type === "integer" &&
    [schema.minimum, schema.maximum].some(
      (value) => value !== undefined && !Number.isSafeInteger(value),
    )
  ) {
    return {
      supported: false,
      code: "invalid-bound",
      reason: "integer bound widening exceeds the safe numeric range",
    };
  }

  if (
    schema.minimum !== undefined &&
    schema.maximum !== undefined &&
    schema.minimum > schema.maximum
  ) {
    return {
      supported: false,
      reason: "exclusive integer widening left an empty interval",
      code: "conflicting-bounds",
    };
  }

  return { supported: true, schema };
}

/** Assemble named field schemas so presence is enforced by the enclosing object. */
export interface ObjectSchema {
  type: "object";
  properties: Record<string, JsonSchema>;
  required?: string[];
}

export function compileObject(
  fields: Readonly<Record<string, FieldIntent>>,
):
  | { supported: true; schema: ObjectSchema }
  | { supported: false; code: CompileFailureCode; reason: string; field?: string } {
  if (typeof fields !== "object" || fields === null || Array.isArray(fields)) {
    return {
      supported: false,
      code: "invalid-type",
      reason: "fields must be a name-to-intent object",
    };
  }

  const properties: Record<string, JsonSchema> = Object.create(null);
  const required: string[] = [];

  for (const [name, intent] of Object.entries(fields)) {
    const result = compile(intent);

    if (!result.supported) return { ...result, field: name };
    properties[name] = result.schema;
    if (intent.required) required.push(name);
  }

  return {
    supported: true,
    schema: { type: "object", properties, ...(required.length ? { required } : {}) },
  };
}
