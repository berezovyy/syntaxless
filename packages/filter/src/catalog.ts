import { validScalar } from "../../shared/values.js";
import { InputError } from "./types.js";
import type { Context, Field, Scalar } from "./types.js";

export function scalar(value: unknown, field: Field): value is Scalar {
  return validScalar(value, field.type);
}

export function catalog(context: Context): Map<string, Field> {
  if (
    !context ||
    !Array.isArray(context.fields) ||
    context.fields.length < 1 ||
    context.fields.length > 64
  ) {
    throw new InputError("invalid-catalog", "provide between one and 64 fields");
  }

  const fields = new Map<string, Field>();
  const names = new Map<string, string>();

  for (const field of context.fields as readonly Field[]) {
    if (
      !field ||
      typeof field.id !== "string" ||
      !field.id.trim() ||
      field.id.length > 128 ||
      fields.has(field.id) ||
      !["string", "number", "boolean"].includes(field.type)
    ) {
      throw new InputError("invalid-catalog", "field identifiers must be unique and typed");
    }

    if (
      (field.aliases !== undefined &&
        (!Array.isArray(field.aliases) || field.aliases.length > 16)) ||
      (field.terms !== undefined && (!Array.isArray(field.terms) || field.terms.length > 32))
    ) {
      throw new InputError("invalid-catalog", "catalog aliases or terms exceed their bounds");
    }

    if (
      field.operators !== undefined &&
      (!Array.isArray(field.operators) ||
        field.operators.some(
          (op) => !["eq", "ne", "lt", "le", "gt", "ge", "in", "range", "null"].includes(op),
        ))
    ) {
      throw new InputError("invalid-catalog", "unknown allowed operator");
    }

    for (const term of field.terms ?? []) {
      if (!term || !scalar(term.value, field)) {
        throw new InputError("invalid-catalog", "shorthand value must match its field type");
      }
    }

    const phrases = [
      ...[field.id, ...(field.aliases ?? [])].map((text) => ({
        text,
        meaning: field.id,
      })),
      ...(field.terms ?? []).map((term) => ({
        text: term.text,
        meaning: JSON.stringify([field.id, term.value]),
      })),
    ];

    for (const { text: name, meaning } of phrases) {
      if (typeof name !== "string" || !name.trim() || name.length > 128) {
        throw new InputError("invalid-catalog", "catalog phrases must be nonempty strings");
      }

      const key = name.trim().toLowerCase();

      if (names.has(key) && names.get(key) !== meaning) {
        throw new InputError("invalid-catalog", `duplicate catalog phrase: ${name}`);
      }

      names.set(key, meaning);
    }

    fields.set(field.id, field);
  }

  return fields;
}
