import { InputError } from "./types.js";
import type { Context, Field, SortIntent } from "./types.js";

export function catalog(context: Context): Map<string, Field> {
  if (
    !context ||
    !Array.isArray(context.fields) ||
    context.fields.length < 1 ||
    context.fields.length > 64
  ) {
    throw new InputError("invalid-catalog", "provide one to 64 typed fields");
  }

  const fields = new Map<string, Field>();
  const phrases = new Map<string, string>();

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
      (field.orderingAliases !== undefined &&
        (!Array.isArray(field.orderingAliases) || field.orderingAliases.length > 16))
    ) {
      throw new InputError("invalid-catalog", "too many aliases");
    }

    const aliases = [field.id, ...(field.aliases ?? [])].map((text) => ({
      text,
      meaning: JSON.stringify([field.id]),
    }));

    for (const alias of field.orderingAliases ?? []) {
      if (!alias || typeof alias !== "object" || !["asc", "desc"].includes(alias.direction)) {
        throw new InputError("invalid-catalog", "invalid ordering alias direction");
      }

      aliases.push({
        text: alias.text,
        meaning: JSON.stringify([field.id, alias.direction]),
      });
    }

    for (const { text, meaning } of aliases) {
      if (typeof text !== "string" || !text.trim() || text.length > 128) {
        throw new InputError("invalid-catalog", "alias must be nonempty");
      }

      const key = text.trim().toLowerCase();

      if (phrases.has(key) && phrases.get(key) !== meaning) {
        throw new InputError("invalid-catalog", "ambiguous catalog alias");
      }

      phrases.set(key, meaning);
    }

    fields.set(field.id, field);
  }

  if (context.defaultNulls !== undefined && !["first", "last"].includes(context.defaultNulls)) {
    throw new InputError("invalid-catalog", "invalid default null placement");
  }

  if (context.tieBreaker !== undefined && !fields.has(context.tieBreaker)) {
    throw new InputError("unknown-field", "unknown tie breaker");
  }

  return fields;
}

export function validate(intent: SortIntent, context: Context): Map<string, Field> {
  const fields = catalog(context);
  const seen = new Set<string>();

  if (
    !intent ||
    typeof intent !== "object" ||
    Object.keys(intent).some((k) => !["keys", "tieBreaker"].includes(k)) ||
    !Array.isArray(intent.keys) ||
    intent.keys.length < 1 ||
    intent.keys.length > 8
  ) {
    throw new InputError("invalid-intent", "provide one to eight sort keys");
  }

  for (const key of intent.keys) {
    if (
      !key ||
      Object.keys(key).some((k) => !["field", "direction", "nulls", "nullsSource"].includes(k)) ||
      !["asc", "desc"].includes(key.direction) ||
      !["first", "last"].includes(key.nulls) ||
      !["explicit", "caller-default"].includes(key.nullsSource)
    ) {
      throw new InputError("invalid-intent", "invalid sort key");
    }

    if (!fields.has(key.field)) {
      throw new InputError("unknown-field", `unknown field: ${key.field}`);
    }

    if (seen.has(key.field)) throw new InputError("conflict", "duplicate field ordering");
    seen.add(key.field);

    if (key.nullsSource === "caller-default" && key.nulls !== context.defaultNulls) {
      throw new InputError("invalid-intent", "default provenance does not match caller context");
    }
  }

  if (
    intent.tieBreaker !== undefined &&
    (!fields.has(intent.tieBreaker) || intent.tieBreaker !== context.tieBreaker)
  ) {
    throw new InputError("invalid-intent", "tie breaker must come from caller context");
  }

  return fields;
}
