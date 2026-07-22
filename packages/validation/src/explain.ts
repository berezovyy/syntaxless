import { validateField } from "./compiler/validate.js";
import type { FieldIntent } from "./types/intent.js";

/** Describe the typed facts, including the required-string convention. */
export function explain(intent: FieldIntent): string {
  const invalid = validateField(intent);

  if (invalid) throw new TypeError(invalid.reason);
  const clauses = [`${intent.required ? "Required" : "Optional"} ${intent.type} field`];

  if (intent.nullable !== undefined) {
    clauses.push(intent.nullable ? "allows null (compilation unsupported)" : "does not allow null");
  }

  const minLength =
    intent.required && ["string", "email", "url", "uuid"].includes(intent.type)
      ? Math.max(1, intent.minLength ?? 0)
      : intent.minLength;

  if (minLength !== undefined) {
    const characters = minLength === 1 ? "character" : "characters";
    const isDefault = intent.required && (intent.minLength ?? 0) < 1;
    const origin = isDefault ? " (required-string default)" : "";

    clauses.push(`at least ${minLength} ${characters}${origin}`);
  }

  if (intent.maxLength !== undefined) {
    clauses.push(
      `at most ${intent.maxLength} ${intent.maxLength === 1 ? "character" : "characters"}`,
    );
  }

  if (intent.minimum !== undefined) {
    clauses.push(
      `${intent.minimum.inclusive ? "at least" : "greater than"} ${intent.minimum.value}`,
    );
  }

  if (intent.maximum !== undefined) {
    clauses.push(`${intent.maximum.inclusive ? "at most" : "less than"} ${intent.maximum.value}`);
  }

  return `${clauses.join("; ")}.`;
}
