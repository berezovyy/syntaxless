import { NUMBER_WORDS } from "../../../shared/numbers.js";

export { NUMBER_WORDS } from "../../../shared/numbers.js";
import type { FieldType } from "../types/intent.js";

export type ComparisonMeaning =
  | { side: "min"; inclusive: boolean }
  | { side: "max"; inclusive: boolean }
  | { side: "between" }
  | { side: "exact" };

export type Unit = "character";

export const TYPE_ALIASES: Record<string, FieldType> = {
  string: "string",
  text: "string",
  "text field": "string",
  "string field": "string",
  email: "email",
  "email address": "email",
  "email field": "email",
  "e-mail": "email",
  "e mail": "email",
  url: "url",
  "web address": "url",
  "url field": "url",
  uuid: "uuid",
  "uuid field": "uuid",
  number: "number",
  "numeric field": "number",
  integer: "integer",
  "whole number": "integer",
  "integer field": "integer",
  boolean: "boolean",
  "boolean field": "boolean",
};

/** `required` → true, `optional` / `not required` → false. Bare `not` is not a row. */
export const PRESENCE_ALIASES: Record<string, boolean> = {
  required: true,
  mandatory: true,
  compulsory: true,
  optional: false,
  "not mandatory": false,
  "not required": false,
};

/** Whole-token nullability. Absent from a sentence means the AST omits `nullable`. */
export const NULLABLE_ALIASES: Record<string, true> = {
  nullable: true,
};

/**
 * Inclusive vs exclusive is owned by the surface. `over`/`above`/`under`/`below`
 * still require a number at parse time; that is not a lexicon concern.
 */
export const COMPARISON_ALIASES: Record<string, ComparisonMeaning> = {
  "at least": { side: "min", inclusive: true },
  "no less than": { side: "min", inclusive: true },
  "not less than": { side: "min", inclusive: true },
  "no fewer than": { side: "min", inclusive: true },
  min: { side: "min", inclusive: true },
  "a minimum of": { side: "min", inclusive: true },
  minimum: { side: "min", inclusive: true },
  "minimum of": { side: "min", inclusive: true },
  ">=": { side: "min", inclusive: true },
  from: { side: "min", inclusive: true },
  "more than": { side: "min", inclusive: false },
  "greater than": { side: "min", inclusive: false },
  over: { side: "min", inclusive: false },
  above: { side: "min", inclusive: false },
  ">": { side: "min", inclusive: false },
  "at most": { side: "max", inclusive: true },
  "no more than": { side: "max", inclusive: true },
  "not more than": { side: "max", inclusive: true },
  max: { side: "max", inclusive: true },
  "a maximum of": { side: "max", inclusive: true },
  "capped at": { side: "max", inclusive: true },
  maximum: { side: "max", inclusive: true },
  "maximum of": { side: "max", inclusive: true },
  "up to": { side: "max", inclusive: true },
  "<=": { side: "max", inclusive: true },
  "less than": { side: "max", inclusive: false },
  "fewer than": { side: "max", inclusive: false },
  under: { side: "max", inclusive: false },
  below: { side: "max", inclusive: false },
  "<": { side: "max", inclusive: false },
  between: { side: "between" },
  exactly: { side: "exact" },
  "equal to": { side: "exact" },
};

export const UNIT_ALIASES: Record<string, Unit> = {
  character: "character",
  characters: "character",
  char: "character",
  chars: "character",
};

export interface AliasPhrase<T> {
  surface: string;
  value: T;
}

function phrasesFrom<T>(aliases: Record<string, T>): AliasPhrase<T>[] {
  return Object.entries(aliases)
    .map(([surface, value]) => ({ surface, value }))
    .sort((a, b) => b.surface.length - a.surface.length || a.surface.localeCompare(b.surface));
}

export const TYPE_PHRASES = phrasesFrom(TYPE_ALIASES);
export const PRESENCE_PHRASES = phrasesFrom(PRESENCE_ALIASES);
export const NULLABLE_PHRASES = phrasesFrom(NULLABLE_ALIASES);
export const COMPARISON_PHRASES = phrasesFrom(COMPARISON_ALIASES);
export const UNIT_PHRASES = phrasesFrom(UNIT_ALIASES);
export const NUMBER_PHRASES = phrasesFrom(NUMBER_WORDS);

/** Longest-first table scan. Surfaces are split on spaces; no discourse regex. */
export function matchLongestAlias<T>(
  tokens: readonly string[],
  index: number,
  phrases: readonly AliasPhrase<T>[],
): { value: T; tokenCount: number } | null {
  for (const phrase of phrases) {
    const parts = phrase.surface.split(" ");

    if (parts.length === 0 || index + parts.length > tokens.length) continue;
    let matched = true;

    for (let offset = 0; offset < parts.length; offset++) {
      if (tokens[index + offset] !== parts[offset]) {
        matched = false;
        break;
      }
    }

    if (matched) return { value: phrase.value, tokenCount: parts.length };
  }

  return null;
}
