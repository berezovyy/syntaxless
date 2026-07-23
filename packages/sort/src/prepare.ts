/** Canonicalize caller facts, without interpreting sorting words. */
import { catalog } from "./catalog.js";
import { InputError } from "./types.js";
import type { Context, Direction, Field } from "./types.js";

export interface Lexeme {
  text: string;
  start: number;
  end: number;
  field?: string;
  direction?: Direction;
}

export interface Prepared {
  text: string;
  lexemes: readonly Lexeme[];
}

export function prepare(text: string, context: Context): Prepared {
  const fields = catalog(context);

  if (typeof text !== "string" || !text.trim() || text.length > 4096) {
    throw new InputError("invalid-input", "provide one to 4096 characters");
  }

  text = text
    .trim()
    .replace(/[.!]+$/u, "")
    .trim();

  const phrases: { text: string; field: Field; direction?: Direction }[] = [];

  for (const field of fields.values()) {
    for (const alias of new Set([field.id, ...(field.aliases ?? [])])) {
      phrases.push({ text: alias.trim(), field });
    }

    for (const alias of field.orderingAliases ?? []) {
      phrases.push({ text: alias.text.trim(), field, direction: alias.direction });
    }
  }

  phrases.sort((a, b) => b.text.length - a.text.length);
  const lexemes: Lexeme[] = [];

  function emit(word: string, facts: { field?: string; direction?: Direction } = {}): void {
    const start = lexemes.length ? lexemes.at(-1)!.end + 1 : 0;

    lexemes.push({ text: word, start, end: start + word.length, ...facts });
  }

  const boundary = (c: string | undefined) => !c || !/[\p{L}\p{N}_]/u.test(c);
  let offset = 0;

  while (offset < text.length) {
    if (/\s/u.test(text[offset]!)) {
      offset++;
      continue;
    }

    const phrase = phrases.find(
      (entry) =>
        boundary(text[offset - 1]) &&
        text.slice(offset, offset + entry.text.length).toLowerCase() === entry.text.toLowerCase() &&
        boundary(text[offset + entry.text.length]),
    );

    if (phrase) {
      emit(`${phrase.field.type}field`, { field: phrase.field.id });

      if (phrase.direction) {
        emit(phrase.direction === "asc" ? "ascending" : "descending", {
          direction: phrase.direction,
        });
      }

      offset += phrase.text.length;
      continue;
    }

    const word = text.slice(offset).match(/^[\p{L}_]+(?:['’][\p{L}]+)*/u)?.[0];

    emit(word?.toLowerCase() ?? text[offset]!);
    offset += word?.length ?? 1;
  }

  if (lexemes.length > 128) throw new InputError("invalid-input", "input exceeds 128 lexemes");

  return { text: lexemes.map((word) => word.text).join(" "), lexemes };
}
