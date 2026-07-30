/** Resolve only explicit caller aliases and exact literals; operators belong to the model. */
import { catalog } from "./catalog.js";
import { InputError } from "./types.js";
import type { Context, Field, Scalar } from "./types.js";

export interface Lexeme {
  text: string;
  start: number;
  end: number;
  field?: string;
  value?: Scalar;
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

  // Drop neutral sentence endings; questions and quoted literals stay intact.
  text = text
    .trim()
    .replace(/[.!]+$/u, "")
    .trim();

  const phrases: { text: string; field: Field; value?: Scalar }[] = [];

  for (const field of fields.values()) {
    for (const alias of new Set([field.id, ...(field.aliases ?? [])])) {
      phrases.push({ text: alias.trim(), field });
    }

    for (const term of field.terms ?? []) {
      phrases.push({ text: term.text.trim(), field, value: term.value });
    }
  }

  phrases.sort((left, right) => right.text.length - left.text.length);
  const words: Lexeme[] = [];
  let offset = 0;

  function emit(word: string, facts: { field?: string; value?: Scalar } = {}): void {
    const start = words.length ? words.at(-1)!.end + 1 : 0;

    words.push({ text: word, start, end: start + word.length, ...facts });
  }

  const boundary = (character: string | undefined) =>
    !character || !/[\p{L}\p{N}_]/u.test(character);

  while (offset < text.length) {
    if (/\s/u.test(text[offset]!)) {
      offset++;
      continue;
    }

    if (text[offset] === '"') {
      const literal = text.slice(offset).match(/^"(?:[^"\\]|\\.)*"/u)?.[0];

      if (!literal) throw new InputError("invalid-input", "unterminated string literal");
      let value: unknown;

      try {
        value = JSON.parse(literal);
      } catch {
        throw new InputError("invalid-input", "use JSON string escaping");
      }

      if (typeof value !== "string") throw new InputError("invalid-input", "invalid quoted value");
      emit("textvalue", { value });
      offset += literal.length;
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

      if (phrase.value !== undefined) {
        emit("equals");

        emit(`${typeof phrase.value === "string" ? "text" : typeof phrase.value}value`, {
          value: phrase.value,
        });
      }

      offset += phrase.text.length;
      continue;
    }

    const number = text
      .slice(offset)
      .match(/^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![\p{L}\p{N}_.])/u)?.[0];

    if (number) {
      emit("numbervalue", { value: Number(number) });
      offset += number.length;
      continue;
    }

    const word = text.slice(offset).match(/^[\p{L}_]+(?:['’][\p{L}]+)*/u)?.[0];

    if (word) {
      if (word.toLowerCase() === "true" || word.toLowerCase() === "false") {
        emit("booleanvalue", { value: word.toLowerCase() === "true" });
      } else {
        emit(word.toLowerCase());
      }

      offset += word.length;
      continue;
    }

    emit(text[offset]!);
    offset++;
  }

  return { text: words.map((word) => word.text).join(" "), lexemes: words };
}
