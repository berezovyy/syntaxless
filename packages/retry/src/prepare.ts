/** Preserve exact decimal and unit facts, leaving their policy role to the model. */
import { NUMBER_WORDS } from "../../shared/numbers.js";
import { InputError } from "./types.js";
import type { Context } from "./types.js";
import { validateContext } from "./validate.js";

export interface Lexeme {
  text: string;
  start: number;
  end: number;
  numeric?: string;
  unitMs?: number;
  percent?: boolean;
  multiplier?: 2 | 3;
}

export interface Prepared {
  text: string;
  lexemes: readonly Lexeme[];
}

const units: Record<string, number> = {
  ms: 1,
  millisecond: 1,
  milliseconds: 1,
  s: 1000,
  sec: 1000,
  secs: 1000,
  second: 1000,
  seconds: 1000,
  m: 60000,
  min: 60000,
  mins: 60000,
  minute: 60000,
  minutes: 60000,
  h: 3600000,
  hr: 3600000,
  hrs: 3600000,
  hour: 3600000,
  hours: 3600000,
  day: 86400000,
  days: 86400000,
};

const words = Object.entries({ ...NUMBER_WORDS, once: 1, twice: 2 }).sort(
  (a, b) => b[0].length - a[0].length,
);

export function prepare(text: string, context: Context): Prepared {
  validateContext(context);

  if (typeof text !== "string" || !text.trim() || text.length > 4096) {
    throw new InputError("invalid-input", "provide one to 4096 characters");
  }

  text = text
    .trim()
    .replace(/[.!]+$/u, "")
    .trim()
    .toLowerCase();

  const lexemes: Lexeme[] = [];

  function emit(
    word: string,
    facts: {
      numeric?: string;
      unitMs?: number;
      percent?: boolean;
      multiplier?: 2 | 3;
    } = {},
  ) {
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

    const numeric = text.slice(offset).match(/^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![\d_.])/u)?.[0];

    if (numeric) {
      emit("numbervalue", { numeric });
      offset += numeric.length;
      continue;
    }

    const phrase = words.find(
      ([word]) =>
        boundary(text[offset - 1]) &&
        text.startsWith(word, offset) &&
        boundary(text[offset + word.length]),
    );

    if (phrase) {
      emit("numbervalue", { numeric: String(phrase[1]) });
      offset += phrase[0].length;
      continue;
    }

    const word = text.slice(offset).match(/^[\p{L}_]+(?:['’][\p{L}]+)*/u)?.[0];

    if (word) {
      const unitMs = Object.hasOwn(units, word) ? units[word] : undefined;

      if (unitMs !== undefined) {
        emit("timeunit", { unitMs });
      } else if (["percent", "percentage"].includes(word)) {
        emit("percentunit", { percent: true });
      } else if (word === "double" || word === "triple") {
        emit(word, { multiplier: word === "double" ? 2 : 3 });
      } else {
        emit(word);
      }

      offset += word.length;
    } else {
      if (text[offset] === "%") emit("percentunit", { percent: true });
      else emit(text[offset]!);
      offset++;
    }
  }

  if (lexemes.length > 64) throw new InputError("invalid-input", "input exceeds 64 lexemes");

  return { text: lexemes.map((l) => l.text).join(" "), lexemes };
}
