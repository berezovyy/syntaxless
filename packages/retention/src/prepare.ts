import { NUMBER_WORDS } from "../../shared/numbers.js";
import { validScalar } from "../../shared/values.js";
import { InputError } from "./types.js";
import type { Bucket } from "./types.js";
import type { Unit } from "./units.js";

export interface Lexeme {
  text: string;
  start: number;
  end: number;
  numeric?: string;
  unit?: Unit;
  bucket?: Bucket;
  rolling?: true;
  calendar?: true;
}

export interface Prepared {
  text: string;
  lexemes: readonly Lexeme[];
}

const aliases: Record<string, Unit> = {
  ms: "millisecond",
  millisecond: "millisecond",
  milliseconds: "millisecond",
  s: "second",
  sec: "second",
  seconds: "second",
  second: "second",
  min: "minute",
  minute: "minute",
  minutes: "minute",
  h: "hour",
  hour: "hour",
  hours: "hour",
  day: "day",
  days: "day",
  week: "week",
  weeks: "week",
  month: "month",
  months: "month",
  year: "year",
  years: "year",
};

const buckets: Record<string, Bucket> = {
  daily: "day",
  weekly: "week",
  monthly: "month",
  yearly: "year",
  annual: "year",
  annually: "year",
};

const words = Object.entries(NUMBER_WORDS).sort((a, b) => b[0].length - a[0].length);

export function prepare(text: string, _context: unknown = {}): Prepared {
  if (
    typeof text !== "string" ||
    !text.trim() ||
    text.length > 4096 ||
    !validScalar(text, "string")
  ) {
    throw new InputError("invalid-input", "provide one to 4096 valid text characters");
  }

  text = text
    .trim()
    .toLowerCase()
    .replace(/[.!]+$/u, "")
    .trim();

  const lexemes: Lexeme[] = [];

  const emit = (word: string, facts: Omit<Lexeme, "text" | "start" | "end"> = {}) => {
    const start = lexemes.length ? lexemes.at(-1)!.end + 1 : 0;

    lexemes.push({ text: word, start, end: start + word.length, ...facts });
  };

  const boundary = (c: string | undefined) => !c || !/[\p{L}\p{N}_]/u.test(c);
  let offset = 0;

  while (offset < text.length) {
    if (/\s/u.test(text[offset]!)) {
      offset++;
      continue;
    }

    const numeric = text.slice(offset).match(/^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![\d_.])/u)?.[0];

    if (numeric) {
      emit("numbervalue", {
        numeric: numeric.startsWith(".") ? "0" + numeric : numeric,
      });

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
      const next = text.slice(offset + word.length).match(/^\s+([a-z]+)/u)?.[1];

      if ((word === "a" || word === "an") && next && Object.hasOwn(aliases, next)) {
        emit("numbervalue", { numeric: "1" });
      } else if (Object.hasOwn(aliases, word)) {
        const unit = aliases[word]!;

        emit(unit + "unit", { unit });
      } else if (Object.hasOwn(buckets, word)) {
        emit(word, { bucket: buckets[word]! });
      } else if (["last", "past", "elapsed", "rolling"].includes(word)) {
        emit(word, { rolling: true });
      } else if (word === "calendar") {
        emit(word, { calendar: true });
      } else {
        emit(word);
      }

      offset += word.length;
    } else {
      emit(text[offset]!);
      offset++;
    }
  }

  if (lexemes.length > 64) throw new InputError("invalid-input", "input exceeds 64 lexemes");

  return { text: lexemes.map((l) => l.text).join(" "), lexemes };
}
