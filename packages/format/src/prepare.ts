/** Preserve literal format values; the model assigns their instruction roles. */
import { NUMBER_WORDS } from "../../shared/numbers.js";
import { CURRENCIES, InputError, STYLES } from "./types.js";
import type { Style } from "./types.js";

export interface Lexeme {
  text: string;
  start: number;
  end: number;
  numeric?: string;
  currency?: string;
  locale?: string;
  timeZone?: string;
  style?: Style;
  hourCycle?: "h12" | "h23";
  display?: "symbol" | "code" | "name";
  marker?:
    | "number"
    | "percent"
    | "date"
    | "time"
    | "datetime"
    | "precision"
    | "grouping"
    | "parentheses"
    | "minus";
}

export interface Prepared {
  text: string;
  lexemes: readonly Lexeme[];
}

const names: Record<string, string> = {
  euros: "EUR",
  euro: "EUR",
  "US dollars": "USD",
  "US dollar": "USD",
  "United States dollars": "USD",
  "British pounds": "GBP",
  "British pound": "GBP",
  "Japanese yen": "JPY",
  yen: "JPY",
  "Canadian dollars": "CAD",
  "Canadian dollar": "CAD",
  "Australian dollars": "AUD",
  "Australian dollar": "AUD",
  "Swiss francs": "CHF",
  "Swiss franc": "CHF",
  "Chinese yuan": "CNY",
  yuan: "CNY",
  "€": "EUR",
};

const facts: Record<string, NonNullable<Lexeme["marker"]>> = {
  number: "number",
  numbers: "number",
  numeric: "number",
  percent: "percent",
  percentage: "percent",
  percentages: "percent",
  "%": "percent",
  date: "date",
  dates: "date",
  time: "time",
  times: "time",
  datetime: "datetime",
  decimals: "precision",
  decimal: "precision",
  places: "precision",
  place: "precision",
  grouping: "grouping",
  separators: "grouping",
  parentheses: "parentheses",
  parenthesis: "parentheses",
  minus: "minus",
};

const displays: Record<string, NonNullable<Lexeme["display"]>> = {
  symbol: "symbol",
  symbols: "symbol",
  code: "code",
  codes: "code",
  name: "name",
  names: "name",
};

const phrases = Object.entries(names).sort((a, b) => b[0].length - a[0].length);
const numbers = Object.entries(NUMBER_WORDS).sort((a, b) => b[0].length - a[0].length);

const hours: Record<string, "h12" | "h23"> = {
  "12-hour": "h12",
  "12 hour": "h12",
  "twelve-hour": "h12",
  "twelve hour": "h12",
  "24-hour": "h23",
  "24 hour": "h23",
  "twenty-four-hour": "h23",
  "twenty four hour": "h23",
  "twenty-four hour": "h23",
  "24h": "h23",
  "12h": "h12",
};

const boundary = (c: string | undefined) => !c || !/[\p{L}\p{N}_]/u.test(c);

export function prepare(input: string): Prepared {
  if (
    typeof input !== "string" ||
    !input.trim() ||
    input.length > 4096 ||
    /[\uD800-\uDFFF]/u.test(input) ||
    input.includes("\0")
  ) {
    throw new InputError("invalid-input", "provide valid Unicode text of one to 4096 characters");
  }

  const text = input
    .trim()
    .replace(/[.!]+$/u, "")
    .trim();

  const lower = text.toLowerCase();
  const lexemes: Lexeme[] = [];

  const emit = (word: string, fact: Partial<Omit<Lexeme, "text" | "start" | "end">> = {}) => {
    const start = lexemes.length ? lexemes.at(-1)!.end + 1 : 0;

    lexemes.push({ text: word, start, end: start + word.length, ...fact });
  };

  let offset = 0;

  const phrase = (entries: readonly (readonly [string, unknown])[]) =>
    entries.find(
      ([word]) =>
        boundary(text[offset - 1]) &&
        lower.startsWith(word.toLowerCase(), offset) &&
        boundary(text[offset + word.length]),
    );

  while (offset < text.length) {
    if (/\s/u.test(text[offset]!)) {
      offset++;
      continue;
    }

    const zoneWords = lower.slice(offset).match(/^time\s+zone(?![\p{L}\p{N}_])/u);

    if (zoneWords) {
      emit("timezone");
      offset += zoneWords[0].length;
      continue;
    }

    const hour = phrase(Object.entries(hours));

    if (hour) {
      emit("hourcyclevalue", { hourCycle: hour[1] as "h12" | "h23" });
      offset += hour[0].length;
      continue;
    }

    const named = phrase(phrases);

    if (named) {
      emit("currencyvalue", { currency: named[1] as string });
      offset += named[0].length;
      continue;
    }

    const sign = lower.slice(offset).match(/^(negative|minus|positive|plus)\s+/u);

    if (sign) {
      const after = offset + sign[0].length;

      const literalNumber = text
        .slice(after)
        .match(/^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![\d_.])/u)?.[0];

      const namedNumber = numbers.find(
        ([word]) => lower.startsWith(word, after) && boundary(text[after + word.length]),
      );

      if (literalNumber || namedNumber) {
        const prefix = sign[1] === "negative" || sign[1] === "minus" ? "-" : "+";

        emit("numbervalue", {
          numeric: prefix + (literalNumber ?? String(namedNumber![1])),
        });

        offset = after + (literalNumber?.length ?? namedNumber![0].length);
        continue;
      }
    }

    const number = phrase(numbers);

    if (number) {
      emit("numbervalue", { numeric: String(number[1]) });
      offset += number[0].length;
      continue;
    }

    const rest = text.slice(offset);
    const quoted = rest.match(/^(?:"([^"]+)"|'([^']+)')/u);

    const literal = quoted
      ? (quoted[1] ?? quoted[2])!
      : rest.match(
          /^(?:[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})+|[A-Za-z_]+(?:\/[A-Za-z0-9_+.-]+)+|UTC|GMT)(?![\p{L}\p{N}_])/iu,
        )?.[0];

    if (literal) {
      if (lexemes.at(-1)?.text === "locale") {
        emit("localevalue", { locale: literal });
      } else if (["timezone", "zone"].includes(lexemes.at(-1)?.text ?? "")) {
        emit("timezonevalue", { timeZone: literal });
      } else if ((CURRENCIES as readonly string[]).includes(literal.toUpperCase())) {
        emit("currencyvalue", { currency: literal.toUpperCase() });
      } else if (literal.includes("/") || /^(utc|gmt)$/iu.test(literal)) {
        emit("timezonevalue", { timeZone: literal });
      } else {
        emit("localevalue", { locale: literal });
      }

      offset += quoted ? quoted[0].length : literal.length;
      continue;
    }

    const numeric = rest.match(/^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![\d_.])/u)?.[0];

    if (numeric) {
      emit("numbervalue", { numeric });
      offset += numeric.length;
      continue;
    }

    const word = rest.match(/^[\p{L}_]+(?:'\p{L}+)*/u)?.[0];

    if (word) {
      const normalized = word.toLowerCase();
      const upper = word.toUpperCase();

      if (lexemes.at(-1)?.text === "locale" && /^[a-z]{2,3}$/u.test(normalized)) {
        emit("localevalue", { locale: word });
      } else if (
        (CURRENCIES as readonly string[]).includes(upper) ||
        (/^[A-Z]{3}$/u.test(word) &&
          ![
            "THE",
            "FOR",
            "AND",
            "NOT",
            "USE",
            "SET",
            "CAN",
            "YOU",
            "ARE",
            "NOW",
            "ALL",
            "LET",
            "ADD",
            "OUR",
            "ANY",
            "GET",
            "BUT",
            "OUT",
          ].includes(word))
      ) {
        emit("currencyvalue", { currency: upper });
      } else if ((STYLES as readonly string[]).includes(normalized)) {
        emit(normalized, { style: normalized as Style });
      } else if (Object.hasOwn(facts, normalized)) {
        emit(normalized === "place" ? "places" : normalized, {
          marker: facts[normalized]!,
        });
      } else if (Object.hasOwn(displays, normalized)) {
        emit(normalized, { display: displays[normalized]! });
      } else {
        emit(normalized);
      }

      offset += word.length;
      continue;
    }

    if (text[offset] === "%") emit("percent", { marker: "percent" });
    else emit(text[offset]!);
    offset++;
  }

  if (lexemes.length > 64) throw new InputError("invalid-input", "input exceeds 64 lexemes");

  return { text: lexemes.map((l) => l.text).join(" "), lexemes };
}
