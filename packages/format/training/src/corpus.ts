/** Bounded sampling of format instructions with explicit literal and role supervision. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";

import { writeJsonl } from "../../../shared/training/jsonl.js";
import { reservedHashes } from "../../../shared/training/reserved.js";
import type { Role } from "../../src/labels.js";
import { prepare } from "../../src/prepare.js";
import { resolveRoles } from "../../src/resolve.js";
import { CURRENCIES, STYLES } from "../../src/types.js";
import type { Context, DateTimeIntent, FormatIntent, NumericIntent } from "../../src/types.js";

export const FAMILIES = [
  "number",
  "currency",
  "percent",
  "precision-range",
  "grouping-negative",
  "currency-display",
  "date",
  "time",
  "date-time",
  "locale-timezone",
] as const;

export interface Row {
  text: string;
  context: Context;
  intent: FormatIntent | null;
  roles: readonly Role[];
  family: string;
  split: "train" | "validation" | "heldout";
  fingerprint: string;
  trainingEligible?: boolean;
}

/** Fixed sampling seed; changing it changes every generated corpus. */
const SEED = 20261012;

export const hash = (bytes: string | Uint8Array) =>
  createHash("sha256").update(bytes).digest("hex");

export const readRows = (path: string): Row[] =>
  readFileSync(path, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));

/** Group value substitutions and their negative variants before assigning a split. */
export function constructionFingerprint(text: string, roles: readonly Role[]): string {
  return hash(
    JSON.stringify(
      prepare(text).lexemes.map((lexeme, i) => [
        lexeme.numeric !== undefined
          ? "<number>"
          : lexeme.currency !== undefined
            ? "<currency>"
            : lexeme.locale !== undefined
              ? "<locale>"
              : lexeme.timeZone !== undefined
                ? "<timezone>"
                : lexeme.style !== undefined
                  ? "<style>"
                  : lexeme.hourCycle !== undefined
                    ? "<hour-cycle>"
                    : lexeme.display !== undefined
                      ? "<display>"
                      : lexeme.text,
        roles[i],
      ]),
    ),
  );
}

const markerRoles = {
  number: "NUMBER",
  percent: "PERCENT",
  date: "DATE",
  time: "TIME",
  datetime: "DATETIME",
  precision: "PRECISION",
  grouping: "GROUP",
  parentheses: "PARENS",
  minus: "MINUS",
} as const;

const wordCounts = [
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "eleven",
  "twelve",
  "thirteen",
  "fourteen",
  "fifteen",
  "sixteen",
  "seventeen",
  "eighteen",
  "nineteen",
  "twenty",
];

const currencyNames = {
  EUR: ["EUR", "euros", "euro", "€"],
  USD: ["USD", "US dollars", "United States dollars"],
  GBP: ["GBP", "British pounds"],
  JPY: ["JPY", "Japanese yen", "yen"],
  CAD: ["CAD", "Canadian dollars"],
  AUD: ["AUD", "Australian dollars"],
  CHF: ["CHF", "Swiss francs"],
  CNY: ["CNY", "Chinese yuan", "yuan"],
};

function* sample(
  count = 8000,
  seed = SEED,
  excluded: ReadonlySet<string> = new Set(),
): Generator<Row> {
  if (!Number.isSafeInteger(count) || count < 1 || count > 100000) {
    throw Error("corpus row count is outside the declared budget");
  }

  let state = seed;

  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;

    return state / 2 ** 32;
  };

  const pick = <T>(xs: readonly T[]): T => xs[Math.floor(random() * xs.length)]!;
  let emitted = 0;
  const seen = new Map<string, string>();

  for (let attempt = 0; emitted < count; attempt++) {
    if (attempt > count * 100) throw Error("corpus deduplication exhausted");

    const family = pick(FAMILIES);
    const positive = random() < 0.67;

    let text = "";
    let roles: Role[] = [];

    const add = (piece: string, role: Role = "GLUE") => {
      if (!piece) return;
      text += (text ? " " : "") + piece;

      roles.push(
        ...prepare(piece).lexemes.map((l) => {
          if (role === "UNSUPPORTED") return role;
          if (l.numeric !== undefined) return role;
          if (l.currency !== undefined) return "CURRENCY";
          if (l.locale !== undefined) return "LOCALE";
          if (l.timeZone !== undefined) return "ZONE";
          if (l.style !== undefined) return role;
          if (l.hourCycle !== undefined) return "HOUR";
          if (l.display !== undefined) return "DISPLAY";

          if (
            role === "DEFAULT_PRECISION" &&
            ["default", "defaults", "native", "overrides"].includes(l.text)
          ) {
            return role;
          }

          if (l.marker) {
            return l.marker === "grouping" && role === "NOGROUP" ? role : markerRoles[l.marker];
          }

          return "GLUE";
        }),
      );
    };

    add(
      pick([
        "",
        "please",
        "please format",
        "format",
        "display",
        "render",
        "show",
        "I would like you to format",
        "I need you to display",
        "I want to show",
        "use",
        "configure the output as",
        "set the presentation to",
        "for the report, display",
        "for the invoice, show",
        "for the table, use",
        "for the total, show",
      ]),
    );

    if (random() < 0.55) {
      add(pick(["please", "I want you to", "I need you to", "I would like you to", "can you"]));
      add(pick(["format", "display", "show", "render", "configure", "arrange", "present", "set"]));

      add(
        pick([
          "the output",
          "the result",
          "the presentation",
          "the displayed result",
          "the rendered output",
          "the formatted value",
          "the output presentation",
        ]),
      );

      add(pick(["as", "to use", "using", "with"]));
    }

    if (random() < 0.45) {
      const document = pick([
        "report",
        "invoice",
        "table",
        "total",
        "summary",
        "statement",
        "receipt",
        "record",
      ]);

      const adjective = pick(["", "exported", "printed", "downloaded", "displayed", "rendered"]);

      const preposition = pick(["for", "within", "in"]);
      const verb = pick(["show", "display", "format", "use", "render"]);

      add(`${preposition} the ${adjective} ${document}, ${verb}`.replace(/ +/g, " "));
    }

    let gold: FormatIntent;
    let kind: NumericIntent["kind"] | "date" | "time" | "date-time";

    if (["date", "time", "date-time"].includes(family)) {
      kind = family as "date" | "time" | "date-time";
    } else if (["currency-display", "currency"].includes(family)) {
      kind = "currency";
    } else if (family === "number" || family === "percent") {
      kind = family;
    } else if (family === "locale-timezone") {
      kind = pick(["number", "currency", "percent", "date", "time", "date-time"]);
    } else {
      kind = pick(["number", "currency", "percent"]);
    }

    if (kind === "date" || kind === "time" || kind === "date-time") {
      const intent: DateTimeIntent = { kind: "date-time" };

      const hasDate = kind !== "time";
      const hasTime = kind !== "date";

      if (hasDate && hasTime && random() < 0.3) {
        const style = pick(STYLES);

        add(style, "STYLE");
        add(kind === "date-time" ? pick(["datetime", "date and time"]) : kind);
        if (hasDate) intent.dateStyle = style;
        if (hasTime) intent.timeStyle = style;
      } else {
        // A shared style and a separate style cannot label the same wording differently.
        const named = random() < 0.8;

        if (hasDate) {
          const style = pick(STYLES);

          if (named) add(style, "DSTYLE");
          add(pick(["date", "dates"]));
          intent.dateStyle = named ? style : "medium";
        }

        if (hasTime) {
          if (hasDate) add(pick(["and", ",", "with"]));
          const style = pick(STYLES);

          if (named) add(style, "TSTYLE");
          add(pick(["time", "times"]));
          intent.timeStyle = named ? style : "medium";
        }
      }

      if (hasTime && random() < 0.6) {
        intent.hourCycle = pick(["h12", "h23"]);
        add(pick(["with", "using", "in", "using a", "with a"]));

        add(
          intent.hourCycle === "h12"
            ? pick(["12-hour", "twelve hour", "12h"])
            : pick(["24-hour", "twenty-four-hour", "24h"]),
        );

        add(pick(["clock", "time"]));
      }

      if (family === "locale-timezone" || random() < 0.85) {
        intent.timeZone = pick(["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"]);
      }

      gold = intent;
    } else {
      const intent: NumericIntent =
        kind === "currency"
          ? {
              kind,
              currency: pick(CURRENCIES),
              currencyDisplay: "symbol",
              negative: "minus",
            }
          : { kind, negative: "minus" };

      if (intent.kind === "currency") {
        add(pick(currencyNames[intent.currency]));
      } else {
        add(
          intent.kind === "number"
            ? pick(["number", "numbers", "numeric output"])
            : pick(["percent", "percentage", "percentages", "%"]),
        );
      }

      if (family === "precision-range") {
        const min = Math.floor(random() * 11);
        const max = min + Math.floor(random() * 11);

        intent.precision = { minimum: min, maximum: max };
        add(pick(["with at least", "with a minimum of", "minimum"]));
        add(random() < 0.5 ? String(min) : wordCounts[min]!, "MIN");
        add(pick(["decimal places", "decimals"]));
        add(pick(["and at most", "and a maximum of", "maximum"]));
        add(random() < 0.5 ? String(max) : wordCounts[max]!, "MAX");
        add(pick(["decimal places", "decimals"]));
      } else if (random() < 0.8) {
        const n = Math.floor(random() * 21);

        intent.precision = { minimum: n, maximum: n };

        add(
          pick([
            "with",
            ",",
            "to",
            "using",
            "showing",
            "with exactly",
            "using exactly",
            "showing exactly",
          ]),
        );

        add(random() < 0.4 ? wordCounts[n]! : String(n), "EXACT");
        add(pick(["decimals", "decimal places"]));
      } else if (random() < 0.8) {
        add(
          pick([
            "using native precision",
            "with default decimals",
            "without decimal precision overrides",
            ...(intent.kind === "currency" ? ["use currency defaults"] : []),
          ]),
          "DEFAULT_PRECISION",
        );
      }

      if (family === "grouping-negative" || random() < 0.4) {
        intent.grouping = random() < 0.5;
        add(pick(["and", ",", "with"]));

        if (intent.grouping) {
          add(pick(["grouping", "grouping separators"]));
        } else {
          add(pick(["no", "without", "disable"]));
          add(pick(["grouping", "grouping separators"]), "NOGROUP");
        }
      }

      if (family === "grouping-negative" || random() < 0.6) {
        intent.negative = random() < 0.5 ? "minus" : "parentheses";
        add(pick(["and", ","]));

        add(
          intent.negative === "parentheses"
            ? pick([
                "negatives in parentheses",
                "parentheses for negatives",
                "use parentheses for negative values",
              ])
            : pick([
                "negatives with minus signs",
                "use minus signs for negatives",
                "minus for negative values",
              ]),
        );
      }

      if (intent.kind === "currency" && (family === "currency-display" || random() < 0.35)) {
        intent.currencyDisplay = pick(["symbol", "code", "name"]);
        add(pick(["with currency", "using the currency", "display the currency as"]));
        add(intent.currencyDisplay);
      }

      gold = intent;
    }

    const earlyZone = gold.kind === "date-time" && gold.timeZone && random() < 0.5;

    if (earlyZone) {
      add(pick(["in timezone", "using timezone", "timezone"]));
      add(gold.timeZone!);
    }

    if (family === "locale-timezone" || random() < (gold.kind === "date-time" ? 0.85 : 0.55)) {
      gold.locale = pick(["en-US", "de-DE", "fr-FR", "en-GB", "ja-JP", "uk-UA"]);
      add(pick(["in locale", "using locale", "in", "locale"]));
      add(gold.locale);
    }

    if (gold.kind === "date-time" && gold.timeZone && !earlyZone) {
      add(pick(["and timezone", "in timezone", "using timezone", "timezone"]));
      add(gold.timeZone);
    }

    // The base construction owns the split, including subsequent hard negatives.
    const fingerprint = constructionFingerprint(text, roles);
    const bucket = parseInt(fingerprint.slice(0, 8), 16) % 10;

    if (!positive) {
      const mode = Math.floor(random() * 10);

      const names = [
        "question",
        "description",
        "uncertainty",
        "precision-overflow",
        "rounding",
        "ambiguous-currency",
        "kind-conflict",
        "grouping-conflict",
        "precision-conflict",
        "native-precision-conflict",
      ];

      if (mode === 0) {
        add("?", "UNSUPPORTED");
      } else if (mode === 1) {
        const body = text;
        const bodyRoles = roles;

        text = "";
        roles = [];

        add(
          pick([
            "the old settings",
            "the saved output",
            "the previous configuration",
            "the documentation",
          ]),
        );

        add(
          pick([
            "previously used",
            "formerly specified",
            "earlier described",
            "historically required",
          ]),
          "UNSUPPORTED",
        );

        text += " " + body;
        roles.push(...bodyRoles);
      } else if (mode === 2) {
        const body = text;
        const bodyRoles = roles;

        text = "";
        roles = [];
        add(pick(["maybe", "perhaps", "probably"]), "UNSUPPORTED");
        text += " " + body;
        roles.push(...bodyRoles);
      } else if (mode === 3) {
        add("with");
        add(String(21 + Math.floor(random() * 79)), "EXACT");
        add("decimals");
      } else if (mode === 4) {
        add(
          pick([
            "with scientific notation",
            "with significant digits",
            "using floor rounding",
            "using ceiling rounding",
            "using ceil rounding",
            "round toward positive infinity",
            "round toward negative infinity",
            "round ties to even",
            "round to the nearest increment",
            "use a custom pattern",
          ]),
          "UNSUPPORTED",
        );
      } else if (mode === 5) {
        add(
          pick(["use dollars", "use pounds", "with dollar signs", "with yen signs"]),
          "UNSUPPORTED",
        );
      } else if (mode === 6) {
        add("and currency");
        add("EUR");
        add("and");
        add("USD");
      } else if (mode === 7) {
        add("and grouping and no");
        add("grouping", "NOGROUP");
      } else if (mode === 9) {
        add("with");
        add("2", "EXACT");
        add("decimal places and");

        add(
          pick(["use native precision", "without decimal precision overrides", "default decimals"]),
          "DEFAULT_PRECISION",
        );
      } else {
        add("with");
        add("0", "EXACT");
        add("decimals and");
        add("1", "EXACT");
        add("decimals");
      }

      const prepared = prepare(text);
      let decoded: FormatIntent | null = null;

      try {
        decoded = resolveRoles(
          prepared,
          roles.map((role) => ({ role, score: 1 })),
        ).intent;
      } catch {}

      if (decoded !== null) throw Error(`negative supervision accepted: ${text}`);

      const key = text.trim().toLowerCase();
      const signature = JSON.stringify([roles, null]);

      if (excluded.has(hash(key))) continue;

      if (seen.has(key)) {
        if (seen.get(key) !== signature) throw Error(`conflicting duplicate supervision: ${text}`);
        continue;
      }

      seen.set(key, signature);

      yield {
        text,
        context: {},
        intent: null,
        roles,
        family: names[mode]!,
        fingerprint,
        split: bucket < 8 ? "train" : bucket === 8 ? "validation" : "heldout",
      };

      emitted++;
      continue;
    }

    const prepared = prepare(text);

    const decoded = resolveRoles(
      prepared,
      roles.map((role) => ({ role, score: 1 })),
    ).intent;

    if (!isDeepStrictEqual(decoded, gold)) {
      throw Error(
        `invalid supervision: ${text}: ${JSON.stringify(decoded)} vs ${JSON.stringify(gold)}`,
      );
    }

    const key = text.trim().toLowerCase();
    const signature = JSON.stringify([roles, gold]);

    if (excluded.has(hash(key))) continue;

    if (seen.has(key)) {
      if (seen.get(key) !== signature) throw Error(`conflicting duplicate supervision: ${text}`);
      continue;
    }

    seen.set(key, signature);

    yield {
      text,
      context: {},
      intent: gold,
      roles,
      family,
      fingerprint,
      split: bucket < 8 ? "train" : bucket === 8 ? "validation" : "heldout",
    };

    emitted++;
  }
}

export function generate(
  count = 8000,
  seed = SEED,
  excluded: ReadonlySet<string> = new Set(),
): Row[] {
  return [...sample(count, seed, excluded)];
}

export function writeCorpus(path: string, count = 8000): Row[] {
  const excluded = reservedHashes(
    resolve(import.meta.dir, "../../evaluation/reserved"),
    resolve(import.meta.dir, "../../evaluation/consumed"),
  );

  const rows: Row[] = [];

  function* records(): Generator<Row> {
    for (const row of sample(count, SEED, excluded)) {
      rows.push(row);
      yield row;
    }
  }

  const written = writeJsonl(path, records());

  writeFileSync(
    path + ".manifest.json",
    JSON.stringify(
      {
        rows: rows.length,
        sha256: written.sha256,
        generatorSha256: hash(readFileSync(import.meta.filename)),
        seed: SEED,
        trainingEligible: true,
        splitPolicy:
          "Literal substitutions and base/negative pairs share a construction fingerprint; " +
          "hash buckets 80/10/10.",
        excludedReservedTexts: excluded.size,
      },
      null,
      2,
    ) + "\n",
    { flag: "wx" },
  );

  return rows;
}
