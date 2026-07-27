/** Evaluation-only authoring. No parser, compiler or training renderer imports. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";

import type { Context, DateTimeIntent, FormatIntent, NumericIntent, Part } from "../src/types.js";
import type { Row } from "../training/src/corpus.js";
import { WITNESSES } from "./fixtures.js";
import type { CompileTarget, ProbeValue } from "./fixtures.js";

export interface Acceptance extends Row {
  critical: boolean;
  compileOptions: Context;
  compileTarget: CompileTarget | null;
}

const hash = (value: string | Uint8Array) => createHash("sha256").update(value).digest("hex");

const families = [
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
];

const currencies = ["EUR", "USD", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY"] as const;

const currencyNames = {
  EUR: "euros",
  USD: "US dollars",
  GBP: "British pounds",
  JPY: "Japanese yen",
  CAD: "Canadian dollars",
  AUD: "Australian dollars",
  CHF: "Swiss francs",
  CNY: "Chinese yuan",
};

const styles = ["short", "medium", "long", "full"] as const;
const locales = ["en-US", "de-DE", "fr-FR", "en-GB", "ja-JP", "uk-UA"];
const zones = ["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"];

const small = [
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

/** Native Intl is the independent standard implementation; parentheses are checked separately. */
function target(intent: FormatIntent, context: Context): CompileTarget {
  const locale = Intl.getCanonicalLocales(intent.locale ?? context.locale!)[0]!;

  if (intent.kind === "date-time") {
    const zone = intent.timeZone ?? context.timeZone!;

    const options: Intl.DateTimeFormatOptions = {
      calendar: "gregory",
      numberingSystem: "latn",
      timeZone: zone,
    };

    if (intent.dateStyle) options.dateStyle = intent.dateStyle;
    if (intent.timeStyle) options.timeStyle = intent.timeStyle;
    if (intent.hourCycle) options.hourCycle = intent.hourCycle;
    const fmt = new Intl.DateTimeFormat(locale, options);

    return {
      supported: true,
      kind: "date-time",
      locale,
      options,
      probes: [1710055800000, 1709253000000].map((value) => ({
        value,
        target: {
          supported: true,
          text: fmt.format(value),
          parts: fmt.formatToParts(value),
        },
      })),
    };
  }

  const options: Intl.NumberFormatOptions = {
    style: "decimal",
    notation: "standard",
    numberingSystem: "latn",
    signDisplay: "auto",
  };

  if (intent.kind === "percent") options.style = "percent";

  if (intent.kind === "currency") {
    options.style = "currency";
    options.currency = intent.currency;
    options.currencyDisplay = intent.currencyDisplay;
    options.currencySign = "standard";
  }

  if (intent.precision?.minimum !== undefined) {
    options.minimumFractionDigits = intent.precision.minimum;
  }

  if (intent.precision?.maximum !== undefined) {
    options.maximumFractionDigits = intent.precision.maximum;
  }

  if (intent.grouping !== undefined) options.useGrouping = intent.grouping;
  const fmt = new Intl.NumberFormat(locale, options);

  const values: ProbeValue[] =
    intent.kind === "percent" ? [0.125, -0.25, "-0"] : [1234.567, -9876.125, "-0"];

  const probes = values.map((value) => {
    const n = value === "-0" ? -0 : (value as number);
    let parts: Part[] = fmt.formatToParts(n);

    if (intent.negative === "parentheses" && (n < 0 || Object.is(n, -0))) {
      parts = [
        { type: "literal", value: "(" },
        ...fmt.formatToParts(-n),
        { type: "literal", value: ")" },
      ];
    }

    return {
      value,
      target: {
        supported: true as const,
        text: parts.map((part) => part.value).join(""),
        parts,
      },
    };
  });

  return {
    supported: true,
    kind: "number",
    locale,
    negative: intent.negative,
    options,
    probes,
  };
}

export function author(revision = 1): Acceptance[] {
  if (!Number.isSafeInteger(revision) || revision < 1 || revision > 10) {
    throw Error("unsupported acceptance revision");
  }

  const rows: Acceptance[] = [];

  function add(
    text: string,
    intent: FormatIntent | null,
    family: string,
    critical = false,
    compileOptions: Context = {
      ...(intent?.locale ? {} : { locale: "en-US" }),
      ...(intent?.kind === "date-time" && !intent.timeZone ? { timeZone: "UTC" } : {}),
    },
    compileTarget: CompileTarget | null = intent ? target(intent, compileOptions) : null,
  ) {
    rows.push({
      text,
      context: {},
      intent,
      roles: [],
      family,
      split: "heldout",
      fingerprint: hash(text),
      trainingEligible: false,
      critical,
      compileOptions,
      compileTarget,
    });
  }

  const prefixes =
    revision >= 4
      ? [
          "please set the output format to",
          "I need the formatted value to use",
          "for the printed invoice, render",
          "please show the presentation using",
          "configure the displayed output as",
        ]
      : revision === 3
        ? [
            "please configure the displayed output to show",
            "I want the output presentation to use",
            "for the report output, display",
            "please set the rendered value to",
            "please arrange the output using",
          ]
        : revision === 1
          ? [
              "please make the output use",
              "I need the displayed result to use",
              "for the exported summary, show",
              "set the displayed value to",
              "please present the result as",
            ]
          : [
              "please arrange the displayed result as",
              "I would like the rendered output to use",
              "for the downloaded summary, display",
              "set the output presentation to",
              "please configure the result as",
            ];

  for (const family of families) {
    for (let i = 0; i < 75; i++) {
      const n = (i + 3 * revision) % 21;
      const min = (i + revision) % 11;
      const max = min + ((i + 3) % 11);
      const currency = currencies[(i + revision) % currencies.length]!;
      const locale = locales[(Math.floor(i / 4) + revision) % locales.length]!;
      const zone = zones[(Math.floor(i / 24) + revision) % zones.length]!;
      const s1 = styles[(i + revision) % 4]!;
      const s2 = styles[(Math.floor(i / 4) + revision) % 4]!;

      const prefix = prefixes[Math.floor(i / 15)]!;
      const count = i % 2 ? String(n) : small[n]!;
      let intent: FormatIntent;
      let body: string;

      if (family === "date" || family === "time" || family === "date-time") {
        const d: DateTimeIntent = { kind: "date-time", locale, timeZone: zone };

        if (family !== "time") d.dateStyle = s1;

        if (family !== "date") {
          d.timeStyle = family === "time" ? s1 : s2;
          d.hourCycle = i % 2 ? "h12" : "h23";
        }

        body =
          family === "date"
            ? `${s1} date`
            : family === "time"
              ? `${s1} time`
              : `${s1} date and ${s2} time`;

        if (family !== "date") body += ` using a ${i % 2 ? "12-hour" : "24-hour"} clock`;
        body += ` in locale ${locale} and timezone ${zone}`;
        intent = d;
      } else if (family === "locale-timezone") {
        if (i % 2) {
          intent = {
            kind: "date-time",
            dateStyle: s1,
            timeStyle: s1,
            hourCycle: "h23",
            locale,
            timeZone: zone,
          };

          const clock = `${s1} date and time with a 24-hour clock`;

          body = `${clock} in locale ${locale} and timezone ${zone}`;
        } else {
          intent = {
            kind: "currency",
            currency,
            currencyDisplay: "symbol",
            negative: "minus",
            precision: { minimum: n, maximum: n },
            locale,
          };

          body = `currency ${currency} using ${count} decimal places and locale ${locale}`;
        }
      } else {
        const kind =
          family === "currency" || family === "currency-display"
            ? "currency"
            : family === "percent"
              ? "percent"
              : "number";

        const numeric: NumericIntent =
          kind === "currency"
            ? {
                kind,
                currency,
                currencyDisplay:
                  family === "currency-display"
                    ? (["symbol", "code", "name"] as const)[i % 3]!
                    : "symbol",
                negative: "minus",
              }
            : { kind, negative: "minus" };

        body = kind === "currency" ? currencyNames[currency] : kind;

        if (family === "precision-range") {
          numeric.precision = { minimum: min, maximum: max };
          body += ` with a minimum of ${small[min]} decimals and a maximum of ${max} decimals`;
        } else {
          numeric.precision = { minimum: n, maximum: n };
          body += ` with ${count} decimal places`;
        }

        if (family === "grouping-negative") {
          numeric.grouping = i % 2 === 0;
          numeric.negative = i % 3 ? "parentheses" : "minus";
          body += numeric.grouping ? " and grouping separators" : " and no grouping separators";

          body +=
            numeric.negative === "parentheses"
              ? " and parentheses for negative values"
              : " and minus signs for negative values";
        }

        if (numeric.kind === "currency" && family === "currency-display") {
          body += ` with currency ${numeric.currencyDisplay} display`;
        }

        numeric.locale = locale;
        body += ` in locale ${locale}`;
        intent = numeric;
      }

      add(`${prefix} ${body}`, intent, family);
    }
  }

  // Every negative retains a supported instruction; the added condition makes it invalid.
  const negatives = [
    "question",
    "description",
    "uncertainty",
    "ambiguous-currency",
    "precision-conflict",
    "fractional-digits",
    "rounding",
    "grouping-conflict",
    "context-conflict",
  ];

  for (const family of negatives) {
    for (let i = 0; i < 70; i++) {
      const n = i % 21;
      const currency = currencies[(i + revision) % 8]!;
      const prefix = prefixes[Math.floor(i / 14)]!;
      const locale = locales[(Math.floor(i / 5) + revision) % 6]!;

      const currencyName = currencyNames[currency];
      let text = `${prefix} ${currencyName} with ${n} decimal places in locale ${locale}`;

      if (family === "question") {
        text += "?";
      } else if (family === "description") {
        text = `the saved display settings previously specified to ${text}`;
      } else if (family === "uncertainty") {
        text = `perhaps ${text}`;
      } else if (family === "ambiguous-currency") {
        text += " and also use dollars";
      } else if (family === "precision-conflict") {
        text += ` and ${n + 1} decimal places`;
      } else if (family === "fractional-digits") {
        text += " and 1.25 decimal places";
      } else if (family === "rounding") {
        text += i % 2 ? " using ceiling rounding" : " using three significant digits";
      } else if (family === "grouping-conflict") {
        text += " with grouping and without grouping";
      } else {
        text += ` and locale ${locale === "en-US" ? "de-DE" : "en-US"}`;
      }

      add(text, null, family);
    }
  }

  const witnessPrefix =
    revision >= 4
      ? "for the invoice output, "
      : revision === 1
        ? ""
        : revision === 2
          ? "for the reviewed output, "
          : "for the requested output, ";

  for (const witness of WITNESSES) {
    add(
      witnessPrefix + witness.text,
      witness.intent,
      witness.family,
      true,
      witness.compileOptions,
      witness.compileTarget,
    );
  }

  const positiveControls: [string, FormatIntent, string][] = [
    [
      "please show number values with positive two decimal places",
      { kind: "number", negative: "minus", precision: { minimum: 2, maximum: 2 } },
      "number",
    ],
    [
      "please show number values with 2.0 decimals",
      { kind: "number", negative: "minus", precision: { minimum: 2, maximum: 2 } },
      "number",
    ],
    [
      "please display short time in locale en-us and timezone utc using a 24-hour clock",
      {
        kind: "date-time",
        timeStyle: "short",
        hourCycle: "h23",
        locale: "en-US",
        timeZone: "UTC",
      },
      "locale-timezone",
    ],
    [
      'please display currency "EUR" with two decimal places',
      {
        kind: "currency",
        currency: "EUR",
        currencyDisplay: "symbol",
        negative: "minus",
        precision: { minimum: 2, maximum: 2 },
      },
      "currency",
    ],
    [
      "please show a number with at least five decimal places",
      { kind: "number", negative: "minus", precision: { minimum: 5 } },
      "precision-range",
    ],
    [
      "please show a number with at most one decimal place",
      { kind: "number", negative: "minus", precision: { maximum: 1 } },
      "precision-range",
    ],
  ];

  for (const [text, intent, family] of positiveControls) {
    add(witnessPrefix + text, intent, family, true);
  }

  for (const [text, family] of [
    ["please show numbers with negative two decimal places", "signed-digits"],
    ["please format numbers with minus 2 decimal places", "signed-digits"],
    ["please show numbers with twenty-one decimal places", "precision-overflow"],
    ["please use EUR with at least four decimals and at most two decimals", "precision-conflict"],
    ["please display short date with a 24-hour clock", "date-hour-conflict"],
    ["please show percentage output with currency EUR", "kind-conflict"],
    ["please format numbers in timezone America/New_York", "timezone-kind-conflict"],
    ["please show numbers in locale en-US-u-nu-arab", "unsupported-locale-extension"],
    ["please format ZZZ with two decimal places", "unsupported-currency"],
    ["please format numbers with two significant digits", "rounding"],
  ]) {
    add(witnessPrefix + text!, null, family!, true);
  }

  if (revision >= 2) {
    const prefix =
      revision >= 4
        ? "for the invoice value, "
        : revision === 2
          ? "for the reviewed value, "
          : "for the requested value, ";

    add(
      prefix + "please use Japanese yen with native precision and currency code display",
      { kind: "currency", currency: "JPY", currencyDisplay: "code", negative: "minus" },
      "currency-display",
      true,
    );

    add(
      prefix + "please show number output with default decimal precision",
      { kind: "number", negative: "minus" },
      "number",
      true,
    );

    add(
      prefix + "please use EUR with three decimals and native precision",
      null,
      "native-precision-conflict",
      true,
    );

    add(
      prefix + "please show percent using two decimals without decimal precision overrides",
      null,
      "native-precision-conflict",
      true,
    );
  }

  return rows;
}

if (import.meta.main) {
  const [scopePath, out, revision] = process.argv.slice(2);

  if (!scopePath || !out) {
    throw Error("usage: author <scope-freeze.json> <new-reserved.jsonl> [revision]");
  }

  const scopeBytes = readFileSync(scopePath);
  const scope = JSON.parse(scopeBytes.toString());

  if (JSON.stringify(scope.families) !== JSON.stringify(families)) {
    throw Error("scope families disagree");
  }

  const rows = author(Number(revision ?? 1));
  const body = rows.map((row) => JSON.stringify(row)).join("\n") + "\n";

  const keys = new Set(rows.map((row) => row.text.trim().toLowerCase()));

  if (keys.size !== rows.length) throw Error("reserved wording must be unique before writing");
  writeFileSync(out, body, { flag: "wx" });

  writeFileSync(
    out + ".meta.json",
    JSON.stringify(
      {
        sha256: hash(body),
        authorSha256: hash(readFileSync(import.meta.filename)),
        candidateSha256: scope.artifactFileSha256,
        scopeFreezeSha256: hash(scopeBytes),
        createdAt: new Date().toISOString(),
        trainingEligible: false,
        rows: rows.length,
        authorship:
          "AI-authored by the same implementation assistant through an evaluation-only writer",
        independenceLimits:
          "No production parser/compiler or training renderer imports. " +
          "Native Intl targets and fixed standard-based witnesses. " +
          "Bounded constructions and numeric variation are not independent human evidence " +
          "or population accuracy.",
        localeData: { ...process.versions },
      },
      null,
      2,
    ) + "\n",
    { flag: "wx" },
  );

  console.log({ rows: rows.length, sha256: hash(body) });
}
