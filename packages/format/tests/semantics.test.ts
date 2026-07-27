import { expect, test } from "bun:test";

import { compile, explain } from "../src/index.js";
import type { Context, FormatIntent, Formatter } from "../src/index.js";

function formatter(intent: FormatIntent, context: Context): Formatter {
  const result = compile(intent, context);

  if (!result.supported) throw Error(result.reason);

  return result.formatter;
}

function text(formatter: Formatter, value: number): string {
  const result = formatter.format(value);

  if (!result.supported) throw Error(result.reason);

  return result.text;
}

test("EUR reference preserves precision, locale placement, parentheses and negative zero", () => {
  const intent: FormatIntent = {
    kind: "currency",
    currency: "EUR",
    currencyDisplay: "symbol",
    precision: { minimum: 2, maximum: 2 },
    negative: "parentheses",
  };

  const us = formatter(intent, { locale: "en-US" });

  expect(text(us, -1234.5)).toBe("(€1,234.50)");
  expect(text(us, 1234.5)).toBe("€1,234.50");
  expect(text(us, -0)).toBe("(€0.00)");
  expect(text(us, 0)).toBe("€0.00");
  expect(text(formatter(intent, { locale: "de-DE" }), -1234.5)).toBe("(1.234,50\u00a0€)");

  expect(us.formatToParts(-1234.5)).toEqual({
    supported: true,
    parts: [
      { type: "literal", value: "(" },
      { type: "currency", value: "€" },
      { type: "integer", value: "1" },
      { type: "group", value: "," },
      { type: "integer", value: "234" },
      { type: "decimal", value: "." },
      { type: "fraction", value: "50" },
      { type: "literal", value: ")" },
    ],
  });

  expect(us.options).toMatchObject({
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
    currencySign: "standard",
  });

  expect(explain(intent)).toContain("including negative zero");
});

test(
  "native precision defaults, percent fractions, " +
    "grouping and explicit currency displays are distinct",
  () => {
    expect(
      text(
        formatter(
          {
            kind: "currency",
            currency: "JPY",
            currencyDisplay: "code",
            negative: "minus",
          },
          { locale: "en-US" },
        ),
        1234.5,
      ),
    ).toBe("JPY\u00a01,235");

    expect(
      text(
        formatter(
          {
            kind: "percent",
            precision: { minimum: 1, maximum: 1 },
            grouping: false,
            negative: "minus",
          },
          { locale: "en-US" },
        ),
        0.125,
      ),
    ).toBe("12.5%");

    const number = formatter(
      { kind: "number", precision: { maximum: 2 }, grouping: false, negative: "minus" },
      { locale: "en-US" },
    );

    expect(text(number, 1234.567)).toBe("1234.57");
    expect(text(number, -0)).toBe("-0");

    const minimum = formatter(
      { kind: "number", precision: { minimum: 5 }, negative: "minus" },
      { locale: "en-US" },
    );

    expect(text(minimum, 1.25)).toBe("1.25000");
    expect(number.format(NaN)).toMatchObject({ supported: false, code: "invalid-value" });

    expect(number.format(Infinity)).toMatchObject({
      supported: false,
      code: "invalid-value",
    });

    expect(number.format("2" as unknown as number)).toMatchObject({
      supported: false,
      code: "invalid-value",
    });

    expect(Reflect.set(number.options, "minimumFractionDigits", 20)).toBe(false);
  },
);

test(
  "date/time requires explicit timezone and exposes " +
    "Gregorian date and hour-cycle behavior across DST",
  () => {
    const intent: FormatIntent = {
      kind: "date-time",
      dateStyle: "short",
      timeStyle: "short",
      hourCycle: "h23",
    };

    const utc = formatter(intent, { locale: "en-US", timeZone: "UTC" });
    const ny = formatter(intent, { locale: "en-US", timeZone: "America/New_York" });

    expect(text(utc, Date.parse("2024-03-10T07:30:00Z"))).toBe("3/10/24, 07:30");
    expect(text(ny, Date.parse("2024-03-10T06:30:00Z"))).toBe("3/10/24, 01:30");
    expect(text(ny, Date.parse("2024-03-10T07:30:00Z"))).toBe("3/10/24, 03:30");

    expect(text(ny, Date.parse("2024-11-03T05:30:00Z"))).toBe(
      text(ny, Date.parse("2024-11-03T06:30:00Z")),
    );

    const date = formatter(
      { kind: "date-time", dateStyle: "short" },
      { locale: "en-US", timeZone: "America/New_York" },
    );

    expect(text(date, Date.parse("2024-03-01T00:30:00Z"))).toBe("2/29/24");

    expect(ny.options).toMatchObject({
      calendar: "gregory",
      numberingSystem: "latn",
      hourCycle: "h23",
    });

    expect(ny.format(0.5)).toMatchObject({ supported: false, code: "invalid-value" });

    expect(ny.format(8640000000000001)).toMatchObject({
      supported: false,
      code: "invalid-value",
    });
  },
);

test(
  "closed edited intents and context refuse conflicts, " +
    "unsupported precision and silent defaults",
  () => {
    const numeric: FormatIntent = { kind: "number", negative: "minus" };

    expect(compile(numeric)).toMatchObject({ supported: false, code: "missing-context" });

    expect(compile({ kind: "date-time", dateStyle: "short" }, { locale: "en-US" })).toMatchObject({
      supported: false,
      code: "missing-context",
    });

    expect(compile({ ...numeric, locale: "en-US" }, { locale: "de-DE" })).toMatchObject({
      supported: false,
      code: "conflict",
    });

    expect(compile({ ...numeric, locale: "en-us" }, { locale: "en-US" }).supported).toBe(true);

    expect(compile(numeric, { locale: "zz-ZZ" })).toMatchObject({
      supported: false,
      code: "unsupported",
    });

    expect(compile(numeric, { locale: "en-US-u-nu-arab" })).toMatchObject({
      supported: false,
      code: "unsupported",
    });

    expect(
      compile({ ...numeric, precision: { minimum: 3, maximum: 2 } }, { locale: "en-US" }),
    ).toMatchObject({ supported: false, code: "conflict" });

    expect(compile({ ...numeric, precision: { maximum: 21 } }, { locale: "en-US" })).toMatchObject({
      supported: false,
      code: "invalid-intent",
    });

    expect(compile({ ...numeric, precision: {} }, { locale: "en-US" })).toMatchObject({
      supported: false,
      code: "invalid-intent",
    });

    expect(
      compile({ ...numeric, grouping: "false" } as unknown as FormatIntent, {
        locale: "en-US",
      }),
    ).toMatchObject({ supported: false, code: "invalid-intent" });

    expect(
      compile({ ...numeric, roundingMode: "ceil" } as unknown as FormatIntent, {
        locale: "en-US",
      }),
    ).toMatchObject({ supported: false, code: "invalid-intent" });

    expect(
      compile(
        { kind: "date-time", dateStyle: "short", hourCycle: "h23" },
        { locale: "en-US", timeZone: "UTC" },
      ),
    ).toMatchObject({ supported: false, code: "invalid-intent" });

    expect(compile(numeric, { locale: "en-US", timeZone: "UTC" })).toMatchObject({
      supported: false,
      code: "unsupported",
    });

    expect(compile(numeric, Object.create({ locale: "en-US" }))).toMatchObject({
      supported: false,
      code: "invalid-context",
    });

    expect(
      compile({ kind: "date-time", timeStyle: "short" }, { locale: "en-US", timeZone: "invalid" }),
    ).toMatchObject({ supported: false, code: "invalid-context" });
  },
);
