import { expect, test } from "bun:test";

import type { Role } from "../src/labels.js";
import { prepare } from "../src/prepare.js";
import { resolveRoles } from "../src/resolve.js";

function decode(text: string, roles: readonly Role[]) {
  return resolveRoles(
    prepare(text),
    roles.map((role) => ({ role, score: 1 })),
  ).intent;
}

test(
  "currency, exact digits and parentheses reference " +
    "preserves literals without caller defaults",
  () => {
    const text = "euros, two decimals, negatives in parentheses";

    const roles: Role[] = [
      "CURRENCY",
      "GLUE",
      "EXACT",
      "PRECISION",
      "GLUE",
      "GLUE",
      "GLUE",
      "PARENS",
    ];

    expect(decode(text, roles)).toEqual({
      kind: "currency",
      currency: "EUR",
      currencyDisplay: "symbol",
      negative: "parentheses",
      precision: { minimum: 2, maximum: 2 },
    });

    expect(() =>
      decode(
        text,
        roles.map((r) => (r === "EXACT" ? "O" : r)),
      ),
    ).toThrow("numeric formatting fact was dropped");

    expect(() =>
      decode(
        text,
        roles.map((r) => (r === "PARENS" ? "O" : r)),
      ),
    ).toThrow("formatting option fact was dropped");

    expect(
      decode("US dollars with at least 2 decimal places and at most 4 decimal places", [
        "CURRENCY",
        "GLUE",
        "GLUE",
        "GLUE",
        "MIN",
        "PRECISION",
        "PRECISION",
        "GLUE",
        "GLUE",
        "GLUE",
        "MAX",
        "PRECISION",
        "PRECISION",
      ]),
    ).toEqual({
      kind: "currency",
      currency: "USD",
      currencyDisplay: "symbol",
      negative: "minus",
      precision: { minimum: 2, maximum: 4 },
    });

    expect(decode("number with 2.0 decimals", ["NUMBER", "GLUE", "EXACT", "PRECISION"])).toEqual({
      kind: "number",
      negative: "minus",
      precision: { minimum: 2, maximum: 2 },
    });

    for (const count of ["2.5", "negative two", "-2"]) {
      expect(() =>
        decode(`number with ${count} decimals`, ["NUMBER", "GLUE", "EXACT", "PRECISION"]),
      ).toThrow("whole digit count");
    }

    expect(
      decode("number with positive two decimals", ["NUMBER", "GLUE", "EXACT", "PRECISION"]),
    ).toEqual({
      kind: "number",
      negative: "minus",
      precision: { minimum: 2, maximum: 2 },
    });
  },
);

test(
  "date and time styles, hour cycle, " +
    "locale and zone remain separate facts and conflicts refuse",
  () => {
    expect(
      decode("short date and long time with 24-hour clock in en-US timezone America/New_York", [
        "DSTYLE",
        "DATE",
        "GLUE",
        "TSTYLE",
        "TIME",
        "GLUE",
        "HOUR",
        "GLUE",
        "GLUE",
        "LOCALE",
        "GLUE",
        "ZONE",
      ]),
    ).toEqual({
      kind: "date-time",
      dateStyle: "short",
      timeStyle: "long",
      hourCycle: "h23",
      locale: "en-US",
      timeZone: "America/New_York",
    });

    expect(decode("full date and time", ["STYLE", "DATE", "GLUE", "TIME"])).toEqual({
      kind: "date-time",
      dateStyle: "full",
      timeStyle: "full",
    });

    expect(
      decode("time in locale en timezone UTC", ["TIME", "GLUE", "GLUE", "LOCALE", "GLUE", "ZONE"]),
    ).toEqual({ kind: "date-time", timeStyle: "medium", locale: "en", timeZone: "UTC" });

    expect(() =>
      decode("number with grouping and no grouping", [
        "NUMBER",
        "GLUE",
        "GROUP",
        "GLUE",
        "GLUE",
        "NOGROUP",
      ]),
    ).toThrow("conflicting grouping");

    expect(() =>
      decode("number with 2 decimals and 3 decimals", [
        "NUMBER",
        "GLUE",
        "EXACT",
        "PRECISION",
        "GLUE",
        "EXACT",
        "PRECISION",
      ]),
    ).toThrow("conflicting exact decimal places");

    expect(() => decode("date with 24-hour clock", ["DATE", "GLUE", "HOUR", "GLUE"])).toThrow(
      "unrequested date/time part",
    );

    expect(() =>
      decode("number and percent currency EUR", ["NUMBER", "GLUE", "PERCENT", "GLUE", "CURRENCY"]),
    ).toThrow("currency and percentage");

    expect(() =>
      decode("EUR as code in en-US", ["CURRENCY", "GLUE", "DISPLAY", "GLUE", "O"]),
    ).toThrow("locale fact was dropped");

    expect(() =>
      decode("number with scientific notation", ["NUMBER", "GLUE", "UNSUPPORTED", "GLUE"]),
    ).toThrow("unsupported");
  },
);

test("explicit native precision preserves defaults and conflicts with stated counts", () => {
  const expected = {
    kind: "currency",
    currency: "JPY",
    currencyDisplay: "symbol",
    negative: "minus",
  } as const;

  expect(
    decode("Japanese yen using native precision", [
      "CURRENCY",
      "GLUE",
      "DEFAULT_PRECISION",
      "GLUE",
    ]),
  ).toEqual(expected);

  expect(
    decode("Japanese yen without decimal precision overrides", [
      "CURRENCY",
      "GLUE",
      "PRECISION",
      "GLUE",
      "DEFAULT_PRECISION",
    ]),
  ).toEqual(expected);

  expect(() =>
    decode("EUR with 2 decimals and native precision", [
      "CURRENCY",
      "GLUE",
      "EXACT",
      "PRECISION",
      "GLUE",
      "DEFAULT_PRECISION",
      "GLUE",
    ]),
  ).toThrow("native precision conflicts");

  expect(
    decode("percent with one decimal place", [
      "PERCENT",
      "GLUE",
      "EXACT",
      "PRECISION",
      "PRECISION",
    ]),
  ).toEqual({
    kind: "percent",
    negative: "minus",
    precision: { minimum: 1, maximum: 1 },
  });
});

test("locale and timezone noun bindings cannot create an unrelated currency or time part", () => {
  expect(
    decode("short date in time zone America/New_York", ["DSTYLE", "DATE", "GLUE", "GLUE", "ZONE"]),
  ).toEqual({ kind: "date-time", dateStyle: "short", timeZone: "America/New_York" });

  expect(() => decode("number in locale CAD", ["NUMBER", "GLUE", "GLUE", "LOCALE"])).toThrow(
    "locale is not supported",
  );

  expect(() => decode("date in timezone en-US", ["DATE", "GLUE", "GLUE", "ZONE"])).toThrow(
    "timezone is not supported",
  );
});
