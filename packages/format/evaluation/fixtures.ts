/** Fixed standard-based targets, without calling the package compiler. */
import type { Context, FormatIntent, Part } from "../src/types.js";

export type ProbeValue = number | "-0" | "NaN" | "Infinity" | "wrong-type";

export type Probe = {
  value: ProbeValue;
  target:
    { supported: true; text: string; parts?: readonly Part[] } | { supported: false; code: string };
};

export type CompileTarget =
  | { supported: false; code: string }
  | {
      supported: true;
      kind: "number" | "date-time";
      locale: string;
      options: Intl.NumberFormatOptions | Intl.DateTimeFormatOptions;
      negative?: "minus" | "parentheses";
      probes: readonly Probe[];
    };

export interface Witness {
  text: string;
  family: string;
  intent: FormatIntent;
  compileOptions: Context;
  compileTarget: CompileTarget;
}

const currency: FormatIntent = {
  kind: "currency",
  currency: "EUR",
  currencyDisplay: "symbol",
  precision: { minimum: 2, maximum: 2 },
  negative: "parentheses",
};

const currencyOptions: Intl.NumberFormatOptions = {
  style: "currency",
  notation: "standard",
  numberingSystem: "latn",
  signDisplay: "auto",
  currency: "EUR",
  currencyDisplay: "symbol",
  currencySign: "standard",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
};

const date: FormatIntent = {
  kind: "date-time",
  dateStyle: "short",
  timeStyle: "short",
  hourCycle: "h23",
};

const dateOptions: Intl.DateTimeFormatOptions = {
  calendar: "gregory",
  numberingSystem: "latn",
  timeZone: "America/New_York",
  dateStyle: "short",
  timeStyle: "short",
  hourCycle: "h23",
};

export const WITNESSES: readonly Witness[] = [
  {
    text:
      "please show euros with exactly two decimal places " +
      "and use parentheses for negative values",
    family: "currency",
    intent: currency,
    compileOptions: { locale: "en-US" },
    compileTarget: {
      supported: true,
      kind: "number",
      locale: "en-US",
      negative: "parentheses",
      options: currencyOptions,
      probes: [
        { value: -1234.5, target: { supported: true, text: "(€1,234.50)" } },
        { value: "-0", target: { supported: true, text: "(€0.00)" } },
        { value: "NaN", target: { supported: false, code: "invalid-value" } },
      ],
    },
  },
  {
    text:
      "please display euro amounts with two decimals " +
      "and parentheses for negatives in locale de-DE",
    family: "grouping-negative",
    intent: { ...currency, locale: "de-DE" },
    compileOptions: {},
    compileTarget: {
      supported: true,
      kind: "number",
      locale: "de-DE",
      negative: "parentheses",
      options: currencyOptions,
      probes: [{ value: -1234.5, target: { supported: true, text: "(1.234,50\u00a0€)" } }],
    },
  },
  {
    text: "please show percent output using one decimal place and no grouping",
    family: "percent",
    intent: {
      kind: "percent",
      negative: "minus",
      precision: { minimum: 1, maximum: 1 },
      grouping: false,
    },
    compileOptions: { locale: "en-US" },
    compileTarget: {
      supported: true,
      kind: "number",
      locale: "en-US",
      negative: "minus",
      options: {
        style: "percent",
        notation: "standard",
        numberingSystem: "latn",
        signDisplay: "auto",
        minimumFractionDigits: 1,
        maximumFractionDigits: 1,
        useGrouping: false,
      },
      probes: [
        { value: 0.125, target: { supported: true, text: "12.5%" } },
        { value: "wrong-type", target: { supported: false, code: "invalid-value" } },
      ],
    },
  },
  {
    text:
      "please format Japanese yen and show the currency code " +
      "without decimal precision overrides",
    family: "currency-display",
    intent: {
      kind: "currency",
      currency: "JPY",
      currencyDisplay: "code",
      negative: "minus",
    },
    compileOptions: { locale: "en-US" },
    compileTarget: {
      supported: true,
      kind: "number",
      locale: "en-US",
      negative: "minus",
      options: {
        style: "currency",
        notation: "standard",
        numberingSystem: "latn",
        signDisplay: "auto",
        currency: "JPY",
        currencyDisplay: "code",
        currencySign: "standard",
      },
      probes: [{ value: 1234.5, target: { supported: true, text: "JPY\u00a01,235" } }],
    },
  },
  {
    text: "please display short date and short time with a 24-hour clock",
    family: "date-time",
    intent: date,
    compileOptions: { locale: "en-US", timeZone: "America/New_York" },
    compileTarget: {
      supported: true,
      kind: "date-time",
      locale: "en-US",
      options: dateOptions,
      probes: [
        { value: 1710052200000, target: { supported: true, text: "3/10/24, 01:30" } },
        { value: 1710055800000, target: { supported: true, text: "3/10/24, 03:30" } },
        { value: 1730611800000, target: { supported: true, text: "11/3/24, 01:30" } },
        { value: 1730615400000, target: { supported: true, text: "11/3/24, 01:30" } },
        { value: 0.5, target: { supported: false, code: "invalid-value" } },
      ],
    },
  },
  {
    text: "please show a short date in timezone America/New_York",
    family: "date",
    intent: { kind: "date-time", dateStyle: "short", timeZone: "America/New_York" },
    compileOptions: { locale: "en-US" },
    compileTarget: {
      supported: true,
      kind: "date-time",
      locale: "en-US",
      options: {
        calendar: "gregory",
        numberingSystem: "latn",
        timeZone: "America/New_York",
        dateStyle: "short",
      },
      probes: [{ value: 1709253000000, target: { supported: true, text: "2/29/24" } }],
    },
  },
  {
    text: "please show number output using two decimal places",
    family: "number",
    intent: {
      kind: "number",
      negative: "minus",
      precision: { minimum: 2, maximum: 2 },
    },
    compileOptions: {},
    compileTarget: { supported: false, code: "missing-context" },
  },
  {
    text: "please display a short time in locale en-US",
    family: "time",
    intent: { kind: "date-time", timeStyle: "short", locale: "en-US" },
    compileOptions: {},
    compileTarget: { supported: false, code: "missing-context" },
  },
  {
    text: "please display EUR with two decimals in locale en-US",
    family: "locale-timezone",
    intent: {
      kind: "currency",
      currency: "EUR",
      currencyDisplay: "symbol",
      negative: "minus",
      precision: { minimum: 2, maximum: 2 },
      locale: "en-US",
    },
    compileOptions: { locale: "de-DE" },
    compileTarget: { supported: false, code: "conflict" },
  },
];
