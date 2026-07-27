export const CURRENCIES = ["EUR", "USD", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY"] as const;
export type Currency = (typeof CURRENCIES)[number];
export const STYLES = ["short", "medium", "long", "full"] as const;
export type Style = (typeof STYLES)[number];

export interface Precision {
  minimum?: number;
  maximum?: number;
}

export interface Context {
  locale?: string;
  timeZone?: string;
}

interface NumericOptions extends Context {
  precision?: Precision;
  grouping?: boolean;
  negative: "minus" | "parentheses";
}

export type NumericIntent = NumericOptions &
  (
    | { kind: "number" | "percent" }
    | {
        kind: "currency";
        currency: Currency;
        currencyDisplay: "symbol" | "code" | "name";
      }
  );

export interface DateTimeIntent extends Context {
  kind: "date-time";
  dateStyle?: Style;
  timeStyle?: Style;
  hourCycle?: "h12" | "h23";
}

export type FormatIntent = NumericIntent | DateTimeIntent;

export type FailureCode =
  | "invalid-input"
  | "invalid-intent"
  | "invalid-context"
  | "missing-context"
  | "conflict"
  | "unsupported"
  | "invalid-value";

export type Result<T> =
  ({ supported: true } & T) | { supported: false; code: FailureCode; reason: string };

export class InputError extends Error {
  constructor(
    readonly code: FailureCode,
    message: string,
  ) {
    super(message);
  }
}

export interface Part {
  type: string;
  value: string;
}

export type Formatter =
  | {
      kind: "number";
      locale: string;
      options: Readonly<Intl.NumberFormatOptions>;
      negative: NumericIntent["negative"];
      format(value: number): Result<{ text: string }>;
      formatToParts(value: number): Result<{ parts: readonly Part[] }>;
    }
  | {
      kind: "date-time";
      locale: string;
      options: Readonly<Intl.DateTimeFormatOptions>;
      format(value: number): Result<{ text: string }>;
      formatToParts(value: number): Result<{ parts: readonly Part[] }>;
    };
