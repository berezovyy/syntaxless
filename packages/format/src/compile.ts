import { InputError } from "./types.js";
import type { Context, FormatIntent, Formatter, Part, Result } from "./types.js";
import { locale, timeZone, validate, validateContext } from "./validate.js";

function failure(error: unknown): {
  supported: false;
  code: InputError["code"];
  reason: string;
} {
  return {
    supported: false,
    code: error instanceof InputError ? error.code : "unsupported",
    reason: error instanceof Error ? error.message : "formatting failed",
  };
}

function contextValue(name: keyof Context, intent: FormatIntent, context: Context): string {
  const normalize = name === "locale" ? locale : timeZone;
  const specified = intent[name] === undefined ? undefined : normalize(intent[name]);
  const supplied = context[name] === undefined ? undefined : normalize(context[name]);

  if (specified !== undefined && supplied !== undefined && specified !== supplied) {
    throw new InputError("conflict", `instruction and caller ${name} disagree`);
  }

  const value = specified ?? supplied;

  if (value === undefined) throw new InputError("missing-context", `provide an explicit ${name}`);

  return value;
}

export function compile(
  intent: FormatIntent,
  context: Context = {},
): Result<{ formatter: Formatter }> {
  try {
    validate(intent);
    validateContext(context);
    const resolvedLocale = contextValue("locale", intent, context);
    let formatter: Intl.NumberFormat | Intl.DateTimeFormat;

    let compiled:
      | Omit<Extract<Formatter, { kind: "number" }>, "format" | "formatToParts">
      | Omit<Extract<Formatter, { kind: "date-time" }>, "format" | "formatToParts">;

    const isDateTime = intent.kind === "date-time";
    let transform: (value: number) => number = (value) => value;
    let parentheses = false;

    if (intent.kind === "date-time") {
      const options: Intl.DateTimeFormatOptions = {
        calendar: "gregory",
        numberingSystem: "latn",
        timeZone: contextValue("timeZone", intent, context),
        ...(intent.dateStyle ? { dateStyle: intent.dateStyle } : {}),
        ...(intent.timeStyle ? { timeStyle: intent.timeStyle } : {}),
        ...(intent.hourCycle ? { hourCycle: intent.hourCycle } : {}),
      };

      formatter = new Intl.DateTimeFormat(resolvedLocale, options);

      compiled = {
        kind: "date-time",
        locale: resolvedLocale,
        options: Object.freeze(options),
      };
    } else {
      if (context.timeZone !== undefined) {
        throw new InputError("unsupported", "timezone is meaningful only for date/time formatting");
      }

      const options: Intl.NumberFormatOptions = {
        style: intent.kind === "number" ? "decimal" : intent.kind,
        notation: "standard",
        numberingSystem: "latn",
        signDisplay: "auto",
        ...(intent.kind === "currency"
          ? {
              currency: intent.currency,
              currencyDisplay: intent.currencyDisplay,
              currencySign: "standard",
            }
          : {}),
        ...(intent.precision?.minimum !== undefined
          ? { minimumFractionDigits: intent.precision.minimum }
          : {}),
        ...(intent.precision?.maximum !== undefined
          ? { maximumFractionDigits: intent.precision.maximum }
          : {}),
        ...(intent.grouping !== undefined ? { useGrouping: intent.grouping } : {}),
      };

      formatter = new Intl.NumberFormat(resolvedLocale, options);

      compiled = {
        kind: "number",
        locale: resolvedLocale,
        options: Object.freeze(options),
        negative: intent.negative,
      };

      parentheses = intent.negative === "parentheses";
      if (parentheses) transform = Math.abs;
    }

    const parts = (value: number): Result<{ parts: readonly Part[] }> => {
      if (
        typeof value !== "number" ||
        !Number.isFinite(value) ||
        (isDateTime && (!Number.isSafeInteger(value) || Math.abs(value) > 8640000000000000))
      ) {
        return {
          supported: false,
          code: "invalid-value",
          reason: isDateTime
            ? "provide a valid whole-millisecond timestamp"
            : "provide a finite number",
        };
      }

      try {
        const result: Part[] = formatter.formatToParts(transform(value));
        const negative = value < 0 || Object.is(value, -0);

        return {
          supported: true,
          parts:
            parentheses && negative
              ? [{ type: "literal", value: "(" }, ...result, { type: "literal", value: ")" }]
              : result,
        };
      } catch (error) {
        return failure(error);
      }
    };

    // Formatting always uses the same checked parts, so text cannot disagree with them.
    const format = (value: number): Result<{ text: string }> => {
      const result = parts(value);

      return result.supported
        ? { supported: true, text: result.parts.map((p) => p.value).join("") }
        : result;
    };

    return {
      supported: true,
      formatter: Object.freeze({ ...compiled, format, formatToParts: parts }),
    };
  } catch (error) {
    return failure(error);
  }
}
