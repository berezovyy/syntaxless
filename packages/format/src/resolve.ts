import type { Role } from "./labels.js";
import type { Lexeme, Prepared } from "./prepare.js";
import { InputError } from "./types.js";
import type { Context, Currency, FormatIntent, Precision, Style } from "./types.js";
import { locale as canonicalLocale, timeZone as canonicalZone, validate } from "./validate.js";

export interface Labeled {
  role: Role;
  score: number;
}

const markerRoles: Record<NonNullable<Lexeme["marker"]>, readonly Role[]> = {
  number: ["NUMBER"],
  percent: ["PERCENT"],
  date: ["DATE"],
  time: ["TIME"],
  datetime: ["DATETIME"],
  precision: ["PRECISION"],
  grouping: ["GROUP", "NOGROUP"],
  parentheses: ["PARENS"],
  minus: ["MINUS"],
};

export function resolveRoles(
  prepared: Prepared,
  labeled: readonly Labeled[],
  _context: unknown = {},
): { intent: FormatIntent; confidence: number } {
  if (labeled.length !== prepared.lexemes.length) {
    throw new InputError("invalid-input", "unaligned format predictions");
  }

  let defaultPrecision = false;

  let numeric = false;
  let percent = false;
  let date = false;
  let time = false;
  let currency: string | undefined;
  let negative: "minus" | "parentheses" | undefined;
  let display: "symbol" | "code" | "name" | undefined;
  let grouping: boolean | undefined;
  let hourCycle: "h12" | "h23" | undefined;
  let style: Style | undefined;
  let dateStyle: Style | undefined;
  let timeStyle: Style | undefined;
  let minimum: number | undefined;
  let maximum: number | undefined;
  let exact: number | undefined;

  const context: Context = {};
  let confidence = 1;

  function once<T>(current: T | undefined, value: T, name: string): T {
    if (current !== undefined && current !== value) {
      throw new InputError("conflict", `conflicting ${name}`);
    }

    return value;
  }

  for (let i = 0; i < labeled.length; i++) {
    const { role, score } = labeled[i]!;
    const lexeme = prepared.lexemes[i]!;

    if (role === "UNSUPPORTED") {
      throw new InputError(
        "unsupported",
        "the request contains unsupported, uncertain or non-actionable instructions",
      );
    }

    if (lexeme.numeric !== undefined && !["EXACT", "MIN", "MAX"].includes(role)) {
      throw new InputError("unsupported", "a numeric formatting fact was dropped");
    }

    if (lexeme.currency !== undefined && role !== "CURRENCY") {
      throw new InputError("unsupported", "a currency fact was dropped");
    }

    if (lexeme.locale !== undefined && role !== "LOCALE") {
      throw new InputError("unsupported", "a locale fact was dropped");
    }

    if (lexeme.timeZone !== undefined && role !== "ZONE") {
      throw new InputError("unsupported", "a timezone fact was dropped");
    }

    if (lexeme.style !== undefined && !["STYLE", "DSTYLE", "TSTYLE"].includes(role)) {
      throw new InputError("unsupported", "a date/time style fact was dropped");
    }

    if (lexeme.hourCycle !== undefined && role !== "HOUR") {
      throw new InputError("unsupported", "an hour cycle fact was dropped");
    }

    if (lexeme.display !== undefined && role !== "DISPLAY") {
      throw new InputError("unsupported", "a currency display fact was dropped");
    }

    if (lexeme.marker !== undefined && !markerRoles[lexeme.marker].includes(role)) {
      throw new InputError("unsupported", "a formatting option fact was dropped");
    }

    if (role === "O" || role === "GLUE") continue;

    if (!Number.isFinite(score) || score < 0 || score > 1) {
      throw new InputError("invalid-input", "invalid model confidence");
    }

    confidence = Math.min(confidence, score);

    switch (role) {
      case "NUMBER":
        numeric = true;
        break;
      case "PERCENT":
        percent = true;
        break;
      case "DATE":
        date = true;
        break;
      case "TIME":
        time = true;
        break;
      case "DATETIME":
        date = true;
        time = true;
        break;
      case "PRECISION":
        break;
      case "DEFAULT_PRECISION":
        if (!["default", "defaults", "native", "overrides"].includes(lexeme.text)) {
          throw new InputError("unsupported", "default precision must be stated");
        }

        defaultPrecision = true;
        break;
      case "CURRENCY":
        if (lexeme.currency === undefined) {
          throw new InputError("unsupported", "currency must be explicit");
        }

        currency = once(currency, lexeme.currency, "currency");
        break;
      case "LOCALE":
        if (lexeme.locale === undefined) {
          throw new InputError("unsupported", "locale must be a literal identifier");
        }

        context.locale = once(context.locale, canonicalLocale(lexeme.locale), "locale");
        break;
      case "ZONE":
        if (lexeme.timeZone === undefined) {
          throw new InputError("unsupported", "timezone must be a literal identifier");
        }

        context.timeZone = once(context.timeZone, canonicalZone(lexeme.timeZone), "timezone");
        break;
      case "DISPLAY":
        if (lexeme.display === undefined) {
          throw new InputError("unsupported", "currency display must be explicit");
        }

        display = once(display, lexeme.display, "currency display");
        break;
      case "HOUR":
        if (lexeme.hourCycle === undefined) {
          throw new InputError("unsupported", "hour cycle must be explicit");
        }

        hourCycle = once(hourCycle, lexeme.hourCycle, "hour cycle");
        break;
      case "STYLE":
      case "DSTYLE":
      case "TSTYLE":
        if (lexeme.style === undefined) {
          throw new InputError("unsupported", "style must be explicit");
        }

        if (role === "STYLE") style = once(style, lexeme.style, "shared style");
        else if (role === "DSTYLE") dateStyle = once(dateStyle, lexeme.style, "date style");
        else timeStyle = once(timeStyle, lexeme.style, "time style");
        break;
      case "GROUP":
      case "NOGROUP":
        grouping = once(grouping, role === "GROUP", "grouping");
        break;
      case "PARENS":
      case "MINUS":
        negative = once(
          negative,
          role === "PARENS" ? "parentheses" : "minus",
          "negative presentation",
        );

        break;
      case "EXACT":
      case "MIN":

      case "MAX": {
        if (lexeme.numeric === undefined || !/^\+?(?:\d+(?:\.0+)?|\.0+)$/u.test(lexeme.numeric)) {
          throw new InputError("unsupported", "decimal places require a whole digit count");
        }

        const n = Number(lexeme.numeric);

        if (!Number.isSafeInteger(n) || n > 20) {
          throw new InputError("unsupported", "decimal places must be from zero to twenty");
        }

        if (role === "EXACT") exact = once(exact, n, "exact decimal places");
        else if (role === "MIN") minimum = once(minimum, n, "minimum decimal places");
        else maximum = once(maximum, n, "maximum decimal places");
        break;
      }
    }

    // Values and semantic markers must come from input facts, never an invented model role.
    if (
      [
        "NUMBER",
        "PERCENT",
        "DATE",
        "TIME",
        "DATETIME",
        "PRECISION",
        "GROUP",
        "NOGROUP",
        "PARENS",
        "MINUS",
      ].includes(role) &&
      lexeme.marker === undefined
    ) {
      throw new InputError("unsupported", "a semantic option was invented");
    }
  }

  let intent: FormatIntent;
  const hasPrecision = exact !== undefined || minimum !== undefined || maximum !== undefined;

  if (date || time) {
    if (
      numeric ||
      percent ||
      currency ||
      hasPrecision ||
      defaultPrecision ||
      grouping !== undefined ||
      negative !== undefined ||
      display !== undefined
    ) {
      throw new InputError("conflict", "numeric and date/time formats cannot be combined");
    }

    if ((dateStyle && !date) || (timeStyle && !time) || (hourCycle && !time)) {
      throw new InputError(
        "conflict",
        "style or hour cycle refers to an unrequested date/time part",
      );
    }

    if (style !== undefined) {
      if (date) dateStyle = once(dateStyle, style, "date style");
      if (time) timeStyle = once(timeStyle, style, "time style");
    }

    intent = {
      kind: "date-time",
      ...context,
      ...(date ? { dateStyle: dateStyle ?? "medium" } : {}),
      ...(time ? { timeStyle: timeStyle ?? "medium" } : {}),
      ...(hourCycle ? { hourCycle } : {}),
    };
  } else {
    if (!numeric && !currency && !percent) {
      throw new InputError(
        "invalid-input",
        "state number, currency, percent, date or time formatting",
      );
    }

    if (currency && percent) {
      throw new InputError("conflict", "currency and percentage cannot be combined");
    }

    if (style || dateStyle || timeStyle || hourCycle || context.timeZone) {
      throw new InputError("conflict", "date/time options require a date/time format");
    }

    if (display && !currency) {
      throw new InputError("conflict", "currency display requires a currency");
    }

    if (defaultPrecision && hasPrecision) {
      throw new InputError("conflict", "native precision conflicts with explicit decimal places");
    }

    if (exact !== undefined) {
      minimum = once(minimum, exact, "minimum decimal places");
      maximum = once(maximum, exact, "maximum decimal places");
    }

    const precision: Precision = {
      ...(minimum === undefined ? {} : { minimum }),
      ...(maximum === undefined ? {} : { maximum }),
    };

    const common = {
      ...context,
      negative: negative ?? "minus",
      ...(hasPrecision ? { precision } : {}),
      ...(grouping === undefined ? {} : { grouping }),
    };

    intent = currency
      ? {
          ...common,
          kind: "currency",
          currency: currency as Currency,
          currencyDisplay: display ?? "symbol",
        }
      : { ...common, kind: percent ? "percent" : "number" };
  }

  validate(intent);

  return { intent, confidence };
}
