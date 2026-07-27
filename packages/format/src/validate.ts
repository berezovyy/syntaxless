import { CURRENCIES, InputError, STYLES } from "./types.js";
import type { Context, FormatIntent } from "./types.js";

const own = (o: object, k: string) => Object.hasOwn(o, k);

function shape(
  value: unknown,
  required: readonly string[],
  optional: readonly string[] = [],
): boolean {
  return (
    !!value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    required.every((k) => own(value, k)) &&
    Object.keys(value).every((k) => required.includes(k) || optional.includes(k)) &&
    optional.every((k) =>
      own(value, k)
        ? (value as Record<string, unknown>)[k] !== undefined
        : (value as Record<string, unknown>)[k] === undefined,
    )
  );
}

export function locale(value: unknown): string {
  if (typeof value !== "string" || !value || value.length > 64) {
    throw new InputError("invalid-context", "provide a supported locale identifier");
  }

  try {
    const canonical = Intl.getCanonicalLocales(value)[0]!;

    if (new Intl.Locale(canonical).baseName !== canonical) {
      throw new InputError("unsupported", "locale extensions are unsupported; use a base locale");
    }

    if (
      !Intl.NumberFormat.supportedLocalesOf([canonical], { localeMatcher: "lookup" }).length ||
      !Intl.DateTimeFormat.supportedLocalesOf([canonical], { localeMatcher: "lookup" }).length
    ) {
      throw new InputError("unsupported", "locale is not supported by this runtime");
    }

    return canonical;
  } catch (error) {
    if (error instanceof InputError) throw error;
    throw new InputError("invalid-context", "invalid locale identifier");
  }
}

export function timeZone(value: unknown): string {
  if (typeof value !== "string" || !value || value.length > 128) {
    throw new InputError("invalid-context", "provide a supported timezone identifier");
  }

  try {
    return new Intl.DateTimeFormat("en-US", { timeZone: value }).resolvedOptions().timeZone;
  } catch {
    throw new InputError("invalid-context", "timezone is not supported by this runtime");
  }
}

export function validateContext(context: Context): void {
  if (!shape(context, [], ["locale", "timeZone"])) {
    throw new InputError("invalid-context", "provide only locale and timezone context");
  }

  if (own(context, "locale")) locale(context.locale);
  if (own(context, "timeZone")) timeZone(context.timeZone);
}

export function validate(intent: FormatIntent): void {
  const common = ["locale", "timeZone"];

  if (intent?.kind === "date-time") {
    if (
      !shape(intent, ["kind"], [...common, "dateStyle", "timeStyle", "hourCycle"]) ||
      (!intent.dateStyle && !intent.timeStyle) ||
      (intent.dateStyle !== undefined && !STYLES.includes(intent.dateStyle)) ||
      (intent.timeStyle !== undefined && !STYLES.includes(intent.timeStyle)) ||
      (intent.hourCycle !== undefined &&
        (!intent.timeStyle || !["h12", "h23"].includes(intent.hourCycle)))
    ) {
      throw new InputError(
        "invalid-intent",
        "provide date/time styles and use hour cycle only with a time style",
      );
    }
  } else if (
    intent?.kind === "number" ||
    intent?.kind === "percent" ||
    intent?.kind === "currency"
  ) {
    const currency = intent.kind === "currency";

    if (
      !shape(
        intent,
        currency ? ["kind", "negative", "currency", "currencyDisplay"] : ["kind", "negative"],
        [...common, "precision", "grouping"],
      ) ||
      !["minus", "parentheses"].includes(intent.negative) ||
      (intent.grouping !== undefined && typeof intent.grouping !== "boolean")
    ) {
      throw new InputError("invalid-intent", "invalid numeric formatting options");
    }

    if (
      currency &&
      (!CURRENCIES.includes(intent.currency) ||
        !["symbol", "code", "name"].includes(intent.currencyDisplay))
    ) {
      throw new InputError("unsupported", "unsupported currency or currency display");
    }

    if (intent.precision !== undefined) {
      const p = intent.precision;

      if (
        !shape(p, [], ["minimum", "maximum"]) ||
        (!own(p, "minimum") && !own(p, "maximum")) ||
        Object.values(p).some((n) => !Number.isInteger(n) || n < 0 || n > 20)
      ) {
        throw new InputError(
          "invalid-intent",
          "precision requires whole digit counts from zero to twenty",
        );
      }

      if (p.minimum !== undefined && p.maximum !== undefined && p.minimum > p.maximum) {
        throw new InputError("conflict", "minimum decimal places exceed maximum decimal places");
      }
    }

    if (own(intent, "timeZone")) {
      throw new InputError("unsupported", "timezone is meaningful only for date/time formatting");
    }
  } else {
    throw new InputError("invalid-intent", "unknown formatting kind");
  }

  if (own(intent, "locale")) locale(intent.locale);
  if (own(intent, "timeZone")) timeZone(intent.timeZone);
}
