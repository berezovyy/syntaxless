import { InputError } from "./types.js";
import type { Context, Jitter, PolicyField, RetryIntent } from "./types.js";

export const FIELDS: readonly PolicyField[] = [
  "retries",
  "strategy",
  "initialDelayMs",
  "multiplier",
  "maxDelayMs",
  "jitter",
];

function safeDelay(value: unknown): boolean {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

export function validateJitter(value: Jitter): void {
  if (
    !value ||
    typeof value !== "object" ||
    !Object.hasOwn(value, "kind") ||
    !["none", "full", "equal", "proportional"].includes(value.kind) ||
    Object.keys(value).some(
      (key) => !["kind", ...(value.kind === "proportional" ? ["fraction"] : [])].includes(key),
    )
  ) {
    throw new InputError("invalid-intent", "provide none, full, equal, or proportional jitter");
  }

  if (
    value.kind === "proportional" &&
    (!Object.hasOwn(value, "fraction") ||
      typeof value.fraction !== "number" ||
      !Number.isFinite(value.fraction) ||
      value.fraction < 0 ||
      value.fraction > 1)
  ) {
    throw new InputError("invalid-intent", "jitter fraction must be between zero and one");
  }
}

function property(field: PolicyField, value: unknown): void {
  if (
    field === "retries" &&
    !(typeof value === "number" && Number.isSafeInteger(value) && value >= 0 && value <= 1000)
  ) {
    throw new InputError("invalid-intent", "retries must be an integer from zero to 1000");
  }

  if (field === "strategy" && !["constant", "exponential"].includes(value as string)) {
    throw new InputError("invalid-intent", "unknown delay strategy");
  }

  if ((field === "initialDelayMs" || field === "maxDelayMs") && !safeDelay(value)) {
    throw new InputError("invalid-intent", "delays must be nonnegative safe whole milliseconds");
  }

  if (
    field === "multiplier" &&
    !(typeof value === "number" && Number.isFinite(value) && value >= 1 && value <= 1000000)
  ) {
    throw new InputError(
      "invalid-intent",
      "multiplier must be finite and between one and one million",
    );
  }

  if (field === "jitter") validateJitter(value as Jitter);
}

export function validateContext(context: Context): void {
  if (!context || typeof context !== "object") {
    throw new InputError("invalid-defaults", "provide caller context");
  }

  const defaults = context.defaults;

  if (defaults === undefined) return;

  if (
    !defaults ||
    typeof defaults !== "object" ||
    Array.isArray(defaults) ||
    Object.keys(defaults).some((key) => !FIELDS.includes(key as PolicyField))
  ) {
    throw new InputError("invalid-defaults", "invalid caller defaults");
  }

  try {
    for (const field of FIELDS) {
      if (Object.hasOwn(defaults, field)) property(field, defaults[field]);
    }
  } catch (error) {
    if (error instanceof InputError) throw new InputError("invalid-defaults", error.message);
    throw error;
  }
}

export function validate(intent: RetryIntent): void {
  if (
    !intent ||
    typeof intent !== "object" ||
    ["retries", "strategy", "initialDelayMs", "defaults"].some(
      (field) => !Object.hasOwn(intent, field),
    ) ||
    Object.keys(intent).some((key) => ![...FIELDS, "defaults"].includes(key)) ||
    !Array.isArray(intent.defaults) ||
    new Set(intent.defaults).size !== intent.defaults.length ||
    intent.defaults.some((field) => !FIELDS.includes(field) || !Object.hasOwn(intent, field))
  ) {
    throw new InputError("invalid-intent", "invalid intent or default provenance");
  }

  for (const field of FIELDS) {
    if (["retries", "strategy", "initialDelayMs"].includes(field) || Object.hasOwn(intent, field)) {
      property(field, intent[field]);
    }
  }

  if (intent.strategy === "exponential" && intent.multiplier === undefined) {
    throw new InputError("missing-multiplier", "provide an exponential multiplier");
  }

  if (intent.strategy === "constant" && intent.multiplier !== undefined) {
    throw new InputError("conflict", "constant delays cannot have a multiplier");
  }
}
