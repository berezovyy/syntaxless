import { validScalar } from "../../shared/values.js";
import { BUCKETS, InputError } from "./types.js";
import type { Backup, Context, RetentionIntent } from "./types.js";

const keys = (value: unknown, required: readonly string[]): boolean =>
  !!value &&
  typeof value === "object" &&
  !Array.isArray(value) &&
  Object.keys(value).length === required.length &&
  required.every((key) => Object.hasOwn(value, key));

const positive = (value: unknown): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value > 0;

/** Supported metadata spans Unix epoch through the end of Gregorian year 9999. */
export const MAX_TIMESTAMP = 253402300799999;

function timestamp(value: unknown): value is number {
  return (
    typeof value === "number" && Number.isSafeInteger(value) && value >= 0 && value <= MAX_TIMESTAMP
  );
}

export function validate(intent: RetentionIntent): void {
  if (
    !keys(intent, ["tiers"]) ||
    !Array.isArray(intent.tiers) ||
    intent.tiers.length < 1 ||
    intent.tiers.length > 4
  ) {
    throw new InputError("invalid-intent", "provide one to four retention tiers");
  }

  const seen = new Set<string>();

  for (const tier of intent.tiers) {
    if (!keys(tier, ["bucket", "window"]) || !BUCKETS.includes(tier.bucket)) {
      throw new InputError("invalid-intent", "unknown retention bucket");
    }

    if (seen.has(tier.bucket)) {
      throw new InputError("conflict", "provide at most one tier for each bucket kind");
    }

    seen.add(tier.bucket);
    const window = tier.window;

    if (window?.kind === "calendar") {
      if (
        !keys(window, ["kind", "unit", "count"]) ||
        !BUCKETS.includes(window.unit) ||
        !positive(window.count) ||
        window.count > 10000
      ) {
        throw new InputError(
          "invalid-intent",
          "calendar windows require one to 10000 calendar periods",
        );
      }
    } else if (window?.kind === "elapsed") {
      if (!keys(window, ["kind", "milliseconds"]) || !positive(window.milliseconds)) {
        throw new InputError(
          "invalid-intent",
          "elapsed windows require positive safe whole milliseconds",
        );
      }
    } else {
      throw new InputError("invalid-intent", "unknown retention window");
    }
  }
}

export function calendar(context: Context): Intl.DateTimeFormat {
  if (!context || context.timeZone === undefined || context.referenceInstant === undefined) {
    throw new InputError(
      "missing-context",
      "provide an explicit caller timezone and reference instant",
    );
  }

  if (
    !Object.hasOwn(context, "timeZone") ||
    !Object.hasOwn(context, "referenceInstant") ||
    typeof context.timeZone !== "string" ||
    !context.timeZone ||
    !timestamp(context.referenceInstant)
  ) {
    throw new InputError("invalid-context", "invalid timezone or reference instant");
  }

  try {
    return new Intl.DateTimeFormat("en-US-u-ca-gregory-nu-latn", {
      timeZone: context.timeZone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });
  } catch {
    throw new InputError("invalid-context", "timezone is not supported by this runtime");
  }
}

export function validateBackups(backups: readonly Backup[], reference: number): void {
  if (!Array.isArray(backups) || backups.length > 100000) {
    throw new InputError("invalid-backups", "provide at most 100000 backups");
  }

  const ids = new Set<string>();

  for (const backup of backups) {
    if (
      !keys(backup, ["id", "timestamp"]) ||
      typeof backup.id !== "string" ||
      !backup.id ||
      backup.id.length > 256 ||
      !validScalar(backup.id, "string") ||
      ids.has(backup.id) ||
      !timestamp(backup.timestamp) ||
      backup.timestamp > reference
    ) {
      throw new InputError(
        "invalid-backups",
        "backup IDs must be unique valid strings and timestamps must not be in the future",
      );
    }

    ids.add(backup.id);
  }
}
