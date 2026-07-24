import { compareScalars } from "../../shared/values.js";
import { InputError } from "./types.js";
import type { Bucket, CompileOptions, Preview, Result, RetentionIntent } from "./types.js";
import { calendar, validate, validateBackups } from "./validate.js";

function indexes(format: Intl.DateTimeFormat, timestamp: number): Record<Bucket, number> {
  const parts = format.formatToParts(timestamp);
  const part = (type: string): number => Number(parts.find((p) => p.type === type)?.value);

  const year = part("year");
  const month = part("month");
  const date = part("day");

  if (![year, month, date].every(Number.isInteger) || year < 1 || year > 9999) {
    throw new InputError("unsupported", "local calendar date is outside supported Gregorian years");
  }

  // Local midnight is represented on a UTC calendar only to count civil dates.
  const civil = new Date(0);

  civil.setUTCFullYear(year, month - 1, date);
  const day = civil.getTime() / 86400000;

  return { day, week: Math.floor((day + 3) / 7), month: year * 12 + month - 1, year };
}

export function compile(intent: RetentionIntent, options: CompileOptions): Result<Preview> {
  try {
    validate(intent);
    const format = calendar(options);

    validateBackups(options.backups, options.referenceInstant);
    const current = indexes(format, options.referenceInstant);
    const dates = options.backups.map((backup) => indexes(format, backup.timestamp));
    const selections = options.backups.map(() => [] as number[]);

    for (const [tierIndex, tier] of intent.tiers.entries()) {
      const newest = new Map<number, number>();

      for (const [index, backup] of options.backups.entries()) {
        const bucket = dates[index]![tier.bucket];

        const eligible =
          tier.window.kind === "calendar"
            ? dates[index]![tier.window.unit] > current[tier.window.unit] - tier.window.count &&
              dates[index]![tier.window.unit] <= current[tier.window.unit]
            : backup.timestamp >= options.referenceInstant - tier.window.milliseconds;

        if (!eligible) continue;
        const prior = newest.get(bucket);

        if (
          prior === undefined ||
          backup.timestamp > options.backups[prior]!.timestamp ||
          (backup.timestamp === options.backups[prior]!.timestamp &&
            compareScalars(backup.id, options.backups[prior]!.id) < 0)
        ) {
          newest.set(bucket, index);
        }
      }

      for (const index of newest.values()) selections[index]!.push(tierIndex);
    }

    const keep: { id: string; tierIndexes: readonly number[] }[] = [];
    const remove: string[] = [];

    for (const [index, backup] of options.backups.entries()) {
      const tierIndexes = selections[index]!;

      if (tierIndexes.length) keep.push({ id: backup.id, tierIndexes });
      else remove.push(backup.id);
    }

    return { supported: true, keep, remove };
  } catch (error) {
    if (error instanceof InputError) {
      return { supported: false, code: error.code, reason: error.message };
    }

    throw error;
  }
}
