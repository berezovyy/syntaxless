import { compareScalars, quoteIdentifier, sqlGuard, validScalar } from "../../shared/values.js";
import { validate } from "./catalog.js";
import { InputError } from "./types.js";
import type { Context, Field, Result, SortIntent, SortKey } from "./types.js";

export type RecordValue = Readonly<Record<string, unknown>>;

export function comparator(
  intent: SortIntent,
  context: Context,
): (a: RecordValue, b: RecordValue) => number {
  const fields = validate(intent, context);
  const keys = [...intent.keys];

  if (intent.tieBreaker && !keys.some((key) => key.field === intent.tieBreaker)) {
    keys.push({
      field: intent.tieBreaker,
      direction: "asc",
      nulls: "last",
      nullsSource: "explicit",
    });
  }

  const value = (row: RecordValue, field: Field) =>
    Object.hasOwn(row, field.id) && validScalar(row[field.id], field.type)
      ? (row[field.id] as string | number | boolean)
      : null;

  return (a, b) => {
    for (const key of keys) {
      const field = fields.get(key.field)!;
      const left = value(a, field);
      const right = value(b, field);

      if (left === null || right === null) {
        if (left === right) continue;

        return (left === null ? 1 : -1) * (key.nulls === "first" ? -1 : 1);
      }

      const order = compareScalars(left, right) * (key.direction === "asc" ? 1 : -1);

      if (order) return order;
    }

    return 0;
  };
}

export function sortRecords<T extends RecordValue>(
  records: readonly T[],
  intent: SortIntent,
  context: Context,
): T[] {
  const compare = comparator(intent, context);

  if (intent.tieBreaker) {
    const field = catalogTie(intent, context);
    const seen = new Set<unknown>();

    for (const row of records) {
      const value = row[field.id];

      if (!Object.hasOwn(row, field.id) || !validScalar(value, field.type) || seen.has(value)) {
        throw new InputError("invalid-intent", "tie-breaker values must be unique and non-null");
      }

      seen.add(value);
    }
  }

  return [...records].sort(compare);
}

function catalogTie(intent: SortIntent, context: Context): Field {
  return validate(intent, context).get(intent.tieBreaker!)!;
}

export function compile(
  intent: SortIntent,
  context: Context & { dialect?: string },
): Result<{ sql: string; dialect: "sqlite" }> {
  try {
    if (context?.dialect !== undefined && context.dialect !== "sqlite") {
      throw new InputError("unsupported-dialect", "only SQLite is supported");
    }

    const fields = validate(intent, context);

    if (!intent.tieBreaker) {
      throw new InputError(
        "missing-tie-breaker",
        "provide a unique non-null caller tie-breaker for reproducible SQL ties",
      );
    }

    const keys: SortKey[] = [...intent.keys];

    if (!keys.some((key) => key.field === intent.tieBreaker)) {
      keys.push({
        field: intent.tieBreaker,
        direction: "asc",
        nulls: "last",
        nullsSource: "explicit",
      });
    }

    const terms = keys.map((key) => {
      const field = fields.get(key.field)!;
      let column;

      try {
        column = quoteIdentifier(field.column);
      } catch {
        throw new InputError(
          "missing-sql-mapping",
          `provide trusted identifier segments for ${field.id}`,
        );
      }

      const value = `(CASE WHEN ${sqlGuard(column, field.type)} THEN ${column} ELSE NULL END)`;

      const collation = field.type === "string" ? " COLLATE BINARY" : "";
      const order = `${key.direction.toUpperCase()} NULLS ${key.nulls.toUpperCase()}`;

      return `${value}${collation} ${order}`;
    });

    return { supported: true, dialect: "sqlite", sql: terms.join(", ") };
  } catch (error) {
    if (error instanceof InputError) {
      return { supported: false, code: error.code, reason: error.message };
    }

    throw error;
  }
}
