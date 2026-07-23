export type Direction = "asc" | "desc";
export type NullPlacement = "first" | "last";

export interface Field {
  id: string;
  type: "string" | "number" | "boolean";
  aliases?: readonly string[];
  orderingAliases?: readonly { text: string; direction: Direction }[];
  column?: string | readonly string[];
}

export interface Context {
  fields: readonly Field[];
  defaultNulls?: NullPlacement;
  /** Unique, non-null field for reproducible JavaScript/SQL ties. */
  tieBreaker?: string;
}

export interface SortKey {
  field: string;
  direction: Direction;
  nulls: NullPlacement;
  nullsSource: "explicit" | "caller-default";
}

export interface SortIntent {
  keys: readonly SortKey[];
  tieBreaker?: string;
}

export type FailureCode =
  | "invalid-input"
  | "ambiguous"
  | "invalid-catalog"
  | "invalid-intent"
  | "unknown-field"
  | "conflict"
  | "missing-direction"
  | "missing-null-placement"
  | "missing-sql-mapping"
  | "missing-tie-breaker"
  | "unsupported-dialect";

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
