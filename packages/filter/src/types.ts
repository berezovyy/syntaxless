export type Scalar = string | number | boolean;
export type FieldType = "string" | "number" | "boolean";
export type CompareOperator = "eq" | "ne" | "lt" | "le" | "gt" | "ge";
export type Operator = CompareOperator | "in" | "range" | "null";

export interface Field {
  id: string;
  type: FieldType;
  aliases?: readonly string[];
  /** Caller-defined shorthand; e.g. "active customers" means status = "active". */
  terms?: readonly { text: string; value: Scalar }[];
  operators?: readonly Operator[];
  /** Trusted identifier segments, never SQL expressions. */
  column?: string | readonly string[];
}

export interface Context {
  fields: readonly Field[];
}

export interface Bound {
  value: string | number;
  inclusive: boolean;
}

export type FilterIntent =
  | { kind: "compare"; field: string; op: CompareOperator; value: Scalar }
  | { kind: "in"; field: string; values: readonly Scalar[] }
  | { kind: "range"; field: string; lower: Bound; upper: Bound }
  | { kind: "null"; field: string; isNull: boolean }
  | { kind: "and"; children: readonly FilterIntent[] }
  | { kind: "or"; children: readonly FilterIntent[] }
  | { kind: "not"; child: FilterIntent };

export type Truth = boolean | null;

export type FailureCode =
  | "invalid-input"
  | "ambiguous"
  | "invalid-catalog"
  | "invalid-intent"
  | "unknown-field"
  | "unsupported-operator"
  | "missing-sql-mapping"
  | "unsupported-dialect";

export type Result<T> =
  ({ supported: true } & T) | { supported: false; code: FailureCode; reason: string };

/** A refusal with a machine-readable code; never thrown for internal faults. */
export class InputError extends Error {
  constructor(
    readonly code: FailureCode,
    message: string,
  ) {
    super(message);
  }
}
