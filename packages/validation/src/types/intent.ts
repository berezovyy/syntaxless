/**
 * Numeric bound. Inclusive vs exclusive is part of the AST so
 * "more than 10" round-trips; JSON Schema keywords are a compile concern.
 */
export interface Bound {
  value: number;
  inclusive: boolean;
}

/** Numeric range. Missing side = open. Exact = min and max equal and inclusive. */
export interface ComparisonIntent {
  min?: Bound;
  max?: Bound;
}

export type FieldType = "string" | "email" | "url" | "uuid" | "number" | "integer" | "boolean";

export type StringLikeType = "string" | "email" | "url" | "uuid";
export type NumberLikeType = "number" | "integer";

/**
 * One field constraint. Freeze:
 * - v0 sentences must state required or optional; the object still always has
 *   `required: boolean` once parsed. Do not default.
 * - Email is a type, not a `format` flag on the AST.
 * - `minimum`/`maximum` stay Bound, not bare numbers.
 * - Length is a bare integer (always inclusive). Exclusive length is widened
 *   in the comparison→field resolver, not stored as exclusive length.
 * - At most one comparison per slot (length vs value). Conflicts fail in validate.
 */
export interface FieldIntent {
  type: FieldType;
  required: boolean;
  nullable?: boolean;
  minLength?: number;
  maxLength?: number;
  minimum?: Bound;
  maximum?: Bound;
}
