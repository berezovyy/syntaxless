/** Exact scalar comparison and trusted SQLite identifiers shared by filter and sort. */
export type ScalarType = "string" | "number" | "boolean";

export function validScalar(value: unknown, type: ScalarType): boolean {
  if (typeof value !== type) return false;

  if (typeof value === "number") {
    return Number.isFinite(value) && Math.abs(value) <= Number.MAX_SAFE_INTEGER;
  }

  if (typeof value === "string") return !/[\uD800-\uDFFF]/u.test(value) && !value.includes("\0");

  return typeof value === "boolean";
}

/** Unicode scalar ordering agrees with SQLite BINARY on valid Unicode text. */
export function compareScalars(a: string | number | boolean, b: string | number | boolean): number {
  if (typeof a !== "string" && typeof b !== "string") return a < b ? -1 : a > b ? 1 : 0;

  const left = [...String(a)];
  const right = [...String(b)];

  for (let i = 0; i < Math.min(left.length, right.length); i++) {
    const delta = left[i]!.codePointAt(0)! - right[i]!.codePointAt(0)!;

    if (delta) return Math.sign(delta);
  }

  return Math.sign(left.length - right.length);
}

export function quoteIdentifier(mapping: string | readonly string[] | undefined): string {
  const parts = typeof mapping === "string" ? [mapping] : mapping;

  if (
    !parts ||
    !Array.isArray(parts) ||
    parts.length < 1 ||
    parts.length > 3 ||
    parts.some((part) => typeof part !== "string" || !part || part.includes("\0"))
  ) {
    throw new TypeError("provide trusted SQL identifier segments");
  }

  return parts.map((part) => `"${part.replaceAll('"', '""')}"`).join(".");
}

export function sqlGuard(column: string, type: ScalarType): string {
  return type === "string"
    ? `typeof(${column}) = 'text' AND instr(${column}, char(0)) = 0`
    : type === "boolean"
      ? `typeof(${column}) = 'integer' AND ${column} IN (0, 1)`
      : `typeof(${column}) IN ('integer', 'real') AND abs(${column}) <= 9007199254740991`;
}
