import { compare, scalar } from "./catalog.js";
import type { Context, FilterIntent, Truth } from "./types.js";
import { validate } from "./validate.js";

/** Null/missing or wrong-typed data produces SQL-style unknown, never coercion. */
export function evaluateTruth(
  intent: FilterIntent,
  record: Readonly<Record<string, unknown>>,
  context: Context,
): Truth {
  const fields = validate(intent, context);

  function visit(node: FilterIntent): Truth {
    if (node.kind === "not") {
      const value = visit(node.child);

      return value === null ? null : !value;
    }

    if (node.kind === "and" || node.kind === "or") {
      const values = node.children.map(visit);

      return node.kind === "and"
        ? values.includes(false)
          ? false
          : values.includes(null)
            ? null
            : true
        : values.includes(true)
          ? true
          : values.includes(null)
            ? null
            : false;
    }

    const predicate = node;
    const value = Object.hasOwn(record, predicate.field) ? record[predicate.field] : null;

    if (predicate.kind === "null") {
      return (value === null || value === undefined) === predicate.isNull;
    }

    if (!scalar(value, fields.get(predicate.field)!)) return null;
    if (predicate.kind === "in") return predicate.values.some((member) => value === member);

    if (predicate.kind === "compare") {
      if (predicate.op === "eq") return value === predicate.value;
      if (predicate.op === "ne") return value !== predicate.value;
      const order = compare(value as string | number, predicate.value as string | number);

      return predicate.op === "lt"
        ? order < 0
        : predicate.op === "le"
          ? order <= 0
          : predicate.op === "gt"
            ? order > 0
            : order >= 0;
    }

    const lower = compare(value as string | number, predicate.lower.value);
    const upper = compare(value as string | number, predicate.upper.value);

    return (
      (lower > 0 || (lower === 0 && predicate.lower.inclusive)) &&
      (upper < 0 || (upper === 0 && predicate.upper.inclusive))
    );
  }

  return visit(intent);
}

/** A record matches only when its complete expression is true. */
export function evaluate(
  intent: FilterIntent,
  record: Readonly<Record<string, unknown>>,
  context: Context,
): boolean {
  return evaluateTruth(intent, record, context) === true;
}
