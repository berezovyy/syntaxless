import type { FilterIntent } from "./types.js";

const value = (input: unknown) => JSON.stringify(input);

const OPERATOR_WORDS = {
  eq: "equals",
  ne: "does not equal",
  lt: "is less than",
  le: "is at most",
  gt: "is greater than",
  ge: "is at least",
} as const;

export function explain(intent: FilterIntent): string {
  if (intent.kind === "not") return `NOT (${explain(intent.child)})`;

  if (intent.kind === "and" || intent.kind === "or") {
    return `(${intent.children.map(explain).join(intent.kind === "and" ? " AND " : " OR ")})`;
  }

  const predicate = intent;
  const name = value(predicate.field);

  if (predicate.kind === "null") {
    return `${name} ${predicate.isNull ? "is missing or null" : "is present and non-null"}`;
  }

  if (predicate.kind === "in") return `${name} is one of ${predicate.values.map(value).join(", ")}`;

  if (predicate.kind === "range") {
    const { lower, upper } = predicate;
    const lowerWords = lower.inclusive ? "at least" : "greater than";
    const upperWords = upper.inclusive ? "at most" : "less than";

    return `${name} ${lowerWords} ${value(lower.value)} and ${upperWords} ${value(upper.value)}`;
  }

  return `${name} ${OPERATOR_WORDS[predicate.op]} ${value(predicate.value)}`;
}
