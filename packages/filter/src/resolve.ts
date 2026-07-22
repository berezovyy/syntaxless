/** Deterministic grammar over predicted roles; never interpret operator spelling. */
import { InputError } from "./catalog.js";
import type { Role } from "./labels.js";
import type { Lexeme, Prepared } from "./prepare.js";
import type { CompareOperator, Context, FilterIntent, Scalar } from "./types.js";
import { validate } from "./validate.js";

export class AmbiguousGrouping extends Error {}

export interface Labeled {
  lexeme: Lexeme;
  role: Role;
  score: number;
}

const comparison: Partial<Record<Role, CompareOperator>> = {
  EQ: "eq",
  NE: "ne",
  LT: "lt",
  LE: "le",
  GT: "gt",
  GE: "ge",
};

export function resolveRoles(
  prepared: Prepared,
  labels: readonly { role: Role; score: number }[],
  context: Context,
): { intent: FilterIntent; confidence: number } {
  if (labels.length !== prepared.lexemes.length) {
    throw new Error("prediction length does not match input");
  }

  const tokens: Labeled[] = [];
  let confidence = 1;

  for (const [index, lexeme] of prepared.lexemes.entries()) {
    const prediction = labels[index]!;

    if (!Number.isFinite(prediction.score) || prediction.score < 0 || prediction.score > 1) {
      throw new Error("invalid prediction confidence");
    }

    if (prediction.role === "UNSUPPORTED") {
      throw new InputError("invalid-intent", "unsupported or non-actionable input");
    }

    if (
      (lexeme.field !== undefined && prediction.role !== "FIELD") ||
      (lexeme.value !== undefined && prediction.role !== "VALUE")
    ) {
      throw new InputError("invalid-intent", "model did not preserve a caller field or literal");
    }

    if (prediction.role === "O") continue;
    confidence = Math.min(confidence, prediction.score);
    const previous = tokens.at(-1);

    if (
      previous?.role === prediction.role &&
      !["FIELD", "VALUE", "LPAREN", "RPAREN"].includes(prediction.role)
    ) {
      continue;
    }

    tokens.push({ lexeme, ...prediction });
  }

  let position = 0;
  let atoms = 0;
  const peek = () => tokens[position]?.role;

  const take = (role: Role): Labeled => {
    const token = tokens[position++];

    if (!token || token.role !== role) throw new InputError("invalid-intent", `expected ${role}`);

    return token;
  };

  function value(): Scalar {
    const token = take("VALUE");

    if (token.lexeme.value === undefined) {
      throw new InputError("invalid-intent", "use an exact number, boolean, or JSON-quoted string");
    }

    return token.lexeme.value;
  }

  function atom(): FilterIntent {
    if (++atoms > 64) throw new InputError("invalid-intent", "predicate exceeds 64 nodes");

    if (peek() === "NOT") {
      take("NOT");

      return { kind: "not", child: atom() };
    }

    if (peek() === "LPAREN") {
      take("LPAREN");
      const intent = group();

      take("RPAREN");

      return intent;
    }

    const field = take("FIELD").lexeme.field;

    if (!field) throw new InputError("unknown-field", "unknown field mention");
    const role = peek();

    if (!role) throw new InputError("invalid-intent", "predicate lacks an operator");
    const op = comparison[role];

    if (op) {
      take(role);

      return { kind: "compare", field, op, value: value() };
    }

    if (role === "IS_NULL" || role === "IS_NOT_NULL") {
      take(role);

      return { kind: "null", field, isNull: role === "IS_NULL" };
    }

    if (role === "BETWEEN") {
      take(role);
      const lower = value();

      take("GLUE");
      const upper = value();

      if (typeof lower === "boolean" || typeof upper === "boolean") {
        throw new InputError("invalid-intent", "boolean ranges are unsupported");
      }

      return {
        kind: "range",
        field,
        lower: { value: lower, inclusive: true },
        upper: { value: upper, inclusive: true },
      };
    }

    if (role === "IN") {
      take(role);
      const values = [value()];

      while (peek() === "GLUE") {
        take("GLUE");
        values.push(value());
      }

      return { kind: "in", field, values };
    }

    throw new InputError("invalid-intent", "unrecognized predicate operator");
  }

  function group(): FilterIntent {
    const children = [atom()];
    let connective: "and" | "or" | undefined;

    while (position < tokens.length && peek() !== "RPAREN") {
      const role = peek();
      const kind = role === "OR" ? "or" : "and";

      if (connective && connective !== kind) {
        throw new AmbiguousGrouping("use parentheses to group mixed AND and OR");
      }

      connective = kind;
      if (role === "AND" || role === "OR") take(role);

      if (peek() === "EXCLUDE") {
        take("EXCLUDE");
        children.push({ kind: "not", child: atom() });
      } else {
        children.push(atom());
      }
    }

    return children.length === 1 ? children[0]! : { kind: connective!, children };
  }

  const intent = group();

  if (position !== tokens.length) {
    throw new InputError("invalid-intent", "unexpected closing group");
  }

  validate(intent, context);

  return { intent, confidence };
}
