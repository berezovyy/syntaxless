import { quoteIdentifier, sqlGuard } from "../../shared/values.js";
import { InputError } from "./catalog.js";
import type { Context, Field, FilterIntent, Result, Scalar } from "./types.js";
import { validate } from "./validate.js";

const SQL_OPERATORS = { eq: "=", ne: "<>", lt: "<", le: "<=", gt: ">", ge: ">=" } as const;

export interface SqlPredicate {
  sql: string;
  parameters: readonly (string | number)[];
  dialect: "sqlite";
}

export function compile(
  intent: FilterIntent,
  context: Context & { dialect?: string },
): Result<SqlPredicate> {
  try {
    if (context?.dialect !== undefined && context.dialect !== "sqlite") {
      throw new InputError("unsupported-dialect", "only SQLite is supported");
    }

    const fields = validate(intent, context);
    const parameters: (string | number)[] = [];

    function bind(value: Scalar): string {
      parameters.push(typeof value === "boolean" ? Number(value) : value);

      return `?${parameters.length}`;
    }

    function identifier(field: Field): string {
      try {
        return quoteIdentifier(field.column);
      } catch {
        throw new InputError(
          "missing-sql-mapping",
          `provide trusted SQL identifier segments for ${field.id}`,
        );
      }
    }

    function visit(node: FilterIntent): string {
      if (node.kind === "not") return `(NOT ${visit(node.child)})`;

      if (node.kind === "and" || node.kind === "or") {
        return `(${node.children.map(visit).join(node.kind === "and" ? " AND " : " OR ")})`;
      }

      const predicate = node;

      const field = fields.get(predicate.field)!;
      const column = identifier(field);

      if (predicate.kind === "null") return `(${column} IS ${predicate.isNull ? "" : "NOT "}NULL)`;
      const operand = field.type === "string" ? `${column} COLLATE BINARY` : column;
      let expression: string;

      if (predicate.kind === "compare") {
        expression = `${operand} ${SQL_OPERATORS[predicate.op]} ${bind(predicate.value)}`;
      } else if (predicate.kind === "in") {
        expression = `${operand} IN (${predicate.values.map(bind).join(", ")})`;
      } else {
        const { lower, upper } = predicate;
        // Bind the lower bound first so parameter numbers follow the SQL text.
        const lowerBound = `${operand} ${lower.inclusive ? ">=" : ">"} ${bind(lower.value)}`;
        const upperBound = `${operand} ${upper.inclusive ? "<=" : "<"} ${bind(upper.value)}`;

        expression = `(${lowerBound} AND ${upperBound})`;
      }

      const guard = sqlGuard(column, field.type);

      return `(CASE WHEN ${guard} THEN ${expression} ELSE NULL END)`;
    }

    return { supported: true, dialect: "sqlite", sql: visit(intent), parameters };
  } catch (error) {
    if (error instanceof InputError) {
      return { supported: false, code: error.code, reason: error.message };
    }

    throw error;
  }
}
