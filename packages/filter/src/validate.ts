import { catalog, compare, InputError, scalar } from "./catalog.js";
import type { Context, Field, FilterIntent, Operator } from "./types.js";

export function validate(intent: FilterIntent, context: Context): Map<string, Field> {
  const fields = catalog(context);
  let nodes = 0;

  function fail(reason: string): never {
    throw new InputError("invalid-intent", reason);
  }

  function object(
    value: unknown,
    keys: readonly string[],
  ): asserts value is Record<string, unknown> {
    if (
      !value ||
      typeof value !== "object" ||
      Array.isArray(value) ||
      Object.keys(value).some((key) => !keys.includes(key))
    ) {
      fail("unexpected intent properties");
    }
  }

  function visit(node: FilterIntent, depth: number): void {
    if (++nodes > 64 || depth > 8) fail("intent exceeds 64 nodes or eight nesting levels");
    if (!node || typeof node !== "object") fail("intent must be an object");

    if (node.kind === "and" || node.kind === "or") {
      object(node, ["kind", "children"]);

      if (!Array.isArray(node.children) || node.children.length < 2) {
        fail("boolean groups need at least two children");
      }

      for (const child of node.children) visit(child, depth + 1);

      return;
    }

    if (node.kind === "not") {
      object(node, ["kind", "child"]);
      visit(node.child, depth + 1);

      return;
    }

    if (!["compare", "in", "range", "null"].includes(node.kind)) fail("unknown predicate kind");
    const predicate = node;
    const field = fields.get(predicate.field);

    if (!field) throw new InputError("unknown-field", `unknown field: ${predicate.field}`);
    const op: Operator = predicate.kind === "compare" ? predicate.op : predicate.kind;

    if (
      !(["eq", "ne", "lt", "le", "gt", "ge", "in", "range", "null"] as string[]).includes(op) ||
      (field.operators && !field.operators.includes(op)) ||
      (field.type === "boolean" && ["lt", "le", "gt", "ge", "range"].includes(op))
    ) {
      throw new InputError("unsupported-operator", `operator ${op} is unavailable for ${field.id}`);
    }

    if (predicate.kind === "compare") {
      object(predicate, ["kind", "field", "op", "value"]);
      if (!scalar(predicate.value, field)) fail("comparison value does not match its field");
    } else if (predicate.kind === "in") {
      object(predicate, ["kind", "field", "values"]);

      if (
        !Array.isArray(predicate.values) ||
        predicate.values.length < 1 ||
        predicate.values.length > 64 ||
        predicate.values.some((value) => !scalar(value, field))
      ) {
        fail("membership needs one to 64 typed non-null values");
      }
    } else if (predicate.kind === "range") {
      object(predicate, ["kind", "field", "lower", "upper"]);

      for (const bound of [predicate.lower, predicate.upper]) {
        object(bound, ["value", "inclusive"]);

        if (!scalar(bound.value, field) || typeof bound.inclusive !== "boolean") {
          fail("invalid range bound");
        }
      }

      const order = compare(predicate.lower.value, predicate.upper.value);

      if (
        order > 0 ||
        (order === 0 && (!predicate.lower.inclusive || !predicate.upper.inclusive))
      ) {
        fail("range is empty or inverted");
      }
    } else {
      object(predicate, ["kind", "field", "isNull"]);
      if (typeof predicate.isNull !== "boolean") fail("null check requires an explicit boolean");
    }
  }

  visit(intent, 0);

  return fields;
}
