import { Database } from "bun:sqlite";
import { expect, test } from "bun:test";

import { compile, evaluateTruth } from "../src/index.js";
import type { Context, FilterIntent } from "../src/index.js";

const context: Context = {
  fields: [
    { id: "amount", type: "number", column: "amount" },
    { id: "label", type: "string", column: 'label"quoted' },
    { id: "trial", type: "boolean", column: "trial" },
  ],
};

const amount = (op: "gt" | "ge" | "eq", value: number): FilterIntent => ({
  kind: "compare",
  field: "amount",
  op,
  value,
});

test(
  "real SQLite and JavaScript agree on comparisons, " +
    "null truth, exclusions, membership and Unicode ordering",
  () => {
    const db = new Database(":memory:");

    try {
      // ANY in a STRICT table preserves stored types instead of coercing strings.
      db.exec(
        'CREATE TABLE records (id INTEGER, amount ANY, "label""quoted" ANY, trial ANY) STRICT',
      );

      const rows: Record<string, unknown>[] = [
        { amount: 100, label: "active", trial: true },
        { amount: 100.25, label: "active", trial: false },
        { amount: -2.75, label: "trial", trial: null },
        { amount: null, label: "active", trial: false },
        { label: "active" },
        { amount: "101", label: 123, trial: "true" },
        { amount: 105, label: "\uE000", trial: false },
        { amount: 106, label: "\u{10000}", trial: true },
        { amount: 9007199254740992, label: "\0", trial: false },
        { amount: 108, label: "x' OR 1=1 --", trial: false },
      ];

      const insert = db.prepare("INSERT INTO records VALUES (?, ?, ?, ?)");

      for (const [id, row] of rows.entries()) {
        const sqlValue = (value: unknown) =>
          value === undefined
            ? null
            : typeof value === "boolean"
              ? Number(value)
              : (value as string | number | null);

        insert.run(id, sqlValue(row.amount), sqlValue(row.label), sqlValue(row.trial));
      }

      const expressions: FilterIntent[] = [
        amount("gt", 100),
        amount("ge", 100),
        { kind: "not", child: amount("gt", 100) },
        { kind: "in", field: "label", values: ["active", "x' OR 1=1 --"] },
        {
          kind: "range",
          field: "amount",
          lower: { value: -2.75, inclusive: false },
          upper: { value: 100, inclusive: true },
        },
        { kind: "null", field: "amount", isNull: true },
        { kind: "null", field: "amount", isNull: false },
        { kind: "compare", field: "label", op: "lt", value: "\u{10000}" },
        { kind: "compare", field: "trial", op: "eq", value: false },
        {
          kind: "and",
          children: [
            amount("gt", 100),
            {
              kind: "not",
              child: { kind: "compare", field: "trial", op: "eq", value: true },
            },
          ],
        },
        {
          kind: "or",
          children: [amount("gt", 100), { kind: "null", field: "amount", isNull: true }],
        },
        {
          kind: "not",
          child: {
            kind: "or",
            children: [
              amount("gt", 100),
              { kind: "compare", field: "trial", op: "eq", value: false },
            ],
          },
        },
      ];

      for (const intent of expressions) {
        const compiled = compile(intent, context);

        if (!compiled.supported) throw new Error(compiled.reason);

        const actual = db
          .prepare(`SELECT id, ${compiled.sql} AS truth FROM records ORDER BY id`)
          .all(...compiled.parameters) as { id: number; truth: number | null }[];

        for (const row of actual) {
          expect(row.truth === null ? null : Boolean(row.truth)).toBe(
            evaluateTruth(intent, rows[row.id]!, context),
          );
        }
      }
    } finally {
      db.close();
    }
  },
);

test("values bind as parameters and caller identifiers are quoted as segments", () => {
  const result = compile(
    { kind: "compare", field: "label", op: "eq", value: "x' OR 1=1 --" },
    context,
  );

  expect(result.supported).toBe(true);
  if (!result.supported) return;
  expect(result.sql).toContain('"label""quoted" COLLATE BINARY = ?1');
  expect(result.sql).not.toContain("x' OR 1=1");
  expect(result.parameters).toEqual(["x' OR 1=1 --"]);
});

test(
  "editing refuses unknown fields, disallowed operators, " + "malformed trees, and empty ranges",
  () => {
    const inputs: FilterIntent[] = [
      { kind: "compare", field: "unknown", op: "eq", value: 1 },
      { kind: "compare", field: "trial", op: "gt", value: false },
      { kind: "compare", field: "amount", op: "eq", value: "100" },
      {
        kind: "range",
        field: "amount",
        lower: { value: 5, inclusive: false },
        upper: { value: 5, inclusive: true },
      },
      { kind: "in", field: "label", values: [] },
      { kind: "and", children: [] },
      { ...amount("gt", 100), guessed: true } as unknown as FilterIntent,
    ];

    for (const intent of inputs) expect(compile(intent, context).supported).toBe(false);

    expect(compile(amount("gt", 100), { ...context, dialect: "postgres" })).toMatchObject({
      supported: false,
      code: "unsupported-dialect",
    });

    expect(
      compile(amount("gt", 100), {
        fields: [{ id: "amount", type: "number", operators: ["eq"] }],
      }),
    ).toMatchObject({ supported: false, code: "unsupported-operator" });

    expect(
      compile(amount("gt", 100), { fields: [{ id: "amount", type: "number" }] }),
    ).toMatchObject({
      supported: false,
      code: "missing-sql-mapping",
    });
  },
);
