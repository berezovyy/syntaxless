import { Database } from "bun:sqlite";
import { expect, test } from "bun:test";

import { comparator, compile, sortRecords } from "../src/compile.js";
import type { Context, SortIntent } from "../src/types.js";

const context: Context = {
  defaultNulls: "last",
  tieBreaker: "id",
  fields: [
    { id: "amount", type: "number", column: 'odd"amount' },
    { id: "name", type: "string", column: "name" },
    { id: "flag", type: "boolean", column: "flag" },
    { id: "id", type: "number", column: "id" },
  ],
};

test(
  "actual SQLite and JavaScript agree on priority, " + "nulls, typed values, Unicode and ties",
  () => {
    const db = new Database(":memory:");

    try {
      db.exec('CREATE TABLE records (id, "odd""amount", name, flag)');

      const rows = [
        { id: 1, amount: 2, name: "😀", flag: true },
        { id: 2, amount: 2, name: "\uE000", flag: false },
        { id: 3, amount: 1, name: "a", flag: null },
        { id: 4, amount: null, name: "b", flag: true },
        { id: 5, amount: 2, name: "😀", flag: true },
        { id: 6, amount: "2", name: 3, flag: 3 },
        { id: 7, amount: 3, name: null, flag: false },
        { id: 8, amount: 0, name: "A", flag: true },
        { id: 9, amount: 2, name: "😀", flag: false },
        { id: 10, amount: 1, name: "\0", flag: true },
      ];

      const insert = db.prepare("INSERT INTO records VALUES (?, ?, ?, ?)");

      for (const r of rows) {
        insert.run(
          r.id,
          r.amount,
          r.name,
          r.flag === null ? null : typeof r.flag === "boolean" ? Number(r.flag) : r.flag,
        );
      }

      for (const [field, direction, nulls] of [
        ["amount", "desc", "first"],
        ["name", "asc", "last"],
        ["flag", "asc", "last"],
      ] as const) {
        const intent: SortIntent = {
          keys: [
            { field, direction, nulls, nullsSource: "explicit" },
            {
              field: field === "name" ? "amount" : "name",
              direction: "desc",
              nulls: "last",
              nullsSource: "explicit",
            },
          ],
          tieBreaker: "id",
        };

        const sql = compile(intent, context);

        expect(sql.supported).toBe(true);
        if (!sql.supported) throw new Error(sql.reason);

        expect(
          db
            .query(`SELECT id FROM records ORDER BY ${sql.sql}`)
            .all()
            .map((r: any) => r.id),
        ).toEqual(sortRecords(rows, intent, context).map((r) => r.id));
      }

      const unicode: SortIntent = {
        keys: [{ field: "name", direction: "asc", nulls: "last", nullsSource: "explicit" }],
        tieBreaker: "id",
      };

      expect(sortRecords(rows, unicode, context).map((r) => r.id)).toEqual([
        8, 3, 4, 2, 1, 5, 9, 6, 7, 10,
      ]);

      expect(rows.map((r) => r.id)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    } finally {
      db.close();
    }
  },
);

test("JavaScript ties remain stable, while reproducible SQL requires a caller tie breaker", () => {
  const intent: SortIntent = {
    keys: [{ field: "amount", direction: "desc", nulls: "last", nullsSource: "explicit" }],
  };

  const rows = [
    { id: 2, amount: 1 },
    { id: 1, amount: 1 },
    { id: 3, amount: 0 },
  ];

  expect(sortRecords(rows, intent, context).map((r) => r.id)).toEqual([2, 1, 3]);

  expect(compile(intent, context)).toMatchObject({
    supported: false,
    code: "missing-tie-breaker",
  });

  expect(comparator(intent, context)(rows[0]!, rows[1]!)).toBe(0);

  expect(() =>
    sortRecords(
      [
        { id: 1, amount: 1 },
        { id: 1, amount: 0 },
      ],
      { ...intent, tieBreaker: "id" },
      context,
    ),
  ).toThrow("unique and non-null");

  expect(() =>
    sortRecords([{ id: null, amount: 1 }], { ...intent, tieBreaker: "id" }, context),
  ).toThrow("unique and non-null");

  expect(
    compile({ ...intent, tieBreaker: "id" }, { ...context, dialect: "postgres" }),
  ).toMatchObject({ supported: false, code: "unsupported-dialect" });

  expect(
    compile(
      {
        ...intent,
        keys: [{ ...intent.keys[0]!, nullsSource: "caller-default", nulls: "first" }],
        tieBreaker: "id",
      },
      context,
    ),
  ).toMatchObject({ supported: false, code: "invalid-intent" });
});
