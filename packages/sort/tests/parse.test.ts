import { expect, test } from "bun:test";

import { compile, dispose, parse, preload, validate } from "../src/index.js";
import type { Context, SortIntent } from "../src/index.js";

const context: Context = {
  defaultNulls: "last",
  tieBreaker: "id",
  fields: [
    {
      id: "created",
      type: "number",
      column: "created",
      orderingAliases: [{ text: "newest", direction: "desc" }],
    },
    {
      id: "price",
      type: "number",
      column: "price",
      aliases: ["prices"],
      orderingAliases: [{ text: "cheapest", direction: "asc" }],
    },
    { id: "id", type: "number", column: "id" },
  ],
};

test("the bundled model parses the reference request and refuses uncertain wording", async () => {
  expect(await parse("newest first, then cheapest, missing prices last", context)).toMatchObject({
    ok: true,
    source: "model",
    intent: {
      keys: [
        {
          field: "created",
          direction: "desc",
          nulls: "last",
          nullsSource: "caller-default",
        },
        { field: "price", direction: "asc", nulls: "last", nullsSource: "explicit" },
      ],
      tieBreaker: "id",
    },
  });

  for (const text of ["newest first?", "price ascending descending"]) {
    expect((await parse(text, context)).ok).toBe(false);
  }

  const { defaultNulls: _, ...withoutDefault } = context;

  expect((await parse("price descending", withoutDefault)).ok).toBe(false);
});

test("bad input text and catalogs use stable codes", async () => {
  expect(await parse("", context)).toMatchObject({ ok: false, code: "invalid-input" });

  expect(await parse("price descending", { fields: [] })).toMatchObject({
    ok: false,
    code: "invalid-catalog",
  });
});

test("custom weights with an unsupported format version or invalid bytes refuse", async () => {
  for (const bytes of [JSON.stringify({ version: 1 }), "not json"]) {
    expect(
      await parse("price descending", {
        ...context,
        weights: new TextEncoder().encode(bytes),
      }),
    ).toMatchObject({ ok: false, code: "incompatible-model" });
  }
});

test("preload and dispose keep parsing available", async () => {
  await preload();
  dispose();
  expect((await parse("price descending", context)).ok).toBe(true);
});

test("SQL guards each key, quotes identifiers and appends the tie breaker", () => {
  const intent: SortIntent = {
    keys: [{ field: "created", direction: "desc", nulls: "last", nullsSource: "explicit" }],
    tieBreaker: "id",
  };

  expect([...validate(intent, context).keys()]).toEqual(["created", "price", "id"]);

  const guard = (column: string) =>
    `(CASE WHEN typeof("${column}") IN ('integer', 'real') ` +
    `AND abs("${column}") <= 9007199254740991 THEN "${column}" ELSE NULL END)`;

  expect(compile(intent, context)).toEqual({
    supported: true,
    dialect: "sqlite",
    sql: `${guard("created")} DESC NULLS LAST, ${guard("id")} ASC NULLS LAST`,
  });
});
