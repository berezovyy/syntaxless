import { expect, test } from "bun:test";

import { explain } from "../src/explain.js";
import type { Role } from "../src/labels.js";
import { prepare } from "../src/prepare.js";
import { resolveRoles } from "../src/resolve.js";
import type { Context } from "../src/types.js";

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

function resolve(text: string, roles: Role[], ctx = context) {
  const prepared = prepare(text, ctx);

  return resolveRoles(
    prepared,
    roles.map((role) => ({ role, score: 1 })),
    ctx,
  ).intent;
}

test("reference keeps caller aliases, priority and explicit null update provenance", () => {
  const intent = resolve("newest first, then cheapest, missing prices last.", [
    "FIELD",
    "DESC",
    "GLUE",
    "NEXT",
    "NEXT",
    "FIELD",
    "ASC",
    "NEXT",
    "NULLS",
    "FIELD",
    "LAST",
  ]);

  expect(intent).toEqual({
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
  });

  expect(explain(intent)).toContain("caller default");
  expect(explain(intent)).toContain('"id", ascending');

  expect(
    resolve("price descending nulls first", ["FIELD", "DESC", "NULLS", "FIRST"], {
      fields: context.fields,
      tieBreaker: context.tieBreaker!,
    }),
  ).toMatchObject({
    keys: [{ field: "price", direction: "desc", nulls: "first", nullsSource: "explicit" }],
  });

  expect(prepare("newest first!", context).text).toBe("numberfield descending first");
  expect(prepare("newest first?", context).lexemes.at(-1)?.text).toBe("?");
});

test(
  "contradictions, dropped caller facts, " + "absent defaults and ambiguous null updates refuse",
  () => {
    expect(() => resolve("price ascending descending", ["FIELD", "ASC", "DESC"])).toThrow(
      "contradictory directions",
    );

    expect(() => resolve("newest", ["FIELD", "O"])).toThrow("preserve");
    expect(() => resolve("price descending", ["O", "DESC"])).toThrow("preserve");
    expect(() => resolve("price", ["FIELD"])).toThrow("explicit direction");

    expect(() =>
      resolve("price descending", ["FIELD", "DESC"], {
        fields: context.fields,
        tieBreaker: context.tieBreaker!,
      }),
    ).toThrow("specify null placement");

    expect(() =>
      resolve("newest then cheapest then nulls first", [
        "FIELD",
        "DESC",
        "NEXT",
        "FIELD",
        "ASC",
        "NEXT",
        "NULLS",
        "FIRST",
      ]),
    ).toThrow("identify");

    expect(() =>
      resolve("price descending then missing price first then missing price last", [
        "FIELD",
        "DESC",
        "NEXT",
        "NULLS",
        "FIELD",
        "FIRST",
        "NEXT",
        "NULLS",
        "FIELD",
        "LAST",
      ]),
    ).toThrow("contradictory null");

    expect(() => resolve("unknown ascending", ["FIELD", "ASC"])).toThrow("unknown field");

    expect(() =>
      prepare("newest", {
        ...context,
        fields: [...context.fields, { id: "other", type: "number", aliases: ["newest"] }],
      }),
    ).toThrow("ambiguous catalog");
  },
);
