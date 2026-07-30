import { expect, test } from "bun:test";

import { dispose, explain, parse, preload, validate } from "../src/index.js";
import type { Context, FilterIntent } from "../src/index.js";
import { prepare } from "../src/prepare.js";
import { resolveRoles } from "../src/resolve.js";
import { InputError } from "../src/types.js";

const context: Context = {
  fields: [
    {
      id: "status",
      type: "string",
      column: "status",
      terms: [{ text: "active customers", value: "active" }],
    },
    { id: "spending", type: "number", column: "spending" },
    {
      id: "trial",
      type: "boolean",
      column: "trial",
      terms: [{ text: "trials", value: true }],
    },
  ],
};

const reference: FilterIntent = {
  kind: "and",
  children: [
    { kind: "compare", field: "status", op: "eq", value: "active" },
    { kind: "compare", field: "spending", op: "gt", value: 100 },
    { kind: "not", child: { kind: "compare", field: "trial", op: "eq", value: true } },
  ],
};

test("the bundled model parses the reference request and refuses uncertain wording", async () => {
  const result = await parse("active customers spending over 100, excluding trials", context);

  expect(result).toMatchObject({ ok: true, intent: reference, source: "model" });

  for (const text of ["spending over 100?", "maybe spending over 100"]) {
    expect((await parse(text, context)).ok).toBe(false);
  }
});

test("bad input text, unknown catalogs and mixed connectives use stable codes", async () => {
  expect(await parse("", context)).toMatchObject({ ok: false, code: "invalid-input" });

  expect(await parse('status equals "open', context)).toMatchObject({
    ok: false,
    code: "invalid-input",
  });

  expect(await parse("spending over 100", { fields: [] })).toMatchObject({
    ok: false,
    code: "invalid-catalog",
  });

  const text = 'spending over 100 or status equals "active" and trial equals false';
  let error: unknown;

  try {
    resolveRoles(
      prepare(text, context),
      ["FIELD", "GT", "VALUE", "OR", "FIELD", "EQ", "VALUE", "AND", "FIELD", "EQ", "VALUE"].map(
        (role) => ({ role: role as never, score: 1 }),
      ),
      context,
    );
  } catch (caught) {
    error = caught;
  }

  expect(error).toBeInstanceOf(InputError);
  expect((error as InputError).code).toBe("ambiguous");
});

test("caller aliases and terms match after trimming surrounding whitespace", () => {
  const padded: Context = {
    fields: [
      { id: "spending", type: "number", aliases: ["  total spend "] },
      { id: "trial", type: "boolean", terms: [{ text: " trials ", value: true }] },
    ],
  };

  const lexemes = prepare("total spend over 5 excluding trials", padded).lexemes;

  expect(lexemes[0]).toMatchObject({ text: "numberfield", field: "spending" });
  expect(lexemes.at(-3)).toMatchObject({ text: "booleanfield", field: "trial" });
  expect(lexemes.at(-1)).toMatchObject({ value: true });
});

test("custom weights with an unsupported format version or invalid bytes refuse", async () => {
  const wrongVersion = new TextEncoder().encode(JSON.stringify({ version: 1 }));

  expect(await parse("spending over 100", { ...context, weights: wrongVersion })).toMatchObject({
    ok: false,
    code: "incompatible-model",
  });

  const invalid = new TextEncoder().encode("not json");

  expect(await parse("spending over 100", { ...context, weights: invalid })).toMatchObject({
    ok: false,
    code: "incompatible-model",
  });
});

test("preload and dispose keep parsing available", async () => {
  await preload();
  dispose();
  expect((await parse("spending over 100", context)).ok).toBe(true);
});

test("explanations name every predicate, exclusion and range bound", () => {
  expect(explain(reference)).toBe(
    '("status" equals "active" AND "spending" is greater than 100 AND NOT ("trial" equals true))',
  );

  expect(
    explain({
      kind: "range",
      field: "spending",
      lower: { value: 1, inclusive: true },
      upper: { value: 5, inclusive: false },
    }),
  ).toBe('"spending" at least 1 and less than 5');

  expect(explain({ kind: "null", field: "spending", isNull: true })).toBe(
    '"spending" is missing or null',
  );
});

test("validate returns the catalog and rejects edited intents with typed codes", () => {
  expect([...validate(reference, context).keys()]).toEqual(["status", "spending", "trial"]);
  let error: unknown;

  try {
    validate({ kind: "compare", field: "trial", op: "gt", value: true }, context);
  } catch (caught) {
    error = caught;
  }

  expect((error as InputError).code).toBe("unsupported-operator");
});
