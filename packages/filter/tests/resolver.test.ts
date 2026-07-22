import { expect, test } from "bun:test";

import type { Role } from "../src/labels.js";
import { prepare } from "../src/prepare.js";
import { AmbiguousGrouping, resolveRoles } from "../src/resolve.js";
import type { Context, FilterIntent } from "../src/types.js";

const context: Context = {
  fields: [
    {
      id: "status",
      type: "string",
      terms: [{ text: "active customers", value: "active" }],
    },
    { id: "spending", type: "number", aliases: ["spending"] },
    { id: "trial", type: "boolean", terms: [{ text: "trials", value: true }] },
  ],
};

function decode(text: string, roles: Role[]): FilterIntent {
  return resolveRoles(
    prepare(text, context),
    roles.map((role) => ({ role, score: 1 })),
    context,
  ).intent;
}

test("catalog shorthand and exact field facts resolve the reference with exclusion", () => {
  const text = "active customers spending over 100, excluding trials";

  expect(
    decode(text, [
      "FIELD",
      "EQ",
      "VALUE",
      "FIELD",
      "GT",
      "VALUE",
      "O",
      "EXCLUDE",
      "FIELD",
      "EQ",
      "VALUE",
    ]),
  ).toEqual({
    kind: "and",
    children: [
      { kind: "compare", field: "status", op: "eq", value: "active" },
      { kind: "compare", field: "spending", op: "gt", value: 100 },
      {
        kind: "not",
        child: { kind: "compare", field: "trial", op: "eq", value: true },
      },
    ],
  });

  expect(prepare('status equals "Spending over 100"', context).lexemes.at(-1)?.value).toBe(
    "Spending over 100",
  );
});

test("unknown catalog names refuse; mixed ungrouped connectors need clarification", () => {
  expect(() => decode("revenue over 100", ["FIELD", "GT", "VALUE"])).toThrow("unknown field");
  const text = 'spending over 100 or status equals "active" and trial equals false';

  expect(() =>
    decode(text, [
      "FIELD",
      "GT",
      "VALUE",
      "OR",
      "FIELD",
      "EQ",
      "VALUE",
      "AND",
      "FIELD",
      "EQ",
      "VALUE",
    ]),
  ).toThrow(AmbiguousGrouping);

  expect(
    decode(`( spending over 100 or status equals "active" ) and trial equals false`, [
      "LPAREN",
      "FIELD",
      "GT",
      "VALUE",
      "OR",
      "FIELD",
      "EQ",
      "VALUE",
      "RPAREN",
      "AND",
      "FIELD",
      "EQ",
      "VALUE",
    ]),
  ).toMatchObject({
    kind: "and",
    children: [{ kind: "or" }, { kind: "compare", field: "trial" }],
  });
});

test("incomplete predicates and dropped field/literal facts refuse", () => {
  expect(() => decode("spending over", ["FIELD", "GT"])).toThrow("expected VALUE");
  expect(() => decode("spending over 100", ["FIELD", "GT", "O"])).toThrow("did not preserve");

  expect(() => decode("maybe spending over 100", ["UNSUPPORTED", "FIELD", "GT", "VALUE"])).toThrow(
    "non-actionable",
  );
});

test("sentence endings preserve signed decimals, quoted punctuation and questions", () => {
  expect(prepare("spending above -2.75.", context).lexemes.at(-1)?.value).toBe(-2.75);
  expect(prepare('status equals "literal.!".', context).lexemes.at(-1)?.value).toBe("literal.!");
  expect(prepare("spending above 100?", context).lexemes.at(-1)?.text).toBe("?");
});
