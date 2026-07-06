import { expect, test } from "bun:test";

import { tokenize } from "../../src/model/tokenizer";
import { spellingVariant } from "./noise";

const clean = {
  template: "test",
  style: "words" as const,
  text: "every monday at nine",
  spans: [
    { start: 0, end: 5, label: "RECUR" as const, clauseStart: false },
    { start: 6, end: 12, label: "WEEKDAY" as const, clauseStart: true },
    { start: 13, end: 15, label: "GLUE" as const, clauseStart: false },
    { start: 16, end: 20, label: "HOUR" as const, clauseStart: false },
  ],
};

test("spelling variants are deterministic and keep spans token aligned", () => {
  let variant: ReturnType<typeof spellingVariant>;

  for (let index = 0; index < 100 && variant === undefined; index++) {
    variant = spellingVariant(clean, `seed-${index}`);
  }

  expect(variant).toBeDefined();
  expect(spellingVariant(clean, "stable")).toEqual(spellingVariant(clean, "stable"));
  if (variant === undefined) return;
  expect(variant.example.text).not.toBe(clean.text);

  for (const token of tokenize(variant.example.text).filter((item) => item.kind !== 3)) {
    expect(
      variant.example.spans.filter((span) => token.start >= span.start && token.end <= span.end),
    ).toHaveLength(1);
  }
});
