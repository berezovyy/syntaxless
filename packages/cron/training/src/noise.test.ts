import { expect, test } from "bun:test";

import type { Label } from "../../src/model/labels";
import { tokenize } from "../../src/model/tokenizer";
import { spellingVariant } from "./noise";
import type { RenderedExample } from "./renderer";

const clean: RenderedExample = {
  template: "noise-fixture",
  style: "words",
  speechAct: "request",
  text: "please run every monday at nine",
  spans: [
    { start: 0, end: 10, label: "O", clauseStart: false, actionable: false },
    { start: 11, end: 16, label: "RECUR", clauseStart: false, actionable: true },
    { start: 17, end: 23, label: "WEEKDAY_MON", clauseStart: true, actionable: true },
    { start: 24, end: 26, label: "GLUE", clauseStart: false, actionable: true },
    { start: 27, end: 31, label: "HOUR_9", clauseStart: false, actionable: true },
  ],
};

test("all spelling operations preserve supervision, offsets, and the untouched input", () => {
  const original = structuredClone(clean);
  const operations = new Set<string>();

  for (let index = 0; index < 100; index++) {
    const seed = `targeted-${index}`;
    const options = { labels: ["WEEKDAY_MON"] as const, force: true };
    const variant = spellingVariant(clean, seed, options);

    if (!variant) throw new Error(`Expected a forced variant for ${seed}`);
    operations.add(variant.noise);
    expect(spellingVariant(clean, seed, options)).toEqual(variant);
    const { example } = variant;
    const changed = example.text.slice(17, -8); // " at nine" stays untouched.

    expect(changed).not.toBe("monday");
    expect(example.text).toBe(`please run every ${changed} at nine`);
    const delta = changed.length - 6;

    expect(delta).toBe(
      {
        deletion: -1,
        insertion: 1,
        transposition: 0,
        repetition: 1,
        "keyboard-neighbor": 0,
      }[variant.noise],
    );

    expect(example).toEqual({
      ...original,
      text: `please run every ${changed} at nine`,
      spans: [
        original.spans[0]!,
        original.spans[1]!,
        {
          start: 17,
          end: 23 + delta,
          label: "WEEKDAY_MON",
          clauseStart: true,
          actionable: true,
        },
        {
          start: 24 + delta,
          end: 26 + delta,
          label: "GLUE",
          clauseStart: false,
          actionable: true,
        },
        {
          start: 27 + delta,
          end: 31 + delta,
          label: "HOUR_9",
          clauseStart: false,
          actionable: true,
        },
      ],
    });

    for (const token of tokenize(example.text).filter((token) => token.kind !== 3)) {
      expect(
        example.spans.filter((span) => token.start >= span.start && token.end <= span.end),
      ).toHaveLength(1);
    }
  }

  expect(operations).toEqual(
    new Set(["deletion", "insertion", "transposition", "repetition", "keyboard-neighbor"]),
  );

  expect(clean).toEqual(original);
});

test("ordinary sampling emits both clean-only and noisy pairs; force bypasses sampling", () => {
  let skipped = 0;
  let emitted = 0;

  for (let index = 0; index < 100; index++) {
    const options = { labels: ["WEEKDAY_MON"] as const };
    const seed = `sample-${index}`;

    if (spellingVariant(clean, seed, options)) emitted++;
    else skipped++;
    expect(spellingVariant(clean, seed, { ...options, force: true })).toBeDefined();
  }

  expect(skipped).toBeGreaterThan(0);
  expect(emitted).toBeGreaterThan(0);
});

test.each(["HOUR_LIT", "MINUTE_LIT", "INTERVAL_LIT", "DOM_LIT"] as const)(
  "default noise excludes %s even when the token is otherwise mutable",
  (label) => {
    // A word isolates label eligibility from the separate numeric-token guard.
    const example: RenderedExample = {
      ...clean,
      text: "eleven",
      spans: [{ start: 0, end: 6, label, clauseStart: true, actionable: true }],
    };

    expect(spellingVariant(example, "literal", { force: true })).toBeUndefined();
    // Explicit opt-in proves this fixture can otherwise produce a mutation.
    expect(spellingVariant(example, "literal", { labels: [label], force: true })).toBeDefined();
  },
);

test.each([
  { text: "12", label: "HOUR_LIT" },
  { text: "at", label: "GLUE" },
  { text: "!", label: "O" },
] satisfies Array<{ text: string; label: Label }>)(
  "non-word or too-short token '$text' is never corrupted",
  ({ text, label }) => {
    const example: RenderedExample = {
      ...clean,
      text,
      spans: [{ start: 0, end: text.length, label, clauseStart: false, actionable: false }],
    };

    expect(spellingVariant(example, "protected", { labels: [label], force: true })).toBeUndefined();
  },
);
