import { expect, test } from "bun:test";

import { tokenize } from "../../src/model/tokenizer";
import { NOISE_SAFE_LABELS, spellingVariant } from "./noise";

const clean = {
  template: "test",
  style: "words" as const,
  speechAct: "request" as const,
  text: "please run every monday at nine",
  spans: [
    { start: 0, end: 6, label: "O" as const, clauseStart: false, actionable: false },
    { start: 7, end: 10, label: "O" as const, clauseStart: false, actionable: false },
    { start: 11, end: 16, label: "RECUR" as const, clauseStart: false, actionable: true },
    { start: 17, end: 23, label: "WEEKDAY_MON" as const, clauseStart: true, actionable: true },
    { start: 24, end: 26, label: "GLUE" as const, clauseStart: false, actionable: true },
    { start: 27, end: 31, label: "HOUR_9" as const, clauseStart: false, actionable: true },
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

test("a forced role-targeted variant mutates only the requested evidence", () => {
  const variant = spellingVariant(clean, "targeted", {
    labels: ["WEEKDAY_MON"],
    force: true,
  });

  expect(variant).toBeDefined();
  if (variant === undefined) return;
  expect(variant.example.text).toStartWith("please run every ");
  expect(variant.example.text).not.toContain("monday");
  expect(variant.example.text).toEndWith(" at nine");
});

test("default noise allowlist is every non-literal label", () => {
  expect(NOISE_SAFE_LABELS).toContain("O");
  expect(NOISE_SAFE_LABELS).toContain("WEEKDAY_MON");
  expect(NOISE_SAFE_LABELS).toContain("HOUR_9");
  expect(NOISE_SAFE_LABELS).not.toContain("HOUR_LIT");
  expect(NOISE_SAFE_LABELS).not.toContain("DOM_LIT");
  expect(NOISE_SAFE_LABELS).not.toContain("INTERVAL_LIT");
  expect(NOISE_SAFE_LABELS).not.toContain("MINUTE_LIT");
  let variant: ReturnType<typeof spellingVariant>;

  for (let index = 0; index < 32 && variant === undefined; index += 1) {
    variant = spellingVariant(clean, `safe-${index}`, { force: true });
  }

  expect(variant).toBeDefined();
  if (variant === undefined) return;
  expect(variant.example.text).not.toBe(clean.text);
});
