import { expect, test } from "bun:test";

import { readNumber } from "../src/comparison/number.ts";
import { labelFromId } from "../src/model/labels.ts";
import { tokenize } from "../src/model/tokenizer.ts";
import { featurizeRecord } from "../training/src/featurize.ts";
import { renderField } from "../training/src/renderer.ts";

test("signed decimal literals stay whole and keep exact source offsets", () => {
  const source = "between -2.5 and +10.25, below .5";
  const numbers = tokenize(source).filter((token) => token.kind === 1);

  expect(numbers.map((token) => token.text)).toEqual(["-2.5", "+10.25", ".5"]);

  expect(numbers.map((token) => source.slice(token.start, token.end))).toEqual([
    "-2.5",
    "+10.25",
    ".5",
  ]);

  expect(numbers.map((token) => readNumber([token.text], 0)?.value)).toEqual([-2.5, 10.25, 0.5]);
  expect(tokenize("e-mail forty-two").filter((token) => token.kind === 1)).toEqual([]);
  expect(readNumber(["1.2.3"], 0)).toBeNull();
});

test("the featurizer supervises an entire negative decimal as one NUMBER", () => {
  const intent = {
    type: "number" as const,
    required: true,
    minimum: { value: -2.5, inclusive: false },
  };

  const rendered = renderField(intent, undefined, { numberStyle: "digits" });
  const tokens = tokenize(rendered.text);

  const features = featurizeRecord(
    {
      ...rendered,
      id: "numeric",
      intent,
      split: "train",
      seed: 1,
      family: "constraint",
      fingerprint: "numeric-literal",
    },
    tokens,
  );

  const numeric = tokens.findIndex((token) => token.text === "-2.5");

  expect(numeric).toBeGreaterThanOrEqual(0);
  expect(labelFromId(features.labels[numeric]!)).toBe("NUMBER");
});
