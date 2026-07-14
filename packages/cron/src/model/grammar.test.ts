import { expect, test } from "bun:test";

import { GRAMMAR_TOKENS, grammarMask, greedyStep, intentFromGrammar } from "./grammar";
import { greedyDecode } from "./inference-seq2seq";

function oneHot(token: (typeof GRAMMAR_TOKENS)[number]): number[] {
  return GRAMMAR_TOKENS.map((item) => (item === token ? 5 : -5));
}

test("grammar mask plus one greedy step on a fixture prefix", () => {
  const start = grammarMask([]);

  expect(start[GRAMMAR_TOKENS.indexOf("(")]).toBe(true);
  expect(start[GRAMMAR_TOKENS.indexOf("every")]).toBe(false);
  const logits = GRAMMAR_TOKENS.map((token) => (token === "(" ? 3 : 0));

  expect(greedyStep([], logits)).toBe("(");

  const afterOpen = GRAMMAR_TOKENS.map((token) =>
    token === "every" ? 5 : token === "reject" ? 1 : 0,
  );

  expect(greedyStep(["("], afterOpen)).toBe("every");
});

test("grammar-constrained greedy decode of a daily fixture yields UNIT_DAY intent", () => {
  const tokens = greedyDecode((["(", "every", "COPY", "UNIT_DAY", ")"] as const).map(oneHot));

  expect(tokens).toEqual(["(", "every", "COPY", "UNIT_DAY", ")"]);

  expect(intentFromGrammar(tokens)).toEqual({
    frequency: { unit: "day", interval: 1 },
  });

  expect(intentFromGrammar(["(", "reject", ")"])).toBeNull();
});
