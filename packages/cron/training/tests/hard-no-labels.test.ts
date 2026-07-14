import { expect, test } from "bun:test";

import { tokenize } from "../../src/model/tokenizer";
import { iterateKinds } from "../src/kinds";
import { drawMix } from "../src/mix/draw";
import { timeShapedNumber } from "../src/mix/filter";

const SCHEDULE_MENTION =
  /\b(every|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekday|weekend|hourly|daily|weekly|monthly|noon|midnight|o'clock|reminder|schedule|cron)\b/i;

test("no remaining negative-kind row paints a schedule mention as all-O", () => {
  for (const example of iterateKinds(new Set(["negative"]))) {
    const hasHole = example.example.spans.some((span) => span.label !== "O");

    if (hasHole) continue;

    expect(
      SCHEDULE_MENTION.test(example.example.text) || timeShapedNumber(example.example.text),
    ).toBe(false);
  }
});

test("mixed H rows keep a labeled hole and a non-request speech-act", () => {
  const { examples } = drawMix({ count: 400, seed: 20260913, split: "train" });
  // Select by emitted population, not the same classifier whose invariants
  // we are checking. Otherwise malformed H rows could disappear from the test.
  const hard = examples.filter((row) => row.example.style === "mix-H");

  expect(hard.length).toBe(88); // 22% of this 400-row draw.

  for (const example of hard) {
    expect(example.example.spans.some((span) => span.label !== "O")).toBe(true);
    expect(example.example.speechAct).not.toBe("request");
    expect(example.example.speechAct).not.toBe("correction");

    expect(
      example.example.spans
        .filter((span) => span.label !== "O")
        .every((span) => span.actionable === false),
    ).toBe(true);

    const tokens = tokenize(example.example.text).filter((token) => token.kind !== 3);

    for (const token of tokens) {
      const covering = example.example.spans.filter(
        (span) => token.start >= span.start && token.end <= span.end,
      );

      expect(covering.length).toBe(1);
    }
  }
});
