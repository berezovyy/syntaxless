import { expect, test } from "bun:test";

import { attributeMiss, type AttributionInput, goldViolatesSemantics } from "./attribution";

test("an unqualified 1:30 labeled as 13:30 is a gold error, not a model error", () => {
  const input: AttributionInput = {
    text: "1:30 daily",
    expectedIntent: {
      frequency: { unit: "day", interval: 1 },
      time: { hour: 13, minute: 30 },
    },
    actualIntent: null,
    accepted: false,
    goldSpans: [
      { start: 0, end: 1, label: "HOUR_LIT", clauseStart: true },
      { start: 1, end: 2, label: "GLUE", clauseStart: false },
      { start: 2, end: 4, label: "MINUTE_30", clauseStart: false },
      { start: 5, end: 10, label: "UNIT_DAY", clauseStart: false },
    ],
    goldLabels: ["HOUR_LIT", "GLUE", "MINUTE_30", "UNIT_DAY"],
    predictedLabels: ["O", "O", "O", "O"],
  };

  expect(goldViolatesSemantics(input)).toBe(true);
  expect(attributeMiss(input)).toBe("GOLD");

  expect(
    goldViolatesSemantics({
      ...input,
      expectedIntent: {
        frequency: { unit: "day", interval: 1 },
        time: { hour: 1, minute: 30 },
      },
    }),
  ).toBe(false);
});
