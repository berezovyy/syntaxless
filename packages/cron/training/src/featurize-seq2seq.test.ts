import { expect, test } from "bun:test";

import { linearizeIntent } from "./featurize-seq2seq";

test("linearizeIntent emits copy/closed-class tokens, not free-form strings", () => {
  expect(linearizeIntent(null)).toEqual(["(", "reject", ")"]);

  expect(
    linearizeIntent({
      frequency: { unit: "week", interval: 1 },
      weekdays: ["monday"],
      time: { hour: 9, minute: 0 },
    }),
  ).toEqual(["(", "every", "1", "week", "at", "9", "0", "days", "monday", ")"]);
});
