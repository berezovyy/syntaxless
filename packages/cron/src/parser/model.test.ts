import { describe, expect, test } from "bun:test";

import { labelId } from "../model/labels";
import { tokenize } from "../model/tokenizer";
import type { LabeledToken } from "./decode";
import { resultFromLabeledTokens } from "./model";
import { normalize } from "./normalize";

describe("heads-only correction span", () => {
  test("the string reaching tokenize equals normalize(source)", () => {
    const source = "Sorry, I meant Tuesday at 10 am rather than Monday at nine.";

    expect(tokenize(normalize(source)).length).toBeGreaterThan(0);
    expect(normalize(source)).toBe(normalize(source));
  });

  test("correction + gold actionable span yields the replacement weekday", () => {
    const tokens = tokenize("thursday friday").filter((token) => token.kind !== 3);

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role: labelId.WEEKDAY,
      boundary: token.text === "thursday",
      score: 1,
      actionable: token.text === "friday",
    }));

    const result = resultFromLabeledTokens(
      labeled,
      undefined,
      "thursday friday",
      "off",
      "correction",
    );

    expect(result.ok).toBe(true);

    if (result.ok) {
      expect(result.intent).toEqual({
        frequency: { unit: "week", interval: 1 },
        weekdays: ["friday"],
      });
    }
  });

  test("correction + empty actionable rejects", () => {
    const tokens = tokenize("thursday friday").filter((token) => token.kind !== 3);

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role: labelId.WEEKDAY,
      boundary: token.text === "thursday",
      score: 1,
      actionable: false,
    }));

    const result = resultFromLabeledTokens(
      labeled,
      undefined,
      "thursday friday",
      "off",
      "correction",
    );

    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.code).toBe("unsafe-context");
  });

  test.each([
    "Cancel the weekday idea.",
    "Actually, never mind.",
    "Monday or Tuesday might work; I have not decided.",
  ])("cancellation or unresolved choice is not rewritten into a request: %s", (source) => {
    const tokens = tokenize(normalize(source)).filter((token) => token.kind !== 3);

    const labeled: LabeledToken[] = tokens.map((token) => ({
      token,
      role: labelId.O,
      boundary: false,
      score: 1,
      actionable: false,
    }));

    const result = resultFromLabeledTokens(labeled, undefined, source, "off", "cancel");

    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.code).toBe("unsafe-context");
  });
});
