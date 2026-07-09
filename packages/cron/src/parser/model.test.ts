import { describe, expect, test } from "bun:test";

import { scopeActionableCorrection } from "./model";

describe("actionable correction scoping", () => {
  test.each([
    ["Actually, make that every 20 minutes, not every hour.", "every 20 minutes"],
    ["Sorry, I meant Tuesday at 10 am rather than Monday at nine.", "Tuesday at 10 am"],
    [
      "I said Friday by mistake. Please make it Thursday every week.",
      "Please make it Thursday every week.",
    ],
    ["Not Monday and Wednesday after all; schedule Friday at four.", "schedule Friday at four."],
    ["Move it from Tuesday at eight to Thursday at eleven.", "schedule Thursday at eleven."],
    ["Instead of noon, make the daily reminder 2 pm.", "make the daily reminder 2 pm."],
  ])("keeps only the selected replacement in %s", (source, expected) => {
    expect(scopeActionableCorrection(source)).toBe(expected);
  });

  test.each([
    "Cancel the weekday idea.",
    "Actually, never mind.",
    "Monday or Tuesday might work; I have not decided.",
  ])("does not turn a cancellation or unresolved choice into a request", (source) => {
    expect(scopeActionableCorrection(source)).toBe(source);
  });
});
