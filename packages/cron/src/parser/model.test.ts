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
    [
      "Actually, forget the hourly check and poll every 26 minutes instead.",
      "poll every 26 minutes",
    ],
    [
      "Not Wednesday at eight—make the meeting reminder Friday at 10:30 a.m.",
      "make the meeting reminder Friday at 10:30 a.m.",
    ],
    ["I said the 12th, but I meant the 21st of every month.", "the 21st of every month."],
    [
      "Scratch the daily plan; schedule it every Monday at 7 p.m.",
      "schedule it every Monday at 7 p.m.",
    ],
    [
      "Change that from weekends to weekdays, still at 11:15.",
      "schedule weekdays, still at 11:15.",
    ],
    ["No, I need it every two hours, not every two minutes.", "every two hours"],
    [
      "Make that 8:05 tonight and every evening thereafter, instead of 8:50.",
      "make 8:05 tonight and every evening thereafter",
    ],
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
