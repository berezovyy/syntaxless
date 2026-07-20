import { expect, test } from "bun:test";

import { normalizeText } from "../src/background";
import { easyNoDropReason } from "../src/mix/filter";

test("easy-negative filtering distinguishes rejected wording and duplicate kept rows", () => {
  const blocked = "Please open the wooden door.";

  const options = {
    blocked: new Set([normalizeText(blocked)]),
    seen: new Set<string>(),
  };

  const reasons = [
    "Please close the window.",
    "Please close the window.",
    "lowercase text is not a sentence.",
    "Too short.",
    "The clock displays 10:30 clearly.",
    "Please close the window every day.",
    blocked,
  ].map((text) => easyNoDropReason(text, options));

  expect(reasons).toEqual([
    null,
    "duplicate",
    "shape",
    "length",
    "timeShaped",
    "timeWord",
    "blocked",
  ]);
});

test("easy negatives are checked after runtime normalization, including dotted meridiems and Unicode ordinals", () => {
  expect(easyNoDropReason("The valve closes at six p.m.")).toBe("timeWord");
  expect(easyNoDropReason("The valve closes at 9 p. m.")).toBe("timeShaped");
  expect(easyNoDropReason("This is the 89ᵗʰ entry in the catalogue.")).toBe("timeShaped");
  expect(easyNoDropReason("Please put these boxes aside . . .")).toBe("normalization");
  expect(easyNoDropReason("Please put these boxes aside...")).toBeNull();
});
