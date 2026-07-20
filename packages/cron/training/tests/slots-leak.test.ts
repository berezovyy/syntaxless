import { expect, test } from "bun:test";

import { normalizeText } from "../src/background";
import { allBagTexts, humanWordingHits, leakHits } from "../src/slots/leak";

test("leak detection reports a planted match, rather than always returning an empty list", () => {
  const row = allBagTexts()[0]!;
  const normalized = normalizeText(row.text);

  expect(leakHits([normalized])).toContainEqual({
    ...row,
    kind: "human",
    detail: normalized,
  });

  const human = "please open the wooden cupboard";

  expect(humanWordingHits("Please open the wooden cupboard, then close it.", [human])).toEqual([
    human,
  ]);

  expect(humanWordingHits("Please close the window.", [human])).toEqual([]);
});
