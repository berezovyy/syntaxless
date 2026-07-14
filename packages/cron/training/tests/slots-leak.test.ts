import { expect, test } from "bun:test";

import { normalizeText } from "../src/background";
import {
  allBagTexts,
  humanEvalTexts,
  humanWordingHits,
  intraBagTrigramCollisions,
  leakHits,
} from "../src/slots/leak";

test("bag parts are disjoint from reserved phrases, fossils, and human-v wording", () => {
  const human = humanEvalTexts();

  expect(human.length).toBeGreaterThan(0);
  expect(allBagTexts().length).toBeGreaterThan(0);
  expect(leakHits(human)).toEqual([]);
});

test("no two parts in the same bag share three content words in order", () => {
  expect(intraBagTrigramCollisions()).toEqual([]);
});

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
