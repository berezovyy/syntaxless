import { expect, test } from "bun:test";

import {
  CONTENT_LABELS,
  isStructural,
  LABEL_COUNT,
  labelFromId,
  labelId,
  LABELS,
  STRUCTURAL_LABELS,
} from "../src/model/labels.ts";

test("label ids are contiguous from 0 and complete", () => {
  LABELS.forEach((label, index) => expect(labelId[label]).toBe(index));
  expect(new Set(Object.values(labelId)).size).toBe(LABEL_COUNT);
  expect(LABEL_COUNT).toBe(LABELS.length);
});

test("structural and content classes partition every label except O", () => {
  const structural = new Set<string>(STRUCTURAL_LABELS);
  const content = new Set<string>(CONTENT_LABELS);

  for (const label of STRUCTURAL_LABELS) expect(content.has(label)).toBe(false);
  for (const label of CONTENT_LABELS) expect(structural.has(label)).toBe(false);

  for (const label of LABELS) {
    const classified = structural.has(label) || content.has(label);

    expect(classified).toBe(label !== "O");
    expect(isStructural(label)).toBe(structural.has(label));
  }
});

test("out-of-range predicted ids are a hard error", () => {
  expect(() => labelFromId(-1)).toThrow(RangeError);
  expect(() => labelFromId(LABEL_COUNT)).toThrow(RangeError);
  expect(() => labelFromId(Number.NaN)).toThrow(RangeError);
  expect(labelFromId(0)).toBe("O");
  expect(labelFromId(LABEL_COUNT - 1)).toBe("EDGE");
});
