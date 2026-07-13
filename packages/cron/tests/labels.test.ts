import { expect, test } from "bun:test";

import {
  CONTENT_LABELS,
  isLiteralLabel,
  isStructural,
  LABEL_COUNT,
  labelFromId,
  labelId,
  LABELS,
  STRUCTURAL_LABELS,
  valueOf,
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

test("closed classes expose valueOf and literals do not", () => {
  expect(valueOf("WEEKDAY_MON")).toBe("monday");
  expect(valueOf("WEEKDAY_SUN")).toBe("sunday");
  expect(valueOf("DAYGROUP_WEEKEND")).toBe("weekend");
  expect(valueOf("HOUR_9")).toBe(9);
  expect(valueOf("HOUR_0")).toBe(0);
  expect(valueOf("HOUR_23")).toBe(23);
  expect(valueOf("MINUTE_15")).toBe(15);
  expect(valueOf("MERIDIEM_PM")).toBe("pm");
  expect(valueOf("ORDINAL_1")).toBe("first");
  expect(valueOf("ORDINAL_LAST")).toBe("last");
  expect(valueOf("BOUND_END_INCL")).toBe("end-incl");
  expect(valueOf("UNIT_MIN")).toBe("minute");
  expect(valueOf("INTERVAL_2")).toBe(2);
  expect(valueOf("INTERVAL_12")).toBe(12);
  expect(valueOf("HOUR_LIT")).toBeUndefined();
  expect(valueOf("MINUTE_LIT")).toBeUndefined();
  expect(valueOf("INTERVAL_LIT")).toBeUndefined();
  expect(valueOf("DOM_LIT")).toBeUndefined();
  expect(isLiteralLabel("HOUR_LIT")).toBe(true);
  expect(isLiteralLabel("WEEKDAY_MON")).toBe(false);
});

test("out-of-range predicted ids are a hard error", () => {
  expect(() => labelFromId(-1)).toThrow(RangeError);
  expect(() => labelFromId(LABEL_COUNT)).toThrow(RangeError);
  expect(() => labelFromId(Number.NaN)).toThrow(RangeError);
  expect(labelFromId(0)).toBe("O");
  expect(labelFromId(LABEL_COUNT - 1)).toBe("DOM_LIT");
});
