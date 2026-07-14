import { expect, test } from "bun:test";

import { labelFromId, LABELS, unitLabel, weekdayLabel } from "../../src/model/labels";
import type { Label } from "../../src/model/labels";
import {
  ADVERB_FREQUENCIES,
  LEXICON,
  LOOKUP,
  NUMBER_WORDS,
  UNIT_ALIASES,
  WEEKDAY_ALIASES,
  WORDS,
} from "../../src/parser/lexicon";
import { renderIntent, renderNegative } from "../src/renderer";
import type { RenderedExample } from "../src/renderer";
import { createRng } from "../src/rng";
import { FAMILY_WEIGHTS, sampleIntent } from "../src/sampler";
import type { Family } from "../src/sampler";

const LABEL_SET = new Set<Label>(LABELS);

test("lookup is complete and consistent with the source aliases", () => {
  const keys = Object.keys(LEXICON);

  expect(LOOKUP.size).toBe(keys.length);

  for (const [word, weekday] of Object.entries(WEEKDAY_ALIASES)) {
    expect(LOOKUP.get(word)).toEqual({ label: weekdayLabel(weekday), value: weekday });
  }

  for (const [word, unit] of Object.entries(UNIT_ALIASES)) {
    expect(LOOKUP.get(word)).toEqual({ label: unitLabel(unit), value: unit });
  }

  for (const [word, unit] of Object.entries(ADVERB_FREQUENCIES)) {
    expect(LOOKUP.get(word)).toEqual({ label: unitLabel(unit), value: unit });
  }

  for (const [word, value] of Object.entries(NUMBER_WORDS)) {
    expect(LOOKUP.get(word)).toEqual({ label: "COUNT", value });
  }
});

test("lexicon entries pin the canonical plan examples", () => {
  expect(LOOKUP.get("mon")).toEqual({ label: "WEEKDAY_MON", value: "monday" });

  expect(LOOKUP.get("mondays")).toEqual({
    label: "WEEKDAY_MON",
    value: "monday",
  });

  expect(LOOKUP.get("every")).toEqual({ label: "RECUR" });
  expect(LOOKUP.get("hourly")).toEqual({ label: "UNIT_HOUR", value: "hour" });
  expect(LOOKUP.get("first")).toEqual({ label: "ORDINAL_1", value: 1 });
  expect(LOOKUP.get("the stride is")).toEqual({ label: "GLUE" });
  expect(LOOKUP.get("the weekdays are")).toEqual({ label: "GLUE" });
  expect(LOOKUP.get("the range runs")).toEqual({ label: "GLUE" });
  expect(LOOKUP.get("the omitted days are")).toEqual({ label: "EXCEPT" });
  expect(LOOKUP.get("ending")).toEqual({ label: "BOUND_END_EXCL" });
});

test("every surface maps to a valid label and never conflicts", () => {
  for (const [word, entry] of Object.entries(LEXICON)) {
    expect(LABEL_SET.has(entry.label)).toBe(true);
    expect(labelFromId(LABELS.indexOf(entry.label))).toBe(entry.label);
    expect(word).toBe(word.toLowerCase());
    expect(LOOKUP.get(word)).toBe(entry);
  }
});

test("the renderer consumes the runtime lexicon", () => {
  const entries = Object.entries(FAMILY_WEIGHTS) as Array<[Family, number]>;
  const rng = createRng(20260913);
  const meridiemPhrases = new Set<string>(Object.values(WORDS.qualifier));

  for (let index = 0; index < 500; index++) {
    const family = rng.weighted(entries);
    const sample = sampleIntent(rng, family);

    const example: RenderedExample =
      sample.intent === null ? renderNegative(rng) : renderIntent(sample.intent, family, rng);

    for (const span of example.spans) {
      if (
        !span.label.startsWith("WEEKDAY_") &&
        !span.label.startsWith("ORDINAL_") &&
        !span.label.startsWith("INTERVAL_") &&
        !span.label.startsWith("HOUR_") &&
        !span.label.startsWith("MINUTE_") &&
        !span.label.startsWith("MERIDIEM_")
      ) {
        continue;
      }

      const text = example.text.slice(span.start, span.end);
      const lower = text.toLowerCase();

      const known =
        LOOKUP.has(lower) ||
        /^\d+$/.test(text) ||
        meridiemPhrases.has(lower) ||
        lower.split(/\s+/).every((word) => LOOKUP.has(word));

      expect(known).toBe(true);
    }
  }
});
