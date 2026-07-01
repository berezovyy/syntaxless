import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { labelFromId, LABELS } from "../../src/model/labels";
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

const RENDERER_PATH = join(import.meta.dir, "..", "src", "renderer.ts");
const BACKGROUND_PATH = join(import.meta.dir, "..", "src", "background.ts");

function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
}

test("lookup is complete and consistent with the source aliases", () => {
  const keys = Object.keys(LEXICON);

  expect(new Set(keys).size).toBe(keys.length);
  expect(LOOKUP.size).toBe(keys.length);

  for (const [word, weekday] of Object.entries(WEEKDAY_ALIASES)) {
    expect(LOOKUP.get(word)).toEqual({ label: "WEEKDAY", value: weekday });
  }

  for (const [word, unit] of Object.entries(UNIT_ALIASES)) {
    expect(LOOKUP.get(word)).toEqual({ label: "INTERVAL", value: unit });
  }

  for (const [word, unit] of Object.entries(ADVERB_FREQUENCIES)) {
    expect(LOOKUP.get(word)).toEqual({ label: "INTERVAL", value: unit });
  }

  for (const [word, value] of Object.entries(NUMBER_WORDS)) {
    expect(LOOKUP.get(word)).toEqual({ label: "COUNT", value });
  }
});

test("lexicon entries pin the canonical plan examples", () => {
  expect(LOOKUP.get("mon")).toEqual({ label: "WEEKDAY", value: "monday" });

  expect(LOOKUP.get("mondays")).toEqual({
    label: "WEEKDAY",
    value: "monday",
  });

  expect(LOOKUP.get("every")).toEqual({ label: "RECUR" });
  expect(LOOKUP.get("hourly")).toEqual({ label: "INTERVAL", value: "hour" });
  expect(LOOKUP.get("first")).toEqual({ label: "ORDINAL", value: 1 });
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
        span.label !== "WEEKDAY" &&
        span.label !== "ORDINAL" &&
        span.label !== "INTERVAL" &&
        span.label !== "HOUR" &&
        span.label !== "MINUTE" &&
        span.label !== "MERIDIEM"
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

test("no schedule literal is reintroduced outside the runtime lexicon", () => {
  const lexiconKeys = new Set(Object.keys(LEXICON));

  const contentKeys = new Set(
    Object.entries(LEXICON)
      .filter(([, entry]) => ["WEEKDAY", "INTERVAL", "ORDINAL", "MERIDIEM"].includes(entry.label))
      .map(([word]) => word),
  );

  for (const path of [RENDERER_PATH, BACKGROUND_PATH]) {
    const source = stripComments(readFileSync(path, "utf8"));

    expect(source).toContain("parser/lexicon");

    // No add() call may pass a raw schedule word.
    const addCall = /\.add\(\s*(["'])([^"']*)\1/g;

    for (const match of source.matchAll(addCall)) {
      expect(lexiconKeys.has(match[2]!.toLowerCase())).toBe(false);
    }

    // Content vocabulary (weekdays, units, ordinals, meridiems) must not even
    // appear as a string literal: callers draw it all from the lexicon.
    const literal = /(["'])([^"']*)\1/g;

    for (const match of source.matchAll(literal)) {
      // Capitalised label constants such as "HOUR" are not surface vocabulary.
      if ((LABEL_SET as Set<string>).has(match[2]!)) continue;
      expect(contentKeys.has(match[2]!.toLowerCase())).toBe(false);
    }
  }
});
