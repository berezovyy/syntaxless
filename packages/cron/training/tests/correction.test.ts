import { expect, test } from "bun:test";

import { tokenize } from "../../src/model/tokenizer";
import { assignSplits, frameKey } from "../src/backward";
import { CORRECTION_FRAMES, wrapCorrection } from "../src/correction";
import { expandKinds } from "../src/kinds";
import { Sentence } from "../src/sentence";

function weekday(name: "thursday" | "friday"): Sentence {
  const sentence = new Sentence();

  sentence.clause();

  sentence.add(
    name === "thursday" ? "Thursday" : "Friday",
    name === "thursday" ? "WEEKDAY_THU" : "WEEKDAY_FRI",
  );

  return sentence;
}

function aligned(example: { text: string; spans: Sentence["spans"] }): void {
  for (const token of tokenize(example.text).filter((item) => item.kind !== 3)) {
    const covering = example.spans.filter(
      (span) => token.start >= span.start && token.end <= span.end,
    );

    expect(covering).toHaveLength(1);
  }
}

test("gen:kinds emits correction rows", () => {
  const rows = expandKinds(new Set(["correction"]));

  expect(rows.length).toBeGreaterThan(0);
  expect(rows.every((row) => row.kind === "correction")).toBe(true);
  expect(rows.every((row) => row.example.speechAct === "correction")).toBe(true);
  expect(rows.every((row) => row.intent !== null)).toBe(true);
});

test("Thursday vs Friday wrap golds Friday and only Friday is actionable", () => {
  const row = expandKinds(new Set(["correction"])).find(
    (example) =>
      example.template === "correction/i-meant" && example.intent?.weekdays?.[0] === "friday",
  );

  expect(row).toBeDefined();
  const text = row!.example.text;

  expect(text).toContain("friday");
  expect(text).toContain("thursday");
  const weekdays = row!.example.spans.filter((span) => span.label.startsWith("WEEKDAY_"));

  expect(weekdays).toHaveLength(2);

  const friday = weekdays.find(
    (span) => text.slice(span.start, span.end).toLowerCase() === "friday",
  );

  const thursday = weekdays.find(
    (span) => text.slice(span.start, span.end).toLowerCase() === "thursday",
  );

  expect(friday?.actionable).toBe(true);
  expect(thursday?.actionable).toBe(false);

  expect(row!.intent).toEqual({
    frequency: { unit: "week", interval: 1 },
    weekdays: ["friday"],
  });

  expect(
    row!.example.spans
      .filter((span) => span.label === "O")
      .every((span) => span.actionable === false),
  ).toBe(true);

  aligned(row!.example);
});

test("reverse-order wrap still golds B", () => {
  const row = expandKinds(new Set(["correction"])).find(
    (example) =>
      example.template === "correction/not-use" && example.intent?.weekdays?.[0] === "friday",
  );

  expect(row).toBeDefined();
  const text = row!.example.text;

  expect(text.indexOf("thursday")).toBeLessThan(text.indexOf("friday"));
  expect(row!.intent?.weekdays).toEqual(["friday"]);

  const friday = row!.example.spans.find(
    (span) =>
      span.label.startsWith("WEEKDAY_") &&
      text.slice(span.start, span.end).toLowerCase() === "friday",
  );

  expect(friday?.actionable).toBe(true);
});

test("colliding identical fragments are skipped", () => {
  const frame = CORRECTION_FRAMES[0]!;
  const thursday = weekday("thursday");

  expect(wrapCorrection(thursday, weekday("thursday"), frame)).toBeNull();
  expect(wrapCorrection(new Sentence(), weekday("friday"), frame)).toBeNull();
});

test("unresolved or-alternatives are not correction frames", () => {
  expect(
    CORRECTION_FRAMES.some((frame) =>
      frame.pieces.some((piece) => typeof piece !== "string" && /\bor\b/.test(piece.glue)),
    ),
  ).toBe(false);

  const rows = expandKinds(new Set(["correction"]));

  expect(rows.some((row) => /\bor\b/.test(row.example.text))).toBe(false);
});

test("correction kinds keep disjoint wrap frames", () => {
  const examples = assignSplits(expandKinds(new Set(["correction"])));

  expect(new Set(examples.map((example) => example.split))).toEqual(
    new Set(["train", "validation", "heldout"]),
  );

  const frameSplits = new Map<string, Set<string>>();

  for (const row of examples) {
    const key = frameKey(row);
    const splits = frameSplits.get(key) ?? new Set<string>();

    splits.add(row.split);
    frameSplits.set(key, splits);
  }

  expect([...frameSplits.values()].every((splits) => splits.size === 1)).toBe(true);
});

test("correction gold keeps the replacement weekday as the intent, not a rewritten string", () => {
  const rows = expandKinds(new Set(["correction"]));

  expect(rows.some((row) => row.example.speechAct === "correction")).toBe(true);

  expect(rows.every((row) => row.example.spans.some((span) => span.actionable === true))).toBe(
    true,
  );
});
