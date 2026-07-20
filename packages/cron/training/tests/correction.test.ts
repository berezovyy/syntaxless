import { expect, test } from "bun:test";

import { tokenize } from "../../src/model/tokenizer";
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

test("colliding identical fragments are skipped", () => {
  const frame = CORRECTION_FRAMES[0]!;
  const thursday = weekday("thursday");

  expect(wrapCorrection(thursday, weekday("thursday"), frame)).toBeNull();
  expect(wrapCorrection(new Sentence(), weekday("friday"), frame)).toBeNull();
});
