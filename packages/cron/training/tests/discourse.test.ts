import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import { tokenize } from "../../src/model/tokenizer";
import { assignSplits, frameKey, generatorSha256 } from "../src/backward";
import { wrapDiscourse } from "../src/discourse";
import { resolveRepo } from "../src/evaluate";
import { fingerprintExample } from "../src/fingerprint";
import { expandKinds } from "../src/kinds";
import type { KindId } from "../src/kinds";
import { Sentence } from "../src/sentence";

function weekdayNine(): Sentence {
  const inner = new Sentence();

  inner.clause();
  inner.add("every", "RECUR");
  inner.add("weekday", "DAYGROUP_WEEKDAY");
  inner.add("at", "GLUE");
  inner.add("9", "HOUR_9");

  return inner;
}

function aligned(sentence: Sentence): void {
  const tokens = tokenize(sentence.text).filter((token) => token.kind !== 3);

  for (const token of tokens) {
    const covering = sentence.spans.filter(
      (span) => token.start >= span.start && token.end <= span.end,
    );

    expect(covering).toHaveLength(1);
  }
}

test("descriptive wrap keeps inner roles and empties the actionable set", () => {
  const wrapped = wrapDiscourse(weekdayNine(), "descriptive", {
    id: "shop-open",
    prefix: "the shop is open",
  });

  expect(wrapped.text).toContain("weekday");
  expect(wrapped.text).toContain("9");
  expect(wrapped.text.startsWith("the shop is open")).toBe(true);

  const labels = Object.fromEntries(
    wrapped.spans.map((span) => [wrapped.text.slice(span.start, span.end), span.label]),
  );

  expect(labels.every).toBe("RECUR");
  expect(labels.weekday).toBe("DAYGROUP_WEEKDAY");
  expect(labels.at).toBe("GLUE");
  expect(labels["9"]).toBe("HOUR_9");
  expect(labels["the shop is open"]).toBe("O");
  expect(wrapped.spans.every((span) => span.actionable === false)).toBe(true);
  aligned(wrapped);
});

test("request wrap keeps inner non-O tokens actionable", () => {
  const wrapped = wrapDiscourse(weekdayNine(), "request", {
    id: "shop-open",
    prefix: "the shop is open",
  });

  const byLabel = new Map(wrapped.spans.map((span) => [span.label, span.actionable] as const));

  expect(byLabel.get("O")).toBe(false);
  expect(byLabel.get("RECUR")).toBe(true);
  expect(byLabel.get("DAYGROUP_WEEKDAY")).toBe(true);
  expect(byLabel.get("GLUE")).toBe(true);
  expect(byLabel.get("HOUR_9")).toBe(true);
});

test("quoted wrap keeps inner labels and marks quotes as O", () => {
  const wrapped = wrapDiscourse(weekdayNine(), "quoted", {
    id: "docs-quote",
    prefix: "the note says",
    quoted: true,
  });

  expect(wrapped.text).toContain('"');
  expect(wrapped.text).toContain("the note says");
  const quotes = wrapped.spans.filter((span) => wrapped.text.slice(span.start, span.end) === '"');

  expect(quotes.length).toBe(2);
  expect(quotes.every((span) => span.label === "O" && span.actionable === false)).toBe(true);

  expect(
    wrapped.spans.some((span) => span.label.startsWith("DAYGROUP_") && span.actionable === false),
  ).toBe(true);

  aligned(wrapped);
});

test("request and descriptive wraps of the same inner have different fingerprints", () => {
  const inner = weekdayNine();
  const template = { id: "shop-open", prefix: "the shop is open" };
  const request = wrapDiscourse(inner, "request", template);
  const descriptive = wrapDiscourse(inner, "descriptive", template);

  expect(request.text).toBe(descriptive.text);

  expect(
    fingerprintExample({
      text: request.text,
      spans: request.spans,
      speechAct: "request",
    }),
  ).not.toBe(
    fingerprintExample({
      text: descriptive.text,
      spans: descriptive.spans,
      speechAct: "descriptive",
    }),
  );
});

test("none is not a wrap frame", () => {
  expect(() => wrapDiscourse(weekdayNine(), "none", { id: "shop-open" })).toThrow(RangeError);
});

const DISCOURSE_KIND_IDS = [
  "descriptive",
  "question",
  "quoted",
  "one-off",
  "cancel",
  "incomplete",
] as const satisfies readonly KindId[];

test("gen:kinds emits each discourse frame with null intent and matching speech-act", () => {
  const examples = expandKinds(new Set<KindId>(DISCOURSE_KIND_IDS));

  for (const id of DISCOURSE_KIND_IDS) {
    const rows = examples.filter((example) => example.kind === id);

    expect(rows.length).toBeGreaterThan(0);
    expect(rows.every((example) => example.intent === null)).toBe(true);
    expect(rows.every((example) => example.example.speechAct === id)).toBe(true);

    expect(rows.every((example) => example.example.spans.every((span) => !span.actionable))).toBe(
      true,
    );

    expect(rows.some((example) => example.example.spans.some((span) => span.label !== "O"))).toBe(
      true,
    );
  }
});

test("descriptive wrap of a daily inner keeps INTERVAL and HOUR", () => {
  const row = expandKinds(new Set<KindId>(["descriptive"])).find((example) =>
    example.example.text.includes("the shop is open"),
  );

  expect(row).toBeDefined();
  expect(row!.example.text).toContain("daily");

  expect(
    row!.example.spans.some(
      (span) => span.label.startsWith("INTERVAL") || span.label.startsWith("UNIT_"),
    ),
  ).toBe(true);

  expect(row!.example.spans.some((span) => span.label.startsWith("HOUR_"))).toBe(true);
  expect(row!.example.spans.filter((span) => span.label === "O").length).toBeGreaterThan(0);
});

test("question wrap of a weekly clock keeps WEEKDAY and HOUR", () => {
  const row = expandKinds(new Set<KindId>(["question"]))[0]!;

  expect(row.example.speechAct).toBe("question");

  expect(
    row.example.spans.some(
      (span) => span.label.startsWith("WEEKDAY_") || span.label.startsWith("DAYGROUP_"),
    ),
  ).toBe(true);

  expect(row.example.spans.some((span) => span.label.startsWith("HOUR_"))).toBe(true);
});

test("quoted wrap of an interval keeps INTERVAL and quote tokens", () => {
  const row = expandKinds(new Set<KindId>(["quoted"])).find((example) =>
    example.example.text.includes('"'),
  );

  expect(row).toBeDefined();

  expect(
    row!.example.spans.some(
      (span) => span.label.startsWith("INTERVAL") || span.label.startsWith("UNIT_"),
    ),
  ).toBe(true);

  expect(
    row!.example.spans.some(
      (span) => row!.example.text.slice(span.start, span.end) === '"' && span.label === "O",
    ),
  ).toBe(true);
});

test("one-off, cancel, and incomplete wraps keep inner labels", () => {
  for (const id of ["one-off", "cancel", "incomplete"] as const) {
    const row = expandKinds(new Set<KindId>([id]))[0]!;

    expect(row.intent).toBeNull();
    expect(row.example.speechAct).toBe(id);
    expect(row.example.spans.some((span) => span.label !== "O")).toBe(true);
  }
});

test("easy negatives remain all-O none rows", () => {
  const rows = expandKinds(new Set<KindId>(["negative"]));

  expect(rows.length).toBeGreaterThan(0);
  expect(rows.every((example) => example.example.speechAct === "none")).toBe(true);

  expect(rows.every((example) => example.example.spans.every((span) => span.label === "O"))).toBe(
    true,
  );
});

test("a wrapped descriptive does not share a fingerprint with the unwrapped request", () => {
  const inner = weekdayNine();

  const wrapped = wrapDiscourse(inner, "descriptive", {
    id: "shop-open",
    prefix: "the shop is open",
  });

  expect(
    fingerprintExample({
      text: inner.text,
      spans: inner.spans,
      speechAct: "request",
    }),
  ).not.toBe(
    fingerprintExample({
      text: wrapped.text,
      spans: wrapped.spans,
      speechAct: "descriptive",
    }),
  );
});

test("discourse kinds have disjoint train, validation, and held-out wrap frames", () => {
  const examples = assignSplits(expandKinds(new Set<KindId>(DISCOURSE_KIND_IDS)));

  for (const id of DISCOURSE_KIND_IDS) {
    const rows = examples.filter((example) => example.kind === id);

    expect(new Set(rows.map((example) => example.split))).toEqual(
      new Set(["train", "validation", "heldout"]),
    );

    const frameSplits = new Map<string, Set<string>>();

    for (const row of rows) {
      const key = frameKey(row);
      const splits = frameSplits.get(key) ?? new Set<string>();

      splits.add(row.split);
      frameSplits.set(key, splits);
    }

    expect([...frameSplits.values()].every((splits) => splits.size === 1)).toBe(true);
  }
});

test("generator sources do not contain consumed human-evaluation wording", () => {
  const sources = [
    "packages/cron/training/src/kinds.ts",
    "packages/cron/training/src/discourse.ts",
    "packages/cron/training/src/renderer.ts",
    "packages/cron/training/src/backward.ts",
  ].map((path) => readFileSync(resolveRepo(path), "utf8"));

  const forbidden = [
    "The bakery opens every morning at half past six.",
    "Flights depart every twenty minutes from this gate.",
    "The tutorial's sample command says “every weekday at 9”.",
  ];

  for (const source of sources) {
    for (const text of forbidden) expect(source).not.toContain(text);
  }

  expect(generatorSha256().length).toBe(64);
});
