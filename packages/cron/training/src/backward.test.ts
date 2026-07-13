import { expect, test } from "bun:test";

import { normalizeText, RESERVED_PHRASES } from "./background";
import { assignSplits, frameKey } from "./backward";
import { expandKinds, iterateKinds } from "./kinds";
import type { KindExample, KindId } from "./kinds";

function firstKind(kind: KindId, template: string): KindExample | undefined {
  for (const example of iterateKinds(new Set([kind]))) {
    if (example.template === template) return example;
  }

  return undefined;
}

function labelsOf(text: string, spans: readonly { start: number; end: number; label: string }[]) {
  return Object.fromEntries(spans.map((span) => [text.slice(span.start, span.end), span.label]));
}

test("reserved-carrier overlay-off gaps have abstract generator frames", () => {
  const cycle = firstKind("n-interval", "n-interval/cycle-is");

  expect(cycle).toBeDefined();
  expect(labelsOf(cycle!.example.text, cycle!.example.spans)["the cycle is"]).toBe("GLUE");

  const days = firstKind("weekday-list", "weekday-list/the-days-are");

  expect(days).toBeDefined();
  expect(labelsOf(days!.example.text, days!.example.spans)["the days are"]).toBe("GLUE");

  const span = firstKind("weekday-range", "weekday-range/span-runs");

  expect(span).toBeDefined();
  expect(labelsOf(span!.example.text, span!.example.spans)["the span runs"]).toBe("GLUE");

  const terse = firstKind("single-weekday", "single-weekday/days-are");

  expect(terse).toBeDefined();
  expect(labelsOf(terse!.example.text, terse!.example.spans)["days are"]).toBe("GLUE");

  const monthlyDay = firstKind("day-of-month", "day-of-month/the-day-monthly");

  expect(monthlyDay).toBeDefined();
  expect(monthlyDay!.example.text).toMatch(/the \d+(st|nd|rd|th) day, monthly/);
  expect(monthlyDay!.intent?.frequency).toEqual({ unit: "month", interval: 1 });

  const monthlyOrdinal = firstKind("ordinal-weekday", "ordinal-weekday/the-ordinal-monthly");

  expect(monthlyOrdinal).toBeDefined();
  expect(monthlyOrdinal!.example.text).toMatch(/the (first|second|third|fourth|last) \w+, monthly/);

  const isoBound = firstKind("bounds", "bounds/iso-every-day");

  expect(isoBound).toBeDefined();
  expect(isoBound!.intent?.bounds?.start).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  expect(isoBound!.example.text).toMatch(/^\d{4}-\d{2}-\d{2}, every day$/);

  const lead = firstKind("exception-days", "exception-days/lead-2");

  expect(lead).toBeDefined();
  expect(lead!.example.text.startsWith("excluding")).toBe(true);
  expect(lead!.example.text.endsWith("every day")).toBe(true);

  const atCarrier = firstKind("n-interval", "n-interval/carrier-8-cycle-is");

  expect(atCarrier).toBeDefined();
  expect(atCarrier!.example.text.startsWith("the next occurrence is at ")).toBe(true);

  expect(
    labelsOf(atCarrier!.example.text, atCarrier!.example.spans)["the next occurrence is at"],
  ).toBe("O");

  expect(labelsOf(atCarrier!.example.text, atCarrier!.example.spans)["the cycle is"]).toBe("GLUE");

  expect(
    atCarrier!.example.spans
      .filter((span) => span.label === "O")
      .every((span) => span.actionable === false),
  ).toBe(true);

  expect(
    atCarrier!.example.spans
      .filter((span) => span.label !== "O")
      .every((span) => span.actionable === true),
  ).toBe(true);

  const bareCycle = firstKind("n-interval", "n-interval/cycle-is");

  expect(
    bareCycle!.example.spans
      .filter((span) => span.label !== "O")
      .every((span) => span.actionable === true),
  ).toBe(true);

  const trailed = firstKind("n-interval", "n-interval/carrier-8-trail-cycle-is");

  expect(trailed).toBeDefined();
  expect(trailed!.example.text.endsWith(" if that is acceptable")).toBe(true);

  const period = firstKind("n-interval", "n-interval/period-is");

  expect(period).toBeDefined();
  expect(labelsOf(period!.example.text, period!.example.spans)["the period is"]).toBe("GLUE");

  const selected = firstKind("weekday-list", "weekday-list/selected-days-are");

  expect(selected).toBeDefined();

  expect(labelsOf(selected!.example.text, selected!.example.spans)["the selected days are"]).toBe(
    "GLUE",
  );

  const window = firstKind("weekday-range", "weekday-range/window-runs");

  expect(window).toBeDefined();
  expect(labelsOf(window!.example.text, window!.example.spans)["the window runs"]).toBe("GLUE");

  const isoEach = firstKind("bounds", "bounds/iso-each-day");

  expect(isoEach).toBeDefined();
  expect(isoEach!.example.text).toMatch(/^\d{4}-\d{2}-\d{2}, each day$/);

  const stride = firstKind("n-interval", "n-interval/stride-is");

  expect(stride).toBeDefined();
  expect(labelsOf(stride!.example.text, stride!.example.spans)["the stride is"]).toBe("GLUE");

  const weekdaysAre = firstKind("weekday-list", "weekday-list/weekdays-are");

  expect(weekdaysAre).toBeDefined();

  expect(labelsOf(weekdaysAre!.example.text, weekdaysAre!.example.spans)["the weekdays are"]).toBe(
    "GLUE",
  );

  const rangeRuns = firstKind("weekday-range", "weekday-range/range-runs");

  expect(rangeRuns).toBeDefined();

  expect(labelsOf(rangeRuns!.example.text, rangeRuns!.example.spans)["the range runs"]).toBe(
    "GLUE",
  );

  const omitted = firstKind("exception-days", "exception-days/omitted-days-are");

  expect(omitted).toBeDefined();

  expect(labelsOf(omitted!.example.text, omitted!.example.spans)["the omitted days are"]).toBe(
    "EXCEPT",
  );

  expect(omitted!.example.text.endsWith("every day")).toBe(true);

  const leadEnd = firstKind("bounds", "bounds/lead-end-6");

  expect(leadEnd).toBeDefined();
  expect(leadEnd!.example.text.startsWith("ending ")).toBe(true);
  expect(leadEnd!.example.text.endsWith(", every day")).toBe(true);
  expect(leadEnd!.intent?.bounds?.end).toMatch(/^\d{4}-\d{2}-\d{2}$/);

  const scene = firstKind("n-interval", "n-interval/carrier-26-cycle-is");

  expect(scene).toBeDefined();
  expect(scene!.example.text.startsWith("the shuttle departs at ")).toBe(true);
  expect(labelsOf(scene!.example.text, scene!.example.spans)["the shuttle departs at"]).toBe("O");
  expect(labelsOf(scene!.example.text, scene!.example.spans)["the cycle is"]).toBe("GLUE");
  expect(RESERVED_PHRASES.has(normalizeText("the shuttle departs at"))).toBe(false);
});

test("scene-setting train carriers stay disjoint from reserved phrases", () => {
  const wanted = new Map<string, string>([
    ["n-interval/carrier-26-cycle-is", "the shuttle departs at"],
    ["n-interval/carrier-27-cycle-is", "the ferry pulls out at"],
    ["n-interval/carrier-28-cycle-is", "our workshop commences at"],
    ["n-interval/carrier-29-cycle-is", "the briefing kicks off at"],
    ["n-interval/carrier-30-cycle-is", "the clinic unlocks at"],
    ["n-interval/carrier-31-cycle-is", "our lesson gets going at"],
    ["n-interval/carrier-32-cycle-is", "the seminar convenes at"],
    ["n-interval/carrier-33-cycle-is", "the shift gets underway at"],
  ]);

  const found = new Map<string, KindExample>();

  for (const example of iterateKinds(new Set(["n-interval"]))) {
    if (!wanted.has(example.template)) continue;
    found.set(example.template, example);
    if (found.size === wanted.size) break;
  }

  expect(found.size).toBe(wanted.size);

  for (const [template, prefix] of wanted) {
    const example = found.get(template)!;

    expect(example.example.text.startsWith(`${prefix} `)).toBe(true);
    expect(labelsOf(example.example.text, example.example.spans)[prefix]).toBe("O");
    expect(RESERVED_PHRASES.has(normalizeText(prefix))).toBe(false);
  }
});

test("every selected kind has frame-disjoint train, validation, and held-out support", () => {
  const wanted = new Set<KindId>([
    "unit-interval",
    "day-rule",
    "exception-group",
    "frequency-idiom",
  ]);

  const examples = assignSplits(expandKinds(wanted));

  for (const kind of wanted) {
    const rows = examples.filter((example) => example.kind === kind);

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
