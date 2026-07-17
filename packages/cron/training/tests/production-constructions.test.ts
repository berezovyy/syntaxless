import { expect, test } from "bun:test";

import { semanticCases } from "../src/corpus/production/cases";
import {
  loadConstructions,
  parsePattern,
  renderConstruction,
  type SemanticCase,
  verifyDiscarded,
} from "../src/corpus/production/constructions";
import { oracleDecode } from "../src/oracle";

const source = loadConstructions();

function findCase(
  set: Parameters<typeof semanticCases>[0],
  predicate: (value: SemanticCase) => boolean,
): SemanticCase {
  for (const value of semanticCases(set)) if (predicate(value)) return value;
  throw new Error(`Missing semantic case: ${set}`);
}

function rowFor(groupId: string, caseValue: SemanticCase, formIndex = 0) {
  const group = source.groups.find((g) => g.id === groupId)!;

  return renderConstruction(source, group, group.forms[formIndex]!, caseValue);
}

function decode(row: ReturnType<typeof rowFor>) {
  return oracleDecode({ ...row, source: "kinds-sample" });
}

test("inclusive leap-day bounds differ from exclusive end dates rather than copying the printed date into gold", () => {
  const semantic = findCase(
    "bounds",
    (c) => c.semanticFamily === "inclusive-end-bound" && c.id.includes("2028-02-28"),
  );

  const row = rowFor("bounds-window-tail", semantic);

  expect(row.text).toContain("through 2028-02-29");
  const outcome = decode(row);

  expect(outcome.ok && outcome.intent.bounds).toEqual({ end: "2028-03-01" });
});

test("leading exceptions do not swallow a later selector or clock", () => {
  const semantic = findCase(
    "exceptions",
    (c) =>
      c.semanticFamily === "selector-with-exception" &&
      c.intent.exceptions?.weekdays?.[0] === "monday",
  );

  const outcome = decode(rowFor("exceptions-skip-first", semantic));

  expect(outcome.ok && outcome.intent).toEqual({
    frequency: { unit: "week", interval: 1 },
    time: { hour: 0, minute: 0 },
    weekdays: ["monday", "tuesday", "wednesday", "thursday", "friday"],
    exceptions: { weekdays: ["monday"] },
  });
});

test("corrections preserve old roles in either clause order, but only the replacement is actionable", () => {
  const semantic = findCase("correction", (c) => c.id === "minute-7-to-20");

  verifyDiscarded(semantic);
  const oldFirst = rowFor("correction-replace-old-first", semantic, 1);
  const nextFirst = rowFor("correction-replacement-first", semantic);

  for (const row of [oldFirst, nextFirst]) {
    expect(decode(row)).toEqual({
      ok: true,
      intent: { frequency: { unit: "minute", interval: 20 } },
    });

    expect(row.spans.find((s) => s.label === "INTERVAL_7")!.actionable).toBe(false);
    expect(row.spans.find((s) => s.label === "INTERVAL_LIT")!.actionable).toBe(true);
  }

  const corrupt = structuredClone(semantic);

  corrupt.discarded!.intent.frequency.interval = 8;
  expect(() => verifyDiscarded(corrupt)).toThrow("Discarded schedule gold mismatch");
});

test("markup mistakes and missing bindings fail instead of silently becoming prose", () => {
  expect(() => parsePattern("run [UNIT_DAY:daily}")).toThrow("Malformed construction markup");
  expect(() => parsePattern("run [NO_SUCH_LABEL:daily]")).toThrow("Unknown label");
  const semantic = findCase("daily", () => true);
  const group = source.groups.find((g) => g.caseSet === "daily")!;

  expect(() =>
    renderConstruction(source, group, { ...group.forms[0]!, pattern: "run {missing}" }, semantic),
  ).toThrow("Missing slot");
});
