import { expect, test } from "bun:test";

import { CorpusAudit } from "../src/corpus/production/audit";
import { semanticCases } from "../src/corpus/production/cases";
import {
  loadConstructions,
  parsePattern,
  renderConstruction,
  type SemanticCase,
} from "../src/corpus/production/constructions";
import { ProtectedText } from "../src/corpus/production/protected-text";
import { verifyDiscarded } from "../src/corpus/production/source-audit";
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

test("every authored form renders each applicable semantic family with sound gold and grouped splits", () => {
  const audit = new CorpusAudit({
    enforceMix: false,
    protectedText: new ProtectedText([]),
  });

  try {
    for (const group of source.groups) {
      const seen = new Set<string>();

      for (const semantic of semanticCases(group.caseSet)) {
        if (seen.has(semantic.semanticFamily)) continue;
        seen.add(semantic.semanticFamily);

        for (const form of group.forms) {
          const row = renderConstruction(source, group, form, semantic);

          audit.add(row);

          expect(row.provenance).toMatchObject({
            authorship: "ai-authored",
            groupId: group.id,
          });
        }
      }
    }

    const report = audit.finish();

    expect(report.violations).toEqual({});
    expect(report.missingProvenance).toBe(0);
    expect(report.explicitCorrections).toBeGreaterThan(0);

    for (const counts of Object.values(report.splits)) {
      expect(counts.R).toBeGreaterThan(0);
      expect(counts.H).toBeGreaterThan(0);
    }
  } finally {
    audit.close();
  }
});

test("interrogative requests are executable, while informational questions keep their inner roles but refuse", () => {
  const semantic = findCase("interval", (c) => c.id === "minute-7-digit-full");
  const request = rowFor("interval-repeat-action", semantic, 2);
  const question = rowFor("interval-repeat-action", semantic, 4);

  expect(request.text).toBe("could you check the queue every 7 minutes from now on");

  expect(decode(request)).toEqual({
    ok: true,
    intent: { frequency: { unit: "minute", interval: 7 } },
  });

  expect(question.speechAct).toBe("question");
  expect(decode(question).ok).toBe(false);
  expect(question.spans.some((s) => s.label === "INTERVAL_7")).toBe(true);
  expect(question.spans.every((s) => !s.actionable)).toBe(true);
});

test("clock words resolve across midnight, and generic minute literals stay numeric", () => {
  const semantic = findCase("daily", (c) => c.id === "23-45-natural");
  const row = rowFor("daily-split-task", semantic);

  expect(row.text).toContain("quarter to twelve am");

  expect(decode(row)).toEqual({
    ok: true,
    intent: { frequency: { unit: "day", interval: 1 }, time: { hour: 23, minute: 45 } },
  });

  const oddMinute = findCase("daily", (c) => c.id === "22-58-12h");

  expect(decode(rowFor("daily-split-task", oddMinute))).toEqual({
    ok: true,
    intent: { frequency: { unit: "day", interval: 1 }, time: { hour: 22, minute: 58 } },
  });

  for (const c of semanticCases("daily")) {
    for (const fragment of c.slots.clock!) {
      if (fragment.label === "MINUTE_LIT") expect(fragment.text).toMatch(/^\d+$/);
    }
  }
});

test("all hours and minutes vary independently in the daily grid, without unsupported spelled-out intervals", () => {
  const clocks = new Set<string>();

  for (const c of semanticCases("daily")) {
    clocks.add(`${c.intent.time!.hour}:${c.intent.time!.minute}`);
  }

  expect(clocks.size).toBe(24 * 60);

  for (const c of semanticCases("interval")) {
    if (c.slots.n![0]!.label === "INTERVAL_LIT") expect(c.slots.n![0]!.text).toMatch(/^\d+$/);
  }
});

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

test("runtime-disallowed interval magnitudes are recorded as exclusions, not mislabeled hard negatives", () => {
  const ids = new Set([...semanticCases("interval")].map((c) => c.id));

  expect(ids.has("minute-59-digit-full")).toBe(true);
  expect(ids.has("minute-60-digit-full")).toBe(false);
  expect(ids.has("hour-23-digit-full")).toBe(true);
  expect(ids.has("hour-25-digit-full")).toBe(false);
  // Day/month compiler limits must not become parser-domain restrictions.
  expect(ids.has("day-120-digit-full")).toBe(true);
  expect(ids.has("month-120-digit-full")).toBe(true);
});

test("punctuation joins naturally across semantic slots, while explicit recurrence words retain their roles", () => {
  const semantic = findCase("weekly", (c) => c.id === "monday-full-0-0-24h");
  const row = rowFor("weekly-selector-first", semantic);

  expect(row.text).toBe("on monday, send the status summary at 00:00 every week");
  const week = row.spans.find((s) => row.text.slice(s.start, s.end) === "week")!;

  expect(week).toMatchObject({ label: "UNIT_WEEK", actionable: true });
  const fact = rowFor("weekly-selector-first", semantic, 3);

  expect(fact.text).toBe("the status summary arrives on monday at 00:00");
  expect(decode(fact).ok).toBe(false);
});

test("interval and clock interact without losing either value, while untimed selectors invent no clock", () => {
  const timed = findCase("interval-clock", (c) => c.id === "day-7-digit-full-7-0-24h");

  expect(decode(rowFor("interval-clock-interleaved", timed))).toEqual({
    ok: true,
    intent: { frequency: { unit: "day", interval: 7 }, time: { hour: 7, minute: 0 } },
  });

  const untimed = findCase("weekly-untimed", (c) => c.id === "monday-full");

  expect(decode(rowFor("weekly-untimed-plain", untimed))).toEqual({
    ok: true,
    intent: { frequency: { unit: "week", interval: 1 }, weekdays: ["monday"] },
  });
});

test("controlled prose typos change words, not schedule values, actionability, or source ownership", () => {
  const group = source.groups.find((g) => g.id === "daily-split-task")!;
  const semantic = findCase("daily", (c) => c.id === "22-58-24h");
  const original = renderConstruction(source, group, group.forms[0]!, semantic);

  const typo = renderConstruction(
    source,
    group,
    group.forms.find((f) => f.id === "request-1-prose-typo")!,
    semantic,
  );

  expect(original.text).not.toBe(typo.text);

  const labeled = (row: typeof original) =>
    row.spans
      .filter((s) => s.label !== "O")
      .map((s) => [row.text.slice(s.start, s.end), s.label, s.actionable]);

  expect(labeled(typo)).toEqual(labeled(original));
  expect(typo.provenance!.groupId).toBe(original.provenance!.groupId);
  expect(decode(typo)).toEqual(decode(original));
});

test("authored ranges cover all 42 directed endpoint pairs, including Sunday-to-Monday and a full week crossing the boundary", () => {
  const endpoints = new Set<string>();

  const audit = new CorpusAudit({
    enforceMix: false,
    protectedText: new ProtectedText([]),
  });

  try {
    for (const semantic of semanticCases("weekly-untimed")) {
      if (semantic.semanticFamily !== "weekday-range") continue;
      const days = semantic.intent.weekdays!;

      endpoints.add(`${days[0]}/${days.at(-1)}`);

      for (const group of source.groups.filter((g) => g.caseSet === "weekly-untimed")) {
        for (const form of group.forms) {
          audit.add(renderConstruction(source, group, form, semantic));
        }
      }
    }

    expect(endpoints.size).toBe(42);
    expect(endpoints.has("sunday/monday")).toBe(true);
    expect(endpoints.has("tuesday/monday")).toBe(true);
    expect(audit.finish().violations).toEqual({});
  } finally {
    audit.close();
  }
});
