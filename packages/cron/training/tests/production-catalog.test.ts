import { expect, test } from "bun:test";

import { canonicalizeIntent } from "../src/canonical-intent";
import { CorpusAudit } from "../src/corpus/production/audit";
import { catalogCandidates } from "../src/corpus/production/catalog";
import { ProtectedText } from "../src/corpus/production/protected-text";
import { type KindDef, KINDS } from "../src/kinds";
import { labeledTokensFromGold, oracleAgrees, oracleDecode } from "../src/oracle";

test("catalog traversal preserves the full interval domain and every applicable template without carrier expansion", () => {
  const kind = KINDS.find((kind) => kind.id === "n-interval")!;
  const seen = new Set<string>();
  const templates = new Set<string>();
  let attempts = 0;

  const audit = new CorpusAudit({
    enforceMix: false,
    protectedText: new ProtectedText([]),
  });

  try {
    for (const candidate of catalogCandidates([kind], () => attempts++)) {
      const i = candidate.record.intent!;

      seen.add(`${i.frequency.unit}/${i.frequency.interval}`);
      templates.add(candidate.constructionId);
      expect(candidate.record).not.toHaveProperty("split");
      expect(candidate.record.speechAct).toBe("request");
      expect(candidate.record.provenance!.authorship).toBe("programmatic");
      audit.add({ ...candidate.record, split: "train" });
    }

    expect(attempts).toBe(kind.intents.length * kind.templates.length);
    expect(templates).toEqual(new Set(kind.templates.map((t) => `${kind.id}/${t.id}`)));
    expect(seen.size).toBe(132);

    for (const [unit, maximum] of [
      ["minute", 59],
      ["hour", 23],
      ["day", 31],
      ["week", 12],
      ["month", 12],
    ] as const) {
      for (let value = 2; value <= maximum; value++) {
        expect(seen.has(`${unit}/${value}`)).toBe(true);
      }
    }

    expect(audit.finish().violations).toEqual({});
  } finally {
    audit.close();
  }
});

test("catalog IDs and labels are independent of input order, render failures are observable, and null discourse is not relabeled as a request", () => {
  const base = KINDS.find((kind) => kind.id === "unit-interval")!;

  const kind: KindDef = {
    ...base,
    templates: [...base.templates, { id: "unsupported", split: "train" }],
  };

  let failed = 0;

  const records = [
    ...catalogCandidates([kind], (attempt) => {
      if (!attempt.rendered) failed++;
    }),
  ];

  expect(failed).toBe(kind.intents.length);

  const reversed = [
    ...catalogCandidates([
      {
        ...kind,
        intents: [...kind.intents].reverse(),
        templates: [...kind.templates].reverse(),
      },
    ]),
  ];

  const sort = (items: typeof records) =>
    items.sort((a, b) => a.record.id.localeCompare(b.record.id));

  expect(sort(records)).toEqual(sort(reversed));

  expect([...catalogCandidates(KINDS.filter((k) => k.intents.every((i) => i === null)))]).toEqual(
    [],
  );
});

test("systematic ranges preserve all 42 directed endpoints, not just non-wrapping subsequences", () => {
  const kind = KINDS.find((kind) => kind.id === "weekday-range")!;
  const endpoints = new Set<string>();

  for (const candidate of catalogCandidates([kind])) {
    const intent = candidate.record.intent!;

    endpoints.add(`${intent.weekdays![0]}/${intent.weekdays!.at(-1)}`);

    expect(
      oracleAgrees(oracleDecode({ ...candidate.record, source: "kinds-sample" }), intent),
    ).toBe(true);
  }

  expect(endpoints.size).toBe(42);
  expect(endpoints.has("friday/monday")).toBe(true);
});

test("overlapping primitive and idiom renderers agree on roles and explicit seven-day semantics", () => {
  const phrases = [
    "monday through friday",
    "monday to friday",
    "every seven days",
    ...["minute", "hour", "day", "week", "month"].map((unit) => `once per ${unit}`),
  ];

  const seen = new Map(phrases.map((phrase) => [phrase, new Set<string>()]));

  for (const candidate of catalogCandidates(
    KINDS.filter((k) =>
      ["unit-interval", "n-interval", "daygroup", "weekday-range", "frequency-idiom"].includes(
        k.id,
      ),
    ),
  )) {
    const row = candidate.record;

    if (!seen.has(row.text)) continue;

    seen.get(row.text)!.add(
      JSON.stringify({
        intent: canonicalizeIntent(row.intent!),
        tokens: labeledTokensFromGold(row.text, row.spans).map((t) => [
          t.role,
          t.boundary,
          t.actionable,
        ]),
      }),
    );

    if (row.text === "every seven days") {
      expect(row.intent!.frequency).toEqual({ unit: "day", interval: 7 });
    }
  }

  for (const phrase of phrases) expect(seen.get(phrase)!.size).toBe(1);
});
