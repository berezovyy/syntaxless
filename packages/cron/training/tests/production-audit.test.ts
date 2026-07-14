import { expect, test } from "bun:test";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type { SpeechAct } from "../../src/model/speech-acts";
import type { CorpusRecord } from "../src/corpus/merge";
import { auditCorpus, CorpusAudit } from "../src/corpus/production/audit";
import { catalogCandidates } from "../src/corpus/production/catalog";
import { loadProtectedText, ProtectedText } from "../src/corpus/production/protected-text";
import { fingerprintExample, sha256Hex } from "../src/fingerprint";
import { KINDS } from "../src/kinds";
import { normalizeAnnotatedText, Sentence } from "../src/sentence";
import { tempDir } from "./helpers/temp-dir";

function interval(
  id = "r1",
  act: SpeechAct = "request",
  prose = "activate the kiln check",
): CorpusRecord {
  const sentence = new Sentence();
  const actionable = act === "request" || act === "correction";

  sentence.add(prose, "O", " ", false);
  sentence.clause();
  sentence.add("every", "RECUR", " ", actionable);
  sentence.add("7", "INTERVAL_7", " ", actionable);
  sentence.add("hours", "UNIT_HOUR", " ", actionable);

  return {
    id,
    split: "train",
    family: "interval",
    text: sentence.text,
    spans: sentence.spans,
    intent: { frequency: { unit: "hour", interval: 7 } },
    speechAct: act,
    source: "rule",
    template: "interval/kiln",
    fingerprint: fingerprintExample({
      ...sentence,
      text: sentence.text,
      spans: sentence.spans,
      speechAct: act,
    }),
    provenance: {
      authorship: "ai-authored",
      sourceId: "test",
      groupId: "kiln",
      constructionId: act,
    },
  };
}

function inspect(
  rows: unknown[],
  options: { protectedText?: ProtectedText; enforceMix?: boolean } = {},
) {
  const audit = new CorpusAudit({
    protectedText: new ProtectedText([]),
    enforceMix: false,
    ...options,
  });

  try {
    for (const row of rows) audit.add(row);

    return audit.finish();
  } finally {
    audit.close();
  }
}

test("audit checks both an executable request and the annotated meaning of its refused twin", () => {
  const report = inspect([
    interval(),
    interval("h1", "descriptive", "the log specifies a kiln check"),
  ]);

  expect(report.passed).toBe(true);
  expect(report.validRecords).toBe(2);
  expect(report.populations).toEqual({ R: 1, H: 1, E: 0 });
  expect(report.values).toEqual({ unit: { hour: 2 }, "interval/hour": { "7": 2 } });
  // Record-level checks never certify source diversity or model readiness.
  expect(report.scope).toBe("record-integrity-and-split-isolation-only");
});

test("wrong gold cannot pass just because hard negatives always refuse", () => {
  const request = interval();
  const negative = interval("h1", "quoted", "the instruction sheet reads");

  request.intent!.frequency.interval = 8;
  negative.intent!.frequency.interval = 8;
  const report = inspect([request, negative]);

  expect(report.violations).toEqual({
    "gold-decode-mismatch": 1,
    "negative-inner-gold-mismatch": 1,
  });
});

test("zeroed negative roles do not masquerade as valid ordinary prose", () => {
  const row = interval("h1", "descriptive");

  for (const span of row.spans) {
    span.label = "O";
    span.clauseStart = false;
  }

  row.fingerprint = sha256Hex(`negative\0${row.text}`);

  expect(inspect([row]).violations).toEqual({
    "invalid-easy-negative": 1,
    "schedule-language-in-easy-negative": 1,
  });
});

test("explicit actionability and token alignment are required, even when decoding could default them", () => {
  const row = interval();

  delete row.spans[1]!.actionable;
  row.spans[0]!.start = 1;
  const report = inspect([row]);

  expect(report.violations["missing-actionability"]).toBe(1);
  expect(report.violations["token-coverage"]).toBe(1);
  expect(report.passed).toBe(false);
});

test("a request and counterfactual with different text/fingerprints must still share a split", () => {
  const a = interval();
  const b = interval("h1", "question", "does the equipment manual specify a kiln check");

  b.split = "heldout";
  const report = inspect([a, b]);

  expect(report.violations).toEqual({ "group-crosses-splits": 1 });
});

test("value substitutions cannot cross splits under newly invented group IDs", () => {
  const a = interval();
  const b = interval("r2");

  b.text = b.text.replace("7", "8");
  b.spans.find((s) => s.label === "INTERVAL_7")!.label = "INTERVAL_8";
  b.intent!.frequency.interval = 8;
  b.provenance!.groupId = "different-name";
  b.split = "validation";
  // Content values intentionally have the same structural fingerprint.
  expect(inspect([a, b]).violations).toEqual({ "fingerprint-crosses-splits": 1 });
});

test("identical wording with conflicting request/refusal supervision is rejected", () => {
  const report = inspect([interval(), interval("h1", "descriptive")]);

  expect(report.violations).toEqual({ "contradictory-supervision": 1 });
});

test("range and list punctuation is semantic identity, while normalized duplicate supervision still conflicts", () => {
  const candidates = [
    ...catalogCandidates(KINDS.filter((k) => k.id === "weekday-list" || k.id === "weekday-range")),
  ];

  const list = candidates.find((c) => c.record.text === "mon/fri")!.record;
  const range = candidates.find((c) => c.record.text === "mon-fri")!.record;

  expect(list.intent!.weekdays).toEqual(["monday", "friday"]);
  expect(range.intent!.weekdays).toEqual(["monday", "tuesday", "wednesday", "thursday", "friday"]);

  const records: CorpusRecord[] = [list, range].map((row) => ({
    ...row,
    split: "train",
  }));

  expect(inspect(records).violations).toEqual({});
  // A conservative contamination guard deliberately still treats these as related.
  expect(new ProtectedText(["mon/fri"]).matches("mon-fri")).toBe(true);

  const duplicate = {
    ...records[0]!,
    id: "duplicate",
    speechAct: "descriptive" as const,
    spans: records[0]!.spans.map((s) => ({ ...s, actionable: false })),
  };

  duplicate.fingerprint = fingerprintExample(duplicate);

  expect(inspect([records[0]!, duplicate]).violations).toEqual({
    "contradictory-supervision": 1,
  });
});

test("repetition is reported separately from distinct sentence count", () => {
  const a = interval();
  const b = { ...a, id: "r2" };
  const report = inspect([a, b]);

  expect(report.passed).toBe(true);

  expect([
    report.records,
    report.uniqueTexts,
    report.repeatedRows,
    report.maxTextRepetitions,
  ]).toEqual([2, 1, 1, 2]);
});

test("protected wording remains blocked after punctuation changes or embedding in a longer message", () => {
  const protectedText = new ProtectedText(["Activate the kiln check every 7 hours."]);

  expect(inspect([interval()], { protectedText }).violations["protected-wording"]).toBe(1);

  expect(
    protectedText.matches("Context: ACTIVATE the kiln check every 7 hours; that is the line"),
  ).toBe(true);

  expect(protectedText.matches("activate the pressure check every 8 hours")).toBe(false);
});

test("production protection includes calibration and reserved gold without exposing their text", () => {
  const protectedText = loadProtectedText();

  const calibration = protectedText.sources.find((s) =>
    s.file.endsWith("human-calibration-v2.jsonl"),
  )!;

  const gold = protectedText.sources.find((s) => s.file.endsWith("reserved-carriers.jsonl"))!;

  for (const source of [calibration, gold]) {
    const row = JSON.parse(readFileSync(source.file, "utf8").split("\n")[0]!) as {
      text: string;
    };

    expect(protectedText.matches(row.text)).toBe(true);
    expect(source.records).toBeGreaterThan(0);
    expect(source.sha256).toMatch(/^[a-f0-9]{64}$/);
  }
});

test("a corpus cannot pass production certification with missing source metadata or empty populations", () => {
  const row = interval();

  delete row.provenance;
  const report = inspect([row], { enforceMix: true });

  expect(report.violations["missing-provenance"]).toBe(1);
  expect(report.violations["empty-split-population"]).toBe(3);
  expect(report.passed).toBe(false);
  expect(inspect([]).violations).toEqual({ "empty-corpus": 1 });
});

test("the streaming audit hashes the actual file and includes malformed lines in failure totals", async () => {
  const path = join(tempDir("cron-production-audit-"), "rows.jsonl");
  const text = `${JSON.stringify(interval())}\nnot json\n`;

  writeFileSync(path, text);

  const report = await auditCorpus(path, {
    protectedText: new ProtectedText([]),
    enforceMix: false,
  });

  expect(report.sha256).toBe(sha256Hex(text));
  expect(report.records).toBe(2);
  expect(report.validRecords).toBe(1);
  expect(report.violations).toEqual({ schema: 1 });
  expect(report.complete).toBe(true);
  expect(report.passed).toBe(false);
});

test("assembled compact meridiems normalize without shifting supervision onto whitespace", () => {
  const sentence = new Sentence();

  sentence.add("daily", "UNIT_DAY");
  sentence.add("at", "GLUE");
  sentence.add("9", "HOUR_9");
  sentence.add(":", "GLUE", "");
  sentence.add("25", "MINUTE_LIT", "");
  sentence.add("am", "MERIDIEM_AM", "");
  const original = structuredClone(sentence.spans);
  const result = normalizeAnnotatedText(sentence.text, sentence.spans);

  expect(result.text).toBe("daily at 9:25 am");

  expect(result.spans.map((s) => [result.text.slice(s.start, s.end), s.label])).toEqual([
    ["daily", "UNIT_DAY"],
    ["at", "GLUE"],
    ["9", "HOUR_9"],
    [":", "GLUE"],
    ["25", "MINUTE_LIT"],
    ["am", "MERIDIEM_AM"],
  ]);

  expect(result.spans.at(-1)).toMatchObject({ start: 14, end: 16, actionable: true });
  expect(sentence.spans).toEqual(original);
});

test("normalization refuses token-changing annotations instead of guessing new gold", () => {
  expect(() =>
    normalizeAnnotatedText("run.every.7.hours", [
      { start: 0, end: 17, label: "O", actionable: false, clauseStart: false },
    ]),
  ).toThrow("Normalization changes annotated tokens");
});

test("self-consistent but lexically false gold is not excused by oracle agreement", () => {
  const row = interval();

  row.spans.find((span) => span.label === "INTERVAL_7")!.label = "INTERVAL_8";
  row.intent!.frequency.interval = 8;
  expect(inspect([row]).violations).toEqual({ "role-contradicts-literal": 1 });
});

test("resolved 24-hour clock labels are not mistaken for literal-value corruption", () => {
  const sentence = new Sentence();

  sentence.clause();
  sentence.add("daily", "UNIT_DAY");
  sentence.add("at", "GLUE");
  sentence.add("8", "HOUR_20");
  sentence.add("pm", "MERIDIEM_PM");
  const row = interval();

  row.text = sentence.text;
  row.spans = sentence.spans;

  row.intent = {
    frequency: { unit: "day", interval: 1 },
    time: { hour: 20, minute: 0 },
  };

  row.fingerprint = fingerprintExample({
    text: row.text,
    spans: sentence.spans,
    speechAct: "request",
  });

  expect(inspect([row]).violations).toEqual({});
});

test("zeroing roles, intent and speech act together cannot hide a schedule in easy negatives", () => {
  const row = interval();

  row.intent = null;
  row.speechAct = "none";

  for (const span of row.spans) {
    span.label = "O";
    span.clauseStart = false;
    span.actionable = false;
  }

  row.fingerprint = sha256Hex(`negative\0${row.text}`);

  expect(inspect([row]).violations).toEqual({
    "schedule-language-in-easy-negative": 1,
  });
});

test("record identities are unique even when different examples have valid supervision", () => {
  const a = interval();
  const b = interval(a.id, "descriptive", "the manual describes a check");

  expect(inspect([a, b]).violations).toEqual({ "duplicate-or-empty-id": 1 });
});

test("canonical frequency idioms preserve meaning without allowing arbitrary unit changes", () => {
  const sentence = new Sentence();

  sentence.clause();
  sentence.add("every fourteen", "INTERVAL_2");
  sentence.add("days", "UNIT_WEEK");
  const row = interval();

  row.text = sentence.text;
  row.spans = sentence.spans;
  row.intent = { frequency: { unit: "week", interval: 2 } };

  row.fingerprint = fingerprintExample({
    ...sentence,
    text: row.text,
    spans: sentence.spans,
    speechAct: "request",
  });

  expect(inspect([row]).violations).toEqual({});
  // Self-consistent wrong gold must still fail the language check.
  row.spans[0]!.label = "INTERVAL_3";
  row.intent.frequency.interval = 3;
  expect(inspect([row]).violations).toEqual({ "role-contradicts-literal": 1 });
  const plain = interval();

  plain.spans.at(-1)!.label = "UNIT_DAY";
  plain.intent!.frequency.unit = "day";
  expect(inspect([plain]).violations).toEqual({ "role-contradicts-literal": 1 });
});
