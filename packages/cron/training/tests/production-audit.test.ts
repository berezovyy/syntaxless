import { expect, test } from "bun:test";

import type { SpeechAct } from "../../src/model/speech-acts";
import type { CorpusRecord } from "../src/corpus/merge";
import { CorpusAudit } from "../src/corpus/production/audit";
import { ProtectedText } from "../src/corpus/production/protected-text";
import { fingerprintExample, sha256Hex } from "../src/fingerprint";
import { Sentence } from "../src/sentence";

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

test("explicit actionability and token alignment are required, even when decoding could default them", () => {
  const row = interval();

  delete row.spans[1]!.actionable;
  row.spans[0]!.start = 1;
  const report = inspect([row]);

  expect(report.violations["missing-actionability"]).toBe(1);
  expect(report.violations["token-coverage"]).toBe(1);
  expect(report.passed).toBe(false);
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

test("protected wording remains blocked after punctuation changes or embedding in a longer message", () => {
  const protectedText = new ProtectedText(["Activate the kiln check every 7 hours."]);

  expect(inspect([interval()], { protectedText }).violations["protected-wording"]).toBe(1);

  expect(
    protectedText.matches("Context: ACTIVATE the kiln check every 7 hours; that is the line"),
  ).toBe(true);

  expect(protectedText.matches("activate the pressure check every 8 hours")).toBe(false);
});

test("self-consistent but lexically false gold is not excused by oracle agreement", () => {
  const row = interval();

  row.spans.find((span) => span.label === "INTERVAL_7")!.label = "INTERVAL_8";
  row.intent!.frequency.interval = 8;
  expect(inspect([row]).violations).toEqual({ "role-contradicts-literal": 1 });
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
