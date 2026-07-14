import { expect, test } from "bun:test";
import { join } from "node:path";

import type { CorpusRecord } from "../src/corpus/merge";
import {
  corpusSourceReference,
  CorpusUnionPlan,
  StreamingCorpusUnion,
} from "../src/corpus/production/global-union";
import { sha256Hex } from "../src/fingerprint";
import { Sentence } from "../src/sentence";
import { tempDir } from "./helpers/temp-dir";

function record(
  id: string,
  text: string,
  options: {
    split?: CorpusRecord["split"];
    authorship?: NonNullable<CorpusRecord["provenance"]>["authorship"];
    interval?: number;
    group?: string;
  } = {},
): CorpusRecord {
  const interval = options.interval ?? 7;
  const sentence = new Sentence();

  sentence.add(text, "O", " ", false);
  sentence.clause();
  sentence.add("every", "RECUR", " ", true);
  sentence.add(String(interval), `INTERVAL_${interval}` as "INTERVAL_7", " ", true);
  sentence.add("hours", "UNIT_HOUR", " ", true);

  return {
    id,
    split: options.split ?? "train",
    family: "interval",
    fingerprint: sha256Hex(`frame/${interval}`),
    text: sentence.text,
    intent: { frequency: { unit: "hour", interval } },
    speechAct: "request",
    spans: sentence.spans,
    source: "rule",
    provenance: {
      authorship: options.authorship ?? "programmatic",
      sourceId: "test",
      groupId: options.group ?? id,
      constructionId: "interval",
      sourceRecordId: id,
    },
  };
}

function build(
  rows: CorpusRecord[],
  partitions: number,
): { plan: CorpusUnionPlan; selected: CorpusRecord[] } {
  const directory = join(tempDir("syntaxless-union-"), "plan");
  const union = new StreamingCorpusUnion({ directory, partitions });

  for (const row of rows) union.add(row);
  const plan = union.finalize();
  // Prove the persisted plan is independently reopenable.
  const replay = new CorpusUnionPlan(directory).replay();
  const selected = rows.filter((row) => replay.add(row).selected);

  replay.finish();

  return { plan, selected };
}

test("global union keeps every unique normalized text without storing record payloads", () => {
  const a = record("a", "Start the kiln", { authorship: "human-authored" });
  const duplicate = record("b", "START THE KILN");
  const c = record("c", "Start the pump");
  const { plan, selected } = build([a, duplicate, c], 3);

  expect(plan.summary).toMatchObject({ candidates: 3, uniqueTexts: 2, duplicates: 1 });
  expect(selected.map((row) => row.id)).toEqual(["a", "c"]);
  expect(corpusSourceReference(a)).toContain("test/a");
});

test("partition count and input order do not change deterministic priority winner", () => {
  const programmatic = record("programmatic", "Notify the operator");

  const human = record("human", "Notify the operator", {
    authorship: "human-authored",
  });

  const other = record("other", "Notify the supervisor");
  const forward = build([programmatic, human, other], 1);
  const reverse = build([other, human, programmatic], 7);

  expect(forward.selected.map((row) => row.id).sort()).toEqual(["human", "other"]);
  expect(reverse.selected.map((row) => row.id).sort()).toEqual(["human", "other"]);
  expect(forward.plan.summary.uniqueTexts).toBe(reverse.plan.summary.uniqueTexts);
});

test("equal-priority duplicates use a stable provenance reference tie-break", () => {
  const alpha = record("alpha", "Notify the operator", { group: "alpha" });
  const beta = record("beta", "Notify the operator", { group: "beta" });
  const forward = build([alpha, beta], 1).selected[0]!;
  const reverse = build([beta, alpha], 13).selected[0]!;

  expect(forward.id).toBe(reverse.id);
  expect(corpusSourceReference(forward)).toBe(corpusSourceReference(reverse));
});

test("same normalized text with conflicting supervision fails closed", () => {
  const directory = join(tempDir("syntaxless-union-conflict-"), "plan");
  const union = new StreamingCorpusUnion({ directory, partitions: 5 });

  union.add(record("seven", "Start the kiln", { interval: 7 }));
  const wrong = record("eight", "Start the kiln", { interval: 7 });

  wrong.intent!.frequency.interval = 8;
  union.add(wrong);
  expect(() => union.finalize()).toThrow("conflicting supervision");
});

test("exact span supervision differences fail closed without tokenizing", () => {
  const directory = join(tempDir("syntaxless-union-spans-"), "plan");
  const union = new StreamingCorpusUnion({ directory, partitions: 3 });
  const first = record("first", "Start the kiln");
  const changed = record("changed", "Start the kiln");

  changed.spans = changed.spans.map((span, index) =>
    index === 0 ? { ...span, clauseStart: !span.clauseStart } : span,
  );

  union.add(first);
  union.add(changed);
  expect(() => union.finalize()).toThrow("conflicting supervision");
});

test("same-supervision duplicates across splits emit only the priority winner's split", () => {
  const directory = join(tempDir("syntaxless-union-split-"), "plan");
  const union = new StreamingCorpusUnion({ directory, partitions: 11 });

  union.add(record("train", "Start the kiln"));

  const heldout = record("heldout", "Start the kiln", {
    split: "heldout",
    authorship: "human-authored",
  });

  union.add(heldout);
  const plan = union.finalize();
  const replay = plan.replay();

  expect(replay.add(record("train", "Start the kiln")).selected).toBe(false);

  expect(replay.add(heldout)).toEqual({
    selected: true,
    reference: corpusSourceReference(heldout),
    split: "heldout",
    textHash: expect.any(String),
  });

  replay.finish();
});

test("family and fingerprint metadata do not conflict when model supervision is equal", () => {
  const first = record("first", "Start the kiln", {
    authorship: "human-authored",
  });

  const duplicate = record("duplicate", "Start the kiln");

  duplicate.family = "terse";
  duplicate.fingerprint = "different-source-frame-metadata";
  const { plan, selected } = build([first, duplicate], 4);

  expect(plan.summary).toMatchObject({ uniqueTexts: 1, duplicates: 1 });
  expect(selected.map((row) => row.id)).toEqual(["first"]);
});

test("replay rejects a changed source sequence instead of emitting a wrong winner", () => {
  const a = record("a", "Start the kiln");
  const b = record("b", "Start the pump");
  const { plan } = build([a, b], 2);
  const replay = plan.replay();

  replay.add(b);
  replay.add(a);
  expect(() => replay.finish()).toThrow("differs from the planning pass");
});
