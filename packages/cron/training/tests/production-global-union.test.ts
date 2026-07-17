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
  bounds: { reduceWindowBytes?: number; mergeFanIn?: number } = {},
): { plan: CorpusUnionPlan; selected: CorpusRecord[] } {
  const directory = join(tempDir("syntaxless-union-"), "plan");
  const union = new StreamingCorpusUnion({ directory, partitions, ...bounds });

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

test("same normalized text with conflicting supervision fails closed", () => {
  const directory = join(tempDir("syntaxless-union-conflict-"), "plan");
  const union = new StreamingCorpusUnion({ directory, partitions: 5 });

  union.add(record("seven", "Start the kiln", { interval: 7 }));
  const wrong = record("eight", "Start the kiln", { interval: 7 });

  wrong.intent!.frequency.interval = 8;
  union.add(wrong);
  expect(() => union.finalize()).toThrow("conflicting supervision");
});

/** One fixed-width scratch candidate; the smallest legal reduce window. */
const CANDIDATE_BYTES = 109;

test("recursive splitting and multi-pass merging select exactly what one window does", () => {
  const rows: CorpusRecord[] = [];

  for (let index = 0; index < 600; index++) {
    rows.push(record(`row-${index}`, `Notify operator ${index % 450}`));

    if (index % 7 === 0) {
      rows.push(
        record(`human-${index}`, `Notify operator ${index % 450}`, {
          authorship: "human-authored",
        }),
      );
    }
  }

  const whole = build(rows, 1);

  // Two candidates per window forces every partition to split on further
  // hash bits, and fan-in 2 forces several on-disk merge passes.
  const bounded = build(rows, 3, {
    reduceWindowBytes: CANDIDATE_BYTES * 2,
    mergeFanIn: 2,
  });

  expect(bounded.plan.summary).toEqual({
    ...whole.plan.summary,
    partitions: 3,
  });

  expect(bounded.selected.map((row) => row.id)).toEqual(whole.selected.map((row) => row.id));
  expect(whole.plan.summary.uniqueTexts).toBe(450);
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
