import { expect, test } from "bun:test";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type { CorpusRecord } from "../src/corpus/merge";
import {
  COMPLETE_SOURCE_VERSION,
  type CompleteSourceCandidate,
  completeSourceCandidates,
  type CompleteSourceName,
} from "../src/corpus/production/complete-sources";
import { ProtectedText } from "../src/corpus/production/protected-text";
import { sha256Hex } from "../src/fingerprint";
import { Sentence } from "../src/sentence";
import { tempDir } from "./helpers/temp-dir";

interface Paths {
  publicCorpusPath: string;
  verifiedLlmPath: string;
  acceptedRealDirectory: string;
}

function emptyPaths(prefix: string): Paths {
  const directory = tempDir(prefix);
  const acceptedRealDirectory = join(directory, "real");

  mkdirSync(acceptedRealDirectory);
  const publicCorpusPath = join(directory, "public.jsonl");
  const verifiedLlmPath = join(directory, "verified.jsonl");

  writeFileSync(publicCorpusPath, "");
  writeFileSync(verifiedLlmPath, "");

  return { publicCorpusPath, verifiedLlmPath, acceptedRealDirectory };
}

async function collect(
  paths: Paths,
  sources: readonly CompleteSourceName[],
  protectedText = new ProtectedText([]),
): Promise<CompleteSourceCandidate[]> {
  const rows: CompleteSourceCandidate[] = [];

  for await (const row of completeSourceCandidates({
    ...paths,
    sources,
    protectedText,
    manualRecords: 320,
    manualKernelsPerKind: 2,
  })) {
    rows.push(row);
  }

  return rows;
}

async function first(paths: Paths, source: CompleteSourceName): Promise<CompleteSourceCandidate> {
  const generator = completeSourceCandidates({
    ...paths,
    sources: [source],
    protectedText: new ProtectedText([]),
    manualRecords: 1,
    manualKernelsPerKind: 1,
  });

  const result = await generator.next();

  await generator.return(undefined);
  if (result.done) throw new Error(`No ${source} candidate`);

  return result.value;
}

function publicNegative(id: string, text: string): CorpusRecord {
  return {
    id,
    split: "train",
    family: "negative",
    fingerprint: sha256Hex(`negative\0${text}`),
    text,
    intent: null,
    speechAct: "none",
    spans: [
      {
        start: 0,
        end: text.length,
        label: "O",
        clauseStart: false,
        actionable: false,
      },
    ],
    source: "rule",
    provenance: {
      authorship: "public-source",
      sourceId: "public-test",
      groupId: `group-${id}`,
      constructionId: "ordinary-english",
      sourceRecordId: id,
      license: "CC-BY-2.0-FR",
    },
  };
}

test("manual traversal is deterministic and identifies programmatic versus public rows", async () => {
  const paths = emptyPaths("complete-sources-manual-");
  const left = await collect(paths, ["manual"]);
  const right = await collect(paths, ["manual"]);

  expect(left).toEqual(right);
  expect(left).toHaveLength(320);
  expect(left.every((row) => row.source === "manual")).toBe(true);

  expect(
    left.every((row) =>
      ["manual-mix-v1", "tatoeba-legacy-manual"].includes(row.record.provenance?.sourceId ?? ""),
    ),
  ).toBe(true);

  expect(new Set(left.map((row) => row.record.provenance!.sourceId))).toEqual(
    new Set(["manual-mix-v1", "tatoeba-legacy-manual"]),
  );

  expect(left.every((row) => row.union.priority === 40)).toBe(true);
});

test("authored and systematic traversals expose canonical provenance", async () => {
  const paths = emptyPaths("complete-sources-generated-");
  const authored = await first(paths, "authored");

  expect(authored.record.provenance).toMatchObject({
    authorship: "ai-authored",
    sourceId: "whole-utterances-v1",
  });

  expect(authored.union.priority).toBe(20);

  const systematic = await first(paths, "systematic");

  expect(systematic.record.provenance).toMatchObject({
    authorship: "programmatic",
    sourceId: "systematic-catalog-v1",
  });

  expect(systematic.record.split).toBe("train");
  expect(systematic.union.priority).toBe(50);
});

test("public and verified LLM inputs are strict, protected, attributed, and ordered deterministically", async () => {
  const paths = emptyPaths("complete-sources-files-");
  const blocked = publicNegative("blocked", "this exact sentence is protected");
  const allowed = publicNegative("allowed", "the kettle is blue");

  writeFileSync(paths.publicCorpusPath, `${JSON.stringify(blocked)}\n${JSON.stringify(allowed)}\n`);

  writeFileSync(
    paths.verifiedLlmPath,
    `${JSON.stringify({
      text: "please print the report",
      intent: null,
      output: "negative",
      spans: [
        {
          start: 0,
          end: 23,
          label: "O",
          clauseStart: false,
          actionable: false,
        },
      ],
      fingerprint: sha256Hex("verified fixture"),
      source: "llm",
      angleId: "chores/negative/chat",
      model: "fixture-model",
      promptHash: "c".repeat(64),
      runId: "fixture-run",
    })}\n`,
  );

  const guard = new ProtectedText([blocked.text]);
  const forward = await collect(paths, ["verified-llm", "public"], guard);
  const reverse = await collect(paths, ["public", "verified-llm"], guard);

  expect(forward).toEqual(reverse);
  expect(forward.map((row) => row.source)).toEqual(["public", "verified-llm"]);
  expect(forward.map((row) => row.record.text)).toEqual([allowed.text, "please print the report"]);
  expect(forward[0]!.record.provenance).toEqual(allowed.provenance);

  expect(forward[1]!.record.provenance).toMatchObject({
    authorship: "ai-authored",
    sourceId: "verified-llm-v1",
    attribution: "fixture-model; run fixture-run",
  });

  writeFileSync(paths.publicCorpusPath, `${JSON.stringify({ ...allowed, unexpected: true })}\n`);
  await expect(collect(paths, ["public"])).rejects.toThrow("Invalid public corpus record 1");
});

function acceptedHumanInput(options: { consent: boolean }) {
  const sentence = new Sentence();

  sentence.add("every", "RECUR", " ", true);
  sentence.clause();
  sentence.add("day", "UNIT_DAY", " ", true);

  return {
    id: "consented-1",
    text: sentence.text,
    intent: { frequency: { unit: "day", interval: 1 } },
    spans: sentence.spans,
    agree: 3,
    exec: true,
    roundtrip: true,
    family: "terse",
    speechAct: "request",
    ...(options.consent
      ? {
          trainingConsent: {
            granted: true,
            scope: "model-training",
            recordedBy: "fixture-consent-ledger",
          },
        }
      : {}),
    provenance: {
      authorship: "human-authored",
      sourceId: "consented-human-fixture",
      groupId: "consented-1",
      constructionId: "direct-request",
      sourceRecordId: "consented-1",
    },
  };
}

test("accepted human rows require explicit provenance and model-training consent", async () => {
  const missing = emptyPaths("complete-sources-human-missing-");

  writeFileSync(
    join(missing.acceptedRealDirectory, "accepted.jsonl"),
    `${JSON.stringify(acceptedHumanInput({ consent: false }))}\n`,
  );

  await expect(collect(missing, ["accepted-human"])).rejects.toThrow(
    "lacks explicit human provenance and model-training consent",
  );

  const accepted = emptyPaths("complete-sources-human-accepted-");

  writeFileSync(
    join(accepted.acceptedRealDirectory, "accepted.jsonl"),
    `${JSON.stringify(acceptedHumanInput({ consent: true }))}\n`,
  );

  const rows = await collect(accepted, ["accepted-human"]);

  expect(rows).toHaveLength(1);

  expect(rows[0]).toMatchObject({
    source: "accepted-human",
    union: { priority: 0 },
    record: {
      id: `${COMPLETE_SOURCE_VERSION}/accepted-human/consented-1`,
      split: "train",
      family: "terse",
      speechAct: "request",
      source: "llm",
      template: "real/accepted",
      provenance: {
        authorship: "human-authored",
        sourceId: "consented-human-fixture",
      },
    },
  });
});
