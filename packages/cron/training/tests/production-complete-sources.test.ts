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
import { type ManualSource } from "../src/corpus/production/manual-source";
import { ProtectedText } from "../src/corpus/production/protected-text";
import { sha256Hex } from "../src/fingerprint";
import { Sentence } from "../src/sentence";
import { tempDir } from "./helpers/temp-dir";

interface Paths {
  manualSource?: ManualSource;
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
  })) {
    rows.push(row);
  }

  return rows;
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
