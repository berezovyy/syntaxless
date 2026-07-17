import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";

import { BACKWARD_SEED } from "../src/backward";
import type { CorpusRecord } from "../src/corpus/merge";
import {
  type CompleteSourceCandidate,
  completeSourceCandidates,
} from "../src/corpus/production/complete-sources";
import {
  MANUAL_SOURCE_VERSION,
  manualCorpusRecord,
  type ManualSource,
  manualSourceManifestSchema,
  streamManualSource,
  writeManualSource,
} from "../src/corpus/production/manual-source";
import { ProtectedText } from "../src/corpus/production/protected-text";
import { publishDirectory } from "../src/corpus/production/publication";
import { iterateMix } from "../src/mix/draw";
import { tempDir } from "./helpers/temp-dir";

const sha256 = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");

async function collectManualCandidates(
  manualSource: ManualSource,
): Promise<CompleteSourceCandidate[]> {
  const directory = tempDir("complete-manual-inputs-");
  const rows: CompleteSourceCandidate[] = [];

  for await (const row of completeSourceCandidates({
    publicCorpusPath: join(directory, "unused-public.jsonl"),
    verifiedLlmPath: join(directory, "unused-llm.jsonl"),
    acceptedRealDirectory: join(directory, "unused-real"),
    sources: ["manual"],
    protectedText: new ProtectedText([]),
    manualSource,
  })) {
    rows.push(row);
  }

  return rows;
}

test("manual source writes one verified gzip and replays exact live records", async () => {
  const directory = tempDir("manual-source-");

  const source = {
    directory: join(directory, "manual"),
    records: 320,
    seed: BACKWARD_SEED,
    kernelsPerKind: 2,
    spellingNoise: 0.25,
  };

  const manifest = await publishDirectory(source.directory, 1, async (staging) => {
    const result = await writeManualSource({ ...source, directory: staging });

    expect(existsSync(source.directory)).toBe(false);
    expect(existsSync(join(staging, "corpus.jsonl.gz"))).toBe(true);
    expect(existsSync(join(staging, "manifest.json"))).toBe(true);

    return result;
  });

  const corpusPath = join(source.directory, "corpus.jsonl.gz");
  const manifestPath = join(source.directory, "manifest.json");

  expect(manualSourceManifestSchema.parse(manifest)).toEqual(manifest);

  expect(manifest).toMatchObject({
    status: "complete-manual-source",
    sourceVersion: MANUAL_SOURCE_VERSION,
    records: 320,
    seed: BACKWARD_SEED,
    kernelsPerKind: 2,
    spellingNoise: 0.25,
  });

  const compressed = readFileSync(corpusPath);
  const uncompressed = gunzipSync(compressed);

  expect(manifest.compressed).toEqual({
    bytes: compressed.length,
    sha256: sha256(compressed),
  });

  expect(manifest.uncompressed).toEqual({
    bytes: uncompressed.length,
    sha256: sha256(uncompressed),
  });

  expect(readFileSync(manifestPath, "utf8").trim().split("\n")).toHaveLength(1);
  expect(uncompressed.toString("utf8").trimEnd().split("\n")).toHaveLength(320);

  const streamed: CorpusRecord[] = [];

  for await (const record of streamManualSource(source)) streamed.push(record);
  expect(streamed).toHaveLength(320);

  expect(streamed[0]).toMatchObject({
    id: `${MANUAL_SOURCE_VERSION}/manual/0`,
    source: "rule",
    provenance: { sourceId: expect.any(String) },
  });

  const live = Array.from(
    iterateMix({
      count: 320,
      seed: BACKWARD_SEED,
      perKindKernels: 2,
      easyNegatives: false,
      spellingNoise: 0.25,
    }),
    (example, ordinal) => manualCorpusRecord(example, ordinal),
  );

  const replayed = await collectManualCandidates(source);

  expect(replayed.map((row) => row.record)).toEqual(live);
  expect(live.every((record) => record.intent !== null)).toBe(true);
  const noisy = live.filter((record) => record.template?.includes("/noise-"));

  expect(noisy.length).toBeGreaterThan(40);
  expect(noisy.length).toBeLessThan(120);
});

test("manual source refuses mismatched configuration and cleans failed publication", async () => {
  const directory = tempDir("manual-source-failure-");

  const source = {
    directory: join(directory, "manual"),
    records: 320,
    seed: BACKWARD_SEED,
    kernelsPerKind: 2,
    spellingNoise: 0.25,
  };

  await expect(
    publishDirectory(source.directory, 1, (staging) =>
      writeManualSource({
        ...source,
        directory: staging,
        records: 1,
        kernelsPerKind: 1,
      }),
    ),
  ).rejects.toThrow("need enough rows to hit all bands");

  expect(existsSync(source.directory)).toBe(false);
  expect(readdirSync(directory)).toEqual([".manual.lock.sqlite"]);

  await publishDirectory(source.directory, 1, (staging) =>
    writeManualSource({ ...source, directory: staging }),
  );

  const mismatched = async (): Promise<void> => {
    for await (const _ of streamManualSource({
      ...source,
      records: 321,
    })) {
      // Configuration validation runs before the first record.
    }
  };

  await expect(mismatched()).rejects.toThrow("configuration mismatch");
  const corpusPath = join(source.directory, "corpus.jsonl.gz");
  const prior = readFileSync(corpusPath);

  await expect(
    publishDirectory(source.directory, 1, () => {
      throw new Error("Existing published source must not invoke its writer");
    }),
  ).rejects.toThrow("Refusing to overwrite");

  expect(readFileSync(corpusPath)).toEqual(prior);
  expect(readdirSync(directory).sort()).toEqual([".manual.lock.sqlite", "manual"]);
});
