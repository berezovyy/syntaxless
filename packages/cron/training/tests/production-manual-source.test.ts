import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";

import { BACKWARD_SEED } from "../src/backward";
import type { CorpusRecord } from "../src/corpus/merge";
import {
  type CompleteSourceCandidate,
  completeSourceCandidates,
} from "../src/corpus/production/complete-sources";
import {
  buildManualSource,
  MANUAL_SOURCE_VERSION,
  manualSourceManifestSchema,
  streamManualSource,
} from "../src/corpus/production/manual-source";
import { ProtectedText } from "../src/corpus/production/protected-text";
import { tempDir } from "./helpers/temp-dir";

const sha256 = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");

async function collectManualCandidates(options: {
  corpusPath?: string;
  manifestPath?: string;
}): Promise<CompleteSourceCandidate[]> {
  const directory = tempDir("complete-manual-inputs-");
  const rows: CompleteSourceCandidate[] = [];

  for await (const row of completeSourceCandidates({
    publicCorpusPath: join(directory, "unused-public.jsonl"),
    verifiedLlmPath: join(directory, "unused-llm.jsonl"),
    acceptedRealDirectory: join(directory, "unused-real"),
    sources: ["manual"],
    protectedText: new ProtectedText([]),
    manualRecords: 320,
    manualSeed: BACKWARD_SEED,
    manualKernelsPerKind: 2,
    ...(options.corpusPath === undefined
      ? {}
      : {
          manualCorpusPath: options.corpusPath,
          manualManifestPath: options.manifestPath!,
        }),
  })) {
    rows.push(row);
  }

  return rows;
}

test("manual source writes one verified gzip and replays exact live records", async () => {
  const directory = tempDir("manual-source-");
  const corpusPath = join(directory, "manual.jsonl.gz");
  const manifestPath = join(directory, "manual.manifest.json");

  const manifest = await buildManualSource({
    outputPath: corpusPath,
    manifestPath,
    records: 320,
    seed: BACKWARD_SEED,
    kernelsPerKind: 2,
  });

  expect(manualSourceManifestSchema.parse(manifest)).toEqual(manifest);

  expect(manifest).toMatchObject({
    status: "complete-manual-source",
    sourceVersion: MANUAL_SOURCE_VERSION,
    records: 320,
    seed: BACKWARD_SEED,
    kernelsPerKind: 2,
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

  for await (const record of streamManualSource({
    corpusPath,
    manifestPath,
    records: 320,
    seed: BACKWARD_SEED,
    kernelsPerKind: 2,
  })) {
    streamed.push(record);
  }

  expect(streamed).toHaveLength(320);

  expect(streamed[0]).toMatchObject({
    id: `${MANUAL_SOURCE_VERSION}/manual/0`,
    source: "rule",
    provenance: { sourceId: expect.any(String) },
  });

  const live = await collectManualCandidates({});
  const replayed = await collectManualCandidates({ corpusPath, manifestPath });

  expect(replayed).toEqual(live);
});

test("manual source refuses mismatched configuration and cleans failed publication", async () => {
  const directory = tempDir("manual-source-failure-");
  const corpusPath = join(directory, "manual.jsonl.gz");
  const manifestPath = join(directory, "manual.manifest.json");

  await expect(
    buildManualSource({
      outputPath: corpusPath,
      manifestPath,
      records: 1,
      seed: BACKWARD_SEED,
      kernelsPerKind: 1,
    }),
  ).rejects.toThrow("need enough rows to hit all bands");

  expect(existsSync(corpusPath)).toBe(false);
  expect(existsSync(manifestPath)).toBe(false);

  await buildManualSource({
    outputPath: corpusPath,
    manifestPath,
    records: 320,
    seed: BACKWARD_SEED,
    kernelsPerKind: 2,
  });

  const mismatched = async (): Promise<void> => {
    for await (const _ of streamManualSource({
      corpusPath,
      manifestPath,
      records: 321,
      seed: BACKWARD_SEED,
      kernelsPerKind: 2,
    })) {
      // Configuration validation runs before the first record.
    }
  };

  await expect(mismatched()).rejects.toThrow("configuration mismatch");

  await expect(
    buildManualSource({
      outputPath: corpusPath,
      manifestPath,
      records: 320,
      seed: BACKWARD_SEED,
      kernelsPerKind: 2,
    }),
  ).rejects.toThrow("Refusing to overwrite");
});

test("manual artifact options must be supplied as a pair", async () => {
  const directory = tempDir("manual-source-path-pair-");

  const run = async (): Promise<void> => {
    for await (const _ of completeSourceCandidates({
      publicCorpusPath: join(directory, "unused-public.jsonl"),
      verifiedLlmPath: join(directory, "unused-llm.jsonl"),
      acceptedRealDirectory: join(directory, "unused-real"),
      sources: ["manual"],
      protectedText: new ProtectedText([]),
      manualCorpusPath: join(directory, "only.jsonl.gz"),
    })) {
      // The missing manifest path is rejected before opening either path.
    }
  };

  await expect(run()).rejects.toThrow(
    "manualCorpusPath and manualManifestPath must be supplied together",
  );
});
