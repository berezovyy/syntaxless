/** Write and verify the two files of the coordinator-owned synthetic source. */
import { createHash } from "node:crypto";
import {
  closeSync,
  existsSync,
  fsyncSync,
  openSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { z } from "zod";

import { generatorSha256 } from "../../backward";
import { fingerprintExample, sha256Hex } from "../../fingerprint";
import type { KindExample } from "../../kinds";
import { iterateMix } from "../../mix/draw";
import { normalizeAnnotatedText } from "../../sentence";
import { jsonlByteChunks, jsonlReader, writeGzip } from "../jsonl-io";
import { type CorpusRecord, corpusRecordSchema } from "../merge";

/** v2: schedule rows only (R/H) and an explicit spelling-noise share. */
export const MANUAL_SOURCE_VERSION = "complete-sources-v2";

const hashSchema = z
  .object({
    bytes: z.number().int().nonnegative(),
    sha256: z.string().regex(/^[0-9a-f]{64}$/),
  })
  .strict();

export const manualSourceManifestSchema = z
  .object({
    version: z.literal(1),
    status: z.literal("complete-manual-source"),
    sourceVersion: z.literal(MANUAL_SOURCE_VERSION),
    records: z.number().int().positive(),
    seed: z.number().int(),
    kernelsPerKind: z.number().int().positive(),
    spellingNoise: z.number().min(0).max(1),
    generatorSha256: z.string().regex(/^[0-9a-f]{64}$/),
    implementationSha256: z.string().regex(/^[0-9a-f]{64}$/),
    uncompressed: hashSchema,
    compressed: hashSchema,
  })
  .strict();

export type ManualSourceManifest = z.infer<typeof manualSourceManifestSchema>;

/** The canonical artifact directory and its required generation recipe. */
export interface ManualSource {
  directory: string;
  records: number;
  seed: number;
  kernelsPerKind: number;
  /** Share of rows that carry one meaning-preserving typo. */
  spellingNoise: number;
}

/** Hash the code that turns one manual KindExample into a CorpusRecord. */
export function manualSourceImplementationSha256(): string {
  return createHash("sha256")
    .update(readFileSync(fileURLToPath(import.meta.url)))
    .digest("hex");
}

/** Canonical conversion from a generator example to a source record. */
export function manualCorpusRecord(example: KindExample, ordinal: number) {
  const annotated = normalizeAnnotatedText(example.example.text, example.example.spans);
  const rendered = { ...example.example, ...annotated };

  const fingerprint =
    example.intent === null
      ? sha256Hex(`negative\0${rendered.text}`)
      : fingerprintExample(rendered);

  const easy = example.intent === null;

  return {
    id: `${MANUAL_SOURCE_VERSION}/manual/${ordinal}`,
    split: example.split,
    family: example.family,
    template: example.template,
    fingerprint,
    text: rendered.text,
    intent: example.intent === null ? null : structuredClone(example.intent),
    speechAct: rendered.speechAct,
    spans: rendered.spans,
    source: "rule",
    provenance: {
      authorship: easy ? "public-source" : "programmatic",
      sourceId: easy ? "tatoeba-legacy-manual" : "manual-mix-v1",
      groupId: easy ? `ordinary-${fingerprint}` : `frame-${fingerprint}`,
      constructionId: example.template,
      sourceRecordId: String(ordinal),
      ...(easy
        ? {
            license: "CC-BY-2.0-FR",
            attribution: "Tatoeba contributors; legacy manual-generator source",
          }
        : {}),
    },
  } satisfies CorpusRecord;
}

function syncFile(path: string): void {
  const fd = openSync(path, "r");

  try {
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

/**
 * Write into the staging directory allocated by publishDirectory. That caller
 * owns publication and removes the whole directory if any preparation fails.
 */
export async function writeManualSource(
  source: ManualSource,
  progress?: (records: number) => void,
): Promise<ManualSourceManifest> {
  const { directory, records, seed, kernelsPerKind, spellingNoise } = source;
  const outputPath = join(directory, "corpus.jsonl.gz");
  const manifestPath = join(directory, "manifest.json");

  if (existsSync(outputPath) || existsSync(manifestPath)) {
    throw new Error("Refusing to overwrite an existing manual corpus or manifest");
  }

  if (!Number.isInteger(records) || records < 1) {
    throw new Error(`Manual source records must be a positive integer: ${records}`);
  }

  if (!Number.isInteger(seed)) throw new Error(`Manual source seed must be an integer: ${seed}`);

  if (!Number.isInteger(kernelsPerKind) || kernelsPerKind < 1) {
    throw new Error(`Manual source kernelsPerKind must be a positive integer: ${kernelsPerKind}`);
  }

  if (!(spellingNoise >= 0 && spellingNoise <= 1)) {
    throw new Error(`Manual source spellingNoise must be within [0, 1]: ${spellingNoise}`);
  }

  const uncompressedHash = createHash("sha256");
  let uncompressedBytes = 0;
  let emitted = 0;
  // The generator's uniqueness index lives beside the output on the data disk
  // and is deleted with it; see mix/disk-hash-set.ts.
  const dedupDirectory = join(directory, ".mix-dedup");

  async function* chunks(): AsyncGenerator<Buffer> {
    let pendingLines: string[] = [];
    let pendingCharacters = 0;

    // Negatives come from the public source, which contains every sentence of
    // the mixer's 50k-line list and outranks it in the union.
    for (const example of iterateMix({
      count: records,
      seed,
      perKindKernels: kernelsPerKind,
      easyNegatives: false,
      spellingNoise,
      dedupDirectory,
    })) {
      const line = `${JSON.stringify(manualCorpusRecord(example, emitted))}\n`;

      pendingLines.push(line);
      pendingCharacters += line.length;
      emitted++;

      // Batch writes to avoid one Buffer allocation and zlib call per row.
      // The manifest commits both the unchanged JSONL bytes and deterministic
      // gzip bytes produced by this batching policy.
      if (pendingCharacters >= 1024 * 1024) {
        const chunk = Buffer.from(pendingLines.join(""), "utf8");

        pendingLines = [];
        pendingCharacters = 0;
        uncompressedHash.update(chunk);
        uncompressedBytes += chunk.length;
        yield chunk;
      }

      if (emitted % 100_000 === 0) progress?.(emitted);
    }

    if (pendingLines.length > 0) {
      const chunk = Buffer.from(pendingLines.join(""), "utf8");

      uncompressedHash.update(chunk);
      uncompressedBytes += chunk.length;
      yield chunk;
    }
  }

  let compressed: Awaited<ReturnType<typeof writeGzip>>;

  try {
    compressed = await writeGzip(outputPath, chunks());
  } finally {
    rmSync(dedupDirectory, { recursive: true, force: true });
  }

  if (emitted !== records) throw new Error(`Manual source emitted ${emitted}/${records} records`);
  syncFile(outputPath);
  const stat = statSync(outputPath);

  if (stat.size !== compressed.bytes) {
    throw new Error(
      `Manual compressed byte count mismatch: stream ${compressed.bytes}, file ${stat.size}`,
    );
  }

  const manifest: ManualSourceManifest = {
    version: 1,
    status: "complete-manual-source",
    sourceVersion: MANUAL_SOURCE_VERSION,
    records,
    seed,
    kernelsPerKind,
    spellingNoise,
    generatorSha256: generatorSha256(),
    implementationSha256: manualSourceImplementationSha256(),
    uncompressed: {
      bytes: uncompressedBytes,
      sha256: uncompressedHash.digest("hex"),
    },
    compressed,
  };

  writeFileSync(manifestPath, `${JSON.stringify(manifest)}\n`, { flag: "wx" });
  syncFile(manifestPath);

  return manifest;
}

/** Validate and stream every exact CorpusRecord from a manual gzip artifact. */
export async function* streamManualSource(source: ManualSource): AsyncGenerator<CorpusRecord> {
  const {
    directory,
    records: expectedRecords,
    seed: expectedSeed,
    kernelsPerKind: expectedKernels,
    spellingNoise: expectedNoise,
  } = source;

  const corpusPath = join(directory, "corpus.jsonl.gz");

  const manifest = manualSourceManifestSchema.parse(
    JSON.parse(readFileSync(join(directory, "manifest.json"), "utf8")),
  );

  if (
    manifest.records !== expectedRecords ||
    manifest.seed !== expectedSeed ||
    manifest.kernelsPerKind !== expectedKernels ||
    manifest.spellingNoise !== expectedNoise
  ) {
    throw new Error(
      `Manual source configuration mismatch: expected ${expectedRecords}/${expectedSeed}/${expectedKernels}/${expectedNoise}, ` +
        `got ${manifest.records}/${manifest.seed}/${manifest.kernelsPerKind}/${manifest.spellingNoise}`,
    );
  }

  if (
    manifest.generatorSha256 !== generatorSha256() ||
    manifest.implementationSha256 !== manualSourceImplementationSha256()
  ) {
    throw new Error("Manual source generator or implementation hash changed");
  }

  const stat = statSync(corpusPath);

  if (stat.size !== manifest.compressed.bytes) {
    throw new Error(
      `Manual source compressed size mismatch: ${stat.size} != ${manifest.compressed.bytes}`,
    );
  }

  const compressedHash = createHash("sha256");
  const uncompressedHash = createHash("sha256");
  let compressedBytes = 0;
  let uncompressedBytes = 0;
  let records = 0;

  const chunks = jsonlByteChunks(corpusPath, (chunk) => {
    compressedHash.update(chunk);
    compressedBytes += chunk.length;
  });

  const observed = (async function* (): AsyncGenerator<Buffer> {
    for await (const chunk of chunks) {
      uncompressedHash.update(chunk);
      uncompressedBytes += chunk.length;
      yield chunk;
    }
  })();

  {
    await using reader = jsonlReader(observed);

    for await (const line of reader.lines) {
      if (!line) throw new Error(`Manual source contains a blank record at line ${records + 1}`);
      records++;
      let record: CorpusRecord;

      try {
        record = corpusRecordSchema.parse(JSON.parse(line)) as CorpusRecord;
      } catch (error) {
        throw new Error(`Invalid manual source record ${records}`, { cause: error });
      }

      yield record;
    }
  }

  if (
    records !== manifest.records ||
    compressedBytes !== manifest.compressed.bytes ||
    uncompressedBytes !== manifest.uncompressed.bytes ||
    compressedHash.digest("hex") !== manifest.compressed.sha256 ||
    uncompressedHash.digest("hex") !== manifest.uncompressed.sha256
  ) {
    throw new Error("Manual source count, byte size, or checksum mismatch");
  }
}
