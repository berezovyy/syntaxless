/** Build and replay the canonical manual workflow as one reusable gzip artifact. */
import { createHash } from "node:crypto";
import { once } from "node:events";
import {
  closeSync,
  createWriteStream,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { createInterface } from "node:readline";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { createGzip } from "node:zlib";

import { z } from "zod";

import { BACKWARD_SEED, generatorSha256 } from "../../backward";
import { fingerprintExample, sha256Hex } from "../../fingerprint";
import type { KindExample } from "../../kinds";
import { iterateMix } from "../../mix/draw";
import { normalizeAnnotatedText } from "../../sentence";
import { jsonlByteChunks } from "../jsonl-io";
import { type CorpusRecord, corpusRecordSchema } from "../merge";

export const MANUAL_SOURCE_VERSION = "complete-sources-v1";
export const MANUAL_SOURCE_RECORDS = 20_000_000;
export const MANUAL_SOURCE_KERNELS_PER_KIND = 2_048;

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
    generatorSha256: z.string().regex(/^[0-9a-f]{64}$/),
    implementationSha256: z.string().regex(/^[0-9a-f]{64}$/),
    uncompressed: hashSchema,
    compressed: hashSchema,
  })
  .strict();

export type ManualSourceManifest = z.infer<typeof manualSourceManifestSchema>;

export interface BuildManualSourceOptions {
  outputPath: string;
  manifestPath: string;
  records?: number;
  seed?: number;
  kernelsPerKind?: number;
  progress?: (records: number) => void;
}

export interface StreamManualSourceOptions {
  corpusPath: string;
  manifestPath: string;
  records?: number;
  seed?: number;
  kernelsPerKind?: number;
  progress?: (records: number) => void;
}

function sha256File(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

/** Hash the code that turns one manual KindExample into a CorpusRecord. */
export function manualSourceImplementationSha256(): string {
  return sha256File(fileURLToPath(import.meta.url));
}

export function defaultManualManifestPath(outputPath: string): string {
  if (!outputPath.endsWith(".jsonl.gz")) {
    throw new Error(`Manual corpus path must end in .jsonl.gz: ${outputPath}`);
  }

  return `${outputPath.slice(0, -".jsonl.gz".length)}.manifest.json`;
}

/** Canonical record conversion shared by live traversal and materialization. */
export function manualCorpusRecord(example: KindExample, ordinal: number): CorpusRecord {
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
  };
}

function syncFile(path: string): void {
  const fd = openSync(path, "r");

  try {
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

function syncDirectory(path: string): void {
  const fd = openSync(path, "r");

  try {
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

function temporaryPath(finalPath: string): string {
  return join(dirname(finalPath), `.${basename(finalPath)}.${process.pid}.${Date.now()}.tmp`);
}

/**
 * Materialize the manual workflow once. The manifest is renamed last and is
 * the publication marker; a failed run removes only temporary/owned outputs.
 */
export async function buildManualSource(
  options: BuildManualSourceOptions,
): Promise<ManualSourceManifest> {
  const outputPath = resolve(options.outputPath);
  const manifestPath = resolve(options.manifestPath);

  if (!outputPath.endsWith(".jsonl.gz")) {
    throw new Error(`Manual corpus path must end in .jsonl.gz: ${outputPath}`);
  }

  if (outputPath === manifestPath) throw new Error("Manual corpus and manifest paths must differ");

  if (existsSync(outputPath) || existsSync(manifestPath)) {
    throw new Error("Refusing to overwrite an existing manual corpus or manifest");
  }

  if (dirname(outputPath) !== dirname(manifestPath)) {
    throw new Error("Manual corpus and manifest must share one publication directory");
  }

  mkdirSync(dirname(outputPath), { recursive: true });

  const records = options.records ?? MANUAL_SOURCE_RECORDS;
  const seed = options.seed ?? BACKWARD_SEED;
  const kernelsPerKind = options.kernelsPerKind ?? MANUAL_SOURCE_KERNELS_PER_KIND;

  if (!Number.isInteger(records) || records < 1) {
    throw new Error(`Manual source records must be a positive integer: ${records}`);
  }

  if (!Number.isInteger(seed)) throw new Error(`Manual source seed must be an integer: ${seed}`);

  if (!Number.isInteger(kernelsPerKind) || kernelsPerKind < 1) {
    throw new Error(`Manual source kernelsPerKind must be a positive integer: ${kernelsPerKind}`);
  }

  const temporaryCorpus = temporaryPath(outputPath);
  const temporaryManifest = temporaryPath(manifestPath);
  const uncompressedHash = createHash("sha256");
  const compressedHash = createHash("sha256");
  let uncompressedBytes = 0;
  let compressedBytes = 0;
  let emitted = 0;
  let corpusPublished = false;
  let manifestPublished = false;
  const gzip = createGzip({ level: 1 });

  gzip.on("data", (chunk: Buffer) => {
    compressedHash.update(chunk);
    compressedBytes += chunk.length;
  });

  const output = createWriteStream(temporaryCorpus, { flags: "wx" });
  const writing = pipeline(gzip, output);
  let pendingLines: string[] = [];
  let pendingCharacters = 0;

  const flushLines = async (): Promise<void> => {
    if (pendingLines.length === 0) return;
    const chunk = Buffer.from(pendingLines.join(""), "utf8");

    pendingLines = [];
    pendingCharacters = 0;
    uncompressedHash.update(chunk);
    uncompressedBytes += chunk.length;
    if (!gzip.write(chunk)) await once(gzip, "drain");
  };

  try {
    for (const example of iterateMix({
      count: records,
      seed,
      perKindKernels: kernelsPerKind,
    })) {
      const line = `${JSON.stringify(manualCorpusRecord(example, emitted))}\n`;

      pendingLines.push(line);
      pendingCharacters += line.length;
      emitted++;
      // Batch writes to avoid 20 million Buffer allocations and zlib calls.
      // The manifest commits both the unchanged JSONL bytes and deterministic
      // gzip bytes produced by this batching policy.
      if (pendingCharacters >= 1024 * 1024) await flushLines();
      if (emitted % 100_000 === 0) options.progress?.(emitted);
    }

    await flushLines();
    gzip.end();
    await writing;
    if (emitted !== records) throw new Error(`Manual source emitted ${emitted}/${records} records`);
    syncFile(temporaryCorpus);
    const stat = statSync(temporaryCorpus);

    if (stat.size !== compressedBytes) {
      throw new Error(
        `Manual compressed byte count mismatch: stream ${compressedBytes}, file ${stat.size}`,
      );
    }

    const manifest: ManualSourceManifest = {
      version: 1,
      status: "complete-manual-source",
      sourceVersion: MANUAL_SOURCE_VERSION,
      records,
      seed,
      kernelsPerKind,
      generatorSha256: generatorSha256(),
      implementationSha256: manualSourceImplementationSha256(),
      uncompressed: {
        bytes: uncompressedBytes,
        sha256: uncompressedHash.digest("hex"),
      },
      compressed: {
        bytes: compressedBytes,
        sha256: compressedHash.digest("hex"),
      },
    };

    writeFileSync(temporaryManifest, `${JSON.stringify(manifest)}\n`, { flag: "wx" });
    syncFile(temporaryManifest);
    renameSync(temporaryCorpus, outputPath);
    corpusPublished = true;
    renameSync(temporaryManifest, manifestPath);
    manifestPublished = true;
    syncDirectory(dirname(outputPath));

    return manifest;
  } catch (error) {
    gzip.destroy();
    await writing.catch(() => undefined);
    rmSync(temporaryCorpus, { force: true });
    rmSync(temporaryManifest, { force: true });
    if (corpusPublished && !manifestPublished) rmSync(outputPath, { force: true });
    throw error;
  }
}

/** Validate and stream every exact CorpusRecord from a manual gzip artifact. */
export async function* streamManualSource(
  options: StreamManualSourceOptions,
): AsyncGenerator<CorpusRecord> {
  const manifest = manualSourceManifestSchema.parse(
    JSON.parse(readFileSync(options.manifestPath, "utf8")),
  );

  const expectedRecords = options.records ?? MANUAL_SOURCE_RECORDS;
  const expectedSeed = options.seed ?? BACKWARD_SEED;
  const expectedKernels = options.kernelsPerKind ?? MANUAL_SOURCE_KERNELS_PER_KIND;

  if (
    manifest.records !== expectedRecords ||
    manifest.seed !== expectedSeed ||
    manifest.kernelsPerKind !== expectedKernels
  ) {
    throw new Error(
      `Manual source configuration mismatch: expected ${expectedRecords}/${expectedSeed}/${expectedKernels}, ` +
        `got ${manifest.records}/${manifest.seed}/${manifest.kernelsPerKind}`,
    );
  }

  if (
    manifest.generatorSha256 !== generatorSha256() ||
    manifest.implementationSha256 !== manualSourceImplementationSha256()
  ) {
    throw new Error("Manual source generator or implementation hash changed");
  }

  const stat = statSync(options.corpusPath);

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

  const chunks = jsonlByteChunks(options.corpusPath, (chunk) => {
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

  const lines = createInterface({
    input: Readable.from(observed),
    crlfDelay: Number.POSITIVE_INFINITY,
  });

  for await (const line of lines) {
    if (!line) throw new Error(`Manual source contains a blank record at line ${records + 1}`);
    records++;
    let record: CorpusRecord;

    try {
      record = corpusRecordSchema.parse(JSON.parse(line)) as CorpusRecord;
    } catch (error) {
      throw new Error(`Invalid manual source record ${records}`, { cause: error });
    }

    options.progress?.(records);
    yield record;
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

if (import.meta.main) {
  const output = process.argv[2];

  if (!output || process.argv.length !== 3) {
    throw new Error("Usage: manual-source.ts NEW_OUTPUT.jsonl.gz");
  }

  const outputPath = resolve(output);
  const manifestPath = defaultManualManifestPath(outputPath);

  const manifest = await buildManualSource({
    outputPath,
    manifestPath,
    progress: (records) => console.error(`manual source: ${records.toLocaleString()} records`),
  });

  console.log(JSON.stringify({ outputPath, manifestPath, ...manifest }, null, 2));
}
