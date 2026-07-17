/** Resumable, disk-partitioned coordinator for the budgeted all-unique corpus. */
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  closeSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { z } from "zod";

import { SHARD_FORMAT } from "../../featurize";
import { featurizeKindsStreaming } from "../../featurize-kinds";
import { classifyPopulation } from "../../mix/quotas";
import { loadTatoebaSentences, TATOEBA_MANIFEST, TATOEBA_SENTENCES } from "../../mix/tatoeba";
import { humanEvalTexts } from "../../slots/leak";
import { writeGzip } from "../jsonl-io";
import { type CorpusRecord, SPLITS } from "../merge";
import {
  COMPLETE_SOURCE_VERSION,
  completeSourceCandidates,
  type CompleteSourceName,
  type CompleteSourceOptions,
} from "./complete-sources";
import { CONSTRUCTIONS_PATH } from "./constructions";
import { CORPUS_UNION_PLAN_FILES, CorpusUnionPlan, StreamingCorpusUnion } from "./global-union";
import { HashOrderedSpool } from "./hash-order";
import { manualSourceManifestSchema, writeManualSource } from "./manual-source";
import { loadProtectedText } from "./protected-text";
import {
  acquirePreparationLock,
  diskPreflight,
  hashArtifact,
  publishDirectory,
  replaceJson,
} from "./publication";
import { sourceExpectations } from "./source-expectations";
import { corpusCodeHashes, fileHash } from "./source-hashes";

const training = resolve(fileURLToPath(new URL("../../../", import.meta.url)));
const repo = resolve(training, "../../..");
const INITIAL_REQUIRED_BYTES = 24 * 1024 ** 3;

const PRODUCTION_CONFIG_PATH = new URL(
  "../../../data/production/production-complete-v1.json",
  import.meta.url,
);

type ProductionPhase = "plan" | "emit" | "expectations" | "shards" | "verify" | "publish";

const configSchema = z
  .object({
    version: z.literal(1),
    corpusVersion: z.literal("production-complete-v1"),
    policy: z.literal("budgeted-runtime-normalized-texts"),
    manual: z
      .object({
        records: z.number().int().positive(),
        seed: z.number().int(),
        kernelsPerKind: z.number().int().positive(),
        spellingNoise: z.number().min(0).max(1),
      })
      .strict(),
    sampling: z
      .object({
        authored: z.number().gt(0).max(1),
        systematic: z.number().gt(0).max(1),
        public: z.number().gt(0).max(1),
      })
      .strict(),
    union: z.object({ partitions: z.number().int().min(1).max(256) }).strict(),
    inputs: z
      .object({
        publicCorpus: z.string().min(1),
        publicManifest: z.string().min(1),
        publicManifestSha256: z.string().regex(/^[a-f0-9]{64}$/),
        verifiedLlm: z.string().min(1),
        acceptedRealDirectory: z.string().min(1),
      })
      .strict(),
  })
  .strict();

type Config = z.infer<typeof configSchema>;
type Artifact = { bytes: number; sha256: string };
type Expectations = Awaited<ReturnType<typeof sourceExpectations>>;
type Population = "R" | "H" | "E";

interface BuildState {
  version: 1;
  corpusVersion: string;
  output: string;
  config: Config;
  configSha256: string;
  inputs: Record<string, string>;
  implementation: Record<string, string>;
  toolchain: Record<string, string | undefined>;
  plan?: {
    summary: {
      candidates: number;
      uniqueTexts: number;
      duplicates: number;
      partitions: number;
      ingestSha256: string;
      selectedReferencesSha256: string;
    };
    candidatesBySource: Record<CompleteSourceName, number>;
  };
  emit?: {
    corpus: Artifact;
    emittedBySource: Record<CompleteSourceName, number>;
    splits: Record<(typeof SPLITS)[number], number>;
    populations: Record<Population, number>;
    authorship: Record<string, number>;
  };
  expectations?: { artifact: Artifact; expectedShardBytes: number };
  shards?: { artifacts: Record<string, Artifact> };
  verify?: { artifact: Artifact };
}

const SOURCE_NAMES: readonly CompleteSourceName[] = [
  "manual",
  "authored",
  "systematic",
  "public",
  "verified-llm",
  "accepted-human",
];

function emptySourceCounts(): Record<CompleteSourceName, number> {
  return Object.fromEntries(SOURCE_NAMES.map((source) => [source, 0])) as Record<
    CompleteSourceName,
    number
  >;
}

function resolveTrainingPath(path: string): string {
  const resolved = resolve(training, path);

  if (resolved !== training && !resolved.startsWith(`${training}/`)) {
    throw new Error(`Production input is outside the training package: ${path}`);
  }

  return resolved;
}

function pathsFor(outputValue: string) {
  const output = resolve(outputValue);
  const parent = dirname(output);
  const name = basename(output);

  return {
    output,
    parent,
    work: join(parent, `.${name}.work`),
    state: join(parent, `.${name}.state.json`),
  };
}

function writeJsonExclusive(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" });
}

function readState(path: string): BuildState {
  return JSON.parse(readFileSync(path, "utf8")) as BuildState;
}

async function artifact(path: string): Promise<Artifact> {
  return { bytes: statSync(path).size, sha256: await hashArtifact(path) };
}

function hashDirectoryFiles(directory: string): Record<string, string> {
  const result: Record<string, string> = {};

  if (!existsSync(directory)) return result;

  const walk = (current: string): void => {
    for (const entry of readdirSync(current, { withFileTypes: true }).sort((a, b) =>
      a.name.localeCompare(b.name, "en"),
    )) {
      const path = join(current, entry.name);

      if (entry.isDirectory()) walk(path);
      else if (entry.isFile()) result[relative(repo, path)] = fileHash(path);
    }
  };

  walk(directory);

  return result;
}

/** Generated inputs belong to this workflow, keyed by their actual producer. */
function manualSourcePaths(config: Config, protectedText: ReturnType<typeof loadProtectedText>) {
  const inputs = Object.fromEntries(
    [TATOEBA_SENTENCES, TATOEBA_MANIFEST, ...protectedText.sources.map((source) => source.file)]
      .sort()
      .map((path) => [relative(repo, path), fileHash(path)]),
  );

  const identity = createHash("sha256")
    .update(
      JSON.stringify({
        manual: config.manual,
        implementation: {
          ...corpusCodeHashes(new URL("./manual-source.ts", import.meta.url).href),
          ...corpusCodeHashes(new URL("./publication.ts", import.meta.url).href),
        },
        inputs,
      }),
    )
    .digest("hex");

  const directory = join(training, "data/synth/production-sources", `manual-${identity}`);

  return {
    directory,
    corpus: join(directory, "corpus.jsonl.gz"),
    manifest: join(directory, "manifest.json"),
    inputs,
  };
}

async function prepareManualSource(
  config: Config,
  progress: (phase: string, count?: number) => void,
): Promise<void> {
  const paths = manualSourcePaths(config, loadProtectedText());

  loadTatoebaSentences({ blocked: new Set(humanEvalTexts()) });
  if (existsSync(paths.directory)) return;
  mkdirSync(dirname(paths.directory), { recursive: true });
  progress("manual-source");

  await publishDirectory(paths.directory, INITIAL_REQUIRED_BYTES, async (staging) => {
    await writeManualSource({ directory: staging, ...config.manual }, (count) =>
      progress("manual-source", count),
    );

    if (manualSourcePaths(config, loadProtectedText()).directory !== paths.directory) {
      throw new Error("Manual source inputs or implementation changed during preparation");
    }
  });
}

function sourceOptions(
  config: Config,
  protectedText: ReturnType<typeof loadProtectedText>,
  progress: (phase: string, count?: number) => void,
): CompleteSourceOptions {
  const manual = manualSourcePaths(config, protectedText);

  return {
    manualSource: {
      directory: manual.directory,
      ...config.manual,
    },
    publicCorpusPath: resolveTrainingPath(config.inputs.publicCorpus),
    verifiedLlmPath: resolveTrainingPath(config.inputs.verifiedLlm),
    acceptedRealDirectory: resolveTrainingPath(config.inputs.acceptedRealDirectory),
    protectedText,
    sampling: config.sampling,
    progress: (source, count) => progress(`source/${source}`, count),
  };
}

async function publicSourceHashes(config: Config): Promise<Record<string, string>> {
  const publicCorpus = resolveTrainingPath(config.inputs.publicCorpus);
  const publicManifest = resolveTrainingPath(config.inputs.publicManifest);

  if (fileHash(publicManifest) !== config.inputs.publicManifestSha256) {
    throw new Error("Pinned public-source manifest checksum mismatch");
  }

  const publicData = JSON.parse(readFileSync(publicManifest, "utf8")) as {
    artifacts?: Record<string, Artifact>;
  };

  const expectedPublic = publicData.artifacts?.["corpus.jsonl"];

  if (!expectedPublic) throw new Error("Public-source manifest lacks corpus.jsonl");
  const actualPublic = await artifact(publicCorpus);

  if (
    actualPublic.bytes !== expectedPublic.bytes ||
    actualPublic.sha256 !== expectedPublic.sha256
  ) {
    throw new Error("Public corpus differs from its pinned manifest");
  }

  return {
    [relative(repo, publicManifest)]: config.inputs.publicManifestSha256,
    [relative(repo, publicCorpus)]: actualPublic.sha256,
  };
}

async function inputHashes(
  config: Config,
  protectedText: ReturnType<typeof loadProtectedText>,
): Promise<Record<string, string>> {
  const manualSource = manualSourcePaths(config, protectedText);
  const manualCorpus = manualSource.corpus;
  const manualManifest = manualSource.manifest;
  const manualManifestSha256 = fileHash(manualManifest);
  const manual = manualSourceManifestSchema.parse(JSON.parse(readFileSync(manualManifest, "utf8")));

  if (
    manual.records !== config.manual.records ||
    manual.seed !== config.manual.seed ||
    manual.kernelsPerKind !== config.manual.kernelsPerKind ||
    manual.spellingNoise !== config.manual.spellingNoise
  ) {
    throw new Error("Manual-source manifest does not match production config");
  }

  const actualManual = await artifact(manualCorpus);

  if (
    actualManual.bytes !== manual.compressed.bytes ||
    actualManual.sha256 !== manual.compressed.sha256
  ) {
    throw new Error("Compressed manual source differs from its manifest");
  }

  const publicHashes = await publicSourceHashes(config);

  const verifiedLlm = resolveTrainingPath(config.inputs.verifiedLlm);
  const constructions = fileURLToPath(CONSTRUCTIONS_PATH);

  const hashes: Record<string, string> = {
    [relative(repo, fileURLToPath(PRODUCTION_CONFIG_PATH))]: fileHash(PRODUCTION_CONFIG_PATH),
    [relative(repo, manualManifest)]: manualManifestSha256,
    [relative(repo, manualCorpus)]: actualManual.sha256,
    ...publicHashes,
    [relative(repo, verifiedLlm)]: fileHash(verifiedLlm),
    [relative(repo, constructions)]: fileHash(constructions),
    ...hashDirectoryFiles(resolveTrainingPath(config.inputs.acceptedRealDirectory)),
    ...manualSource.inputs,
  };

  for (const source of protectedText.sources) hashes[relative(repo, source.file)] = source.sha256;

  return Object.fromEntries(Object.entries(hashes).sort(([a], [b]) => a.localeCompare(b)));
}

function verifyPythonAvailable(python: string): Record<string, string> {
  const result = spawnSync(
    python,
    [
      "-c",
      "import json,sys,numpy,torch; print(json.dumps({'python':sys.version,'numpy':numpy.__version__,'torch':torch.__version__}))",
    ],
    { encoding: "utf8" },
  );

  if (result.error || result.status !== 0) {
    throw new Error(`Python training environment unavailable: ${result.error ?? result.stderr}`);
  }

  return JSON.parse(result.stdout) as Record<string, string>;
}

function assertState(state: BuildState, output: string, config: Config): void {
  if (
    state.version !== 1 ||
    state.output !== output ||
    state.corpusVersion !== config.corpusVersion ||
    state.configSha256 !== fileHash(PRODUCTION_CONFIG_PATH)
  ) {
    throw new Error("Production work state belongs to different inputs or output");
  }

  if (JSON.stringify(corpusCodeHashes(import.meta.url)) !== JSON.stringify(state.implementation)) {
    throw new Error("Corpus implementation changed since the plan phase");
  }
}

function nextPhase(state: BuildState | undefined): ProductionPhase {
  if (!state?.plan) return "plan";
  if (!state.emit) return "emit";
  if (!state.expectations) return "expectations";
  if (!state.shards) return "shards";
  if (!state.verify) return "verify";

  return "publish";
}

async function planPhase(
  statePath: string,
  work: string,
  output: string,
  config: Config,
  progress: (phase: string, count?: number) => void,
): Promise<void> {
  if (existsSync(statePath)) throw new Error("Plan phase is already complete");
  // Without a committed state file, everything at this owned hidden path is
  // an interrupted plan attempt and is safe to rebuild.
  rmSync(work, { recursive: true, force: true });
  mkdirSync(work, { recursive: true });
  diskPreflight(work, INITIAL_REQUIRED_BYTES);
  const protectedText = loadProtectedText();
  const inputs = await inputHashes(config, protectedText);
  const implementation = corpusCodeHashes(import.meta.url);
  const python = join(repo, "training/cron/.venv/bin/python");

  const toolchain = {
    bun: process.versions.bun,
    platform: process.platform,
    architecture: process.arch,
    ...verifyPythonAvailable(python),
  };

  const scratch = join(work, ".union-scratch");

  rmSync(scratch, { recursive: true, force: true });

  const union = new StreamingCorpusUnion({
    directory: scratch,
    partitions: config.union.partitions,
  });

  const candidatesBySource = emptySourceCounts();

  try {
    for await (const candidate of completeSourceCandidates(
      sourceOptions(config, protectedText, progress),
    )) {
      candidatesBySource[candidate.source]++;
      union.add(candidate.record, candidate.union);
    }

    const plan = union.finalize();
    const target = join(work, "union-plan");

    rmSync(target, { recursive: true, force: true });
    mkdirSync(target);
    // Same filesystem: move the plan instead of copying its winner file.
    for (const name of CORPUS_UNION_PLAN_FILES) renameSync(join(scratch, name), join(target, name));

    const state: BuildState = {
      version: 1,
      corpusVersion: config.corpusVersion,
      output,
      config,
      configSha256: fileHash(PRODUCTION_CONFIG_PATH),
      inputs,
      implementation,
      toolchain,
      plan: { summary: plan.summary, candidatesBySource },
    };

    replaceJson(statePath, state);
  } finally {
    union.abort();
  }
}

/**
 * Replay the union and write the winners in text-hash order. Source traversal
 * order groups rows by source, and the trainer shuffles only inside bounded
 * locality blocks, so the published file itself must already be well mixed.
 */
async function emitPhase(
  state: BuildState,
  statePath: string,
  work: string,
  config: Config,
  progress: (phase: string, count?: number) => void,
): Promise<void> {
  if (!state.plan || state.emit) throw new Error("Emit phase is not pending");
  const protectedText = loadProtectedText();
  const plan = new CorpusUnionPlan(join(work, "union-plan"));
  const replay = plan.replay();
  const partial = join(work, "corpus.jsonl.partial.gz");
  const corpusPath = join(work, "corpus.jsonl.gz");

  rmSync(partial, { force: true });
  rmSync(corpusPath, { force: true });
  const spool = new HashOrderedSpool(join(work, ".emit-spool"));
  // Every counter below is keyed by a closed vocabulary (source, split,
  // population, authorship), never by a row value.
  const emittedBySource = emptySourceCounts();
  const splits = { train: 0, validation: 0, heldout: 0 };
  const populations: Record<Population, number> = { R: 0, H: 0, E: 0 };
  const authorship: Record<string, number> = {};
  let corpus: Artifact;

  try {
    for await (const candidate of completeSourceCandidates(
      sourceOptions(config, protectedText, progress),
    )) {
      const decision = replay.add(candidate.record, candidate.union);

      if (!decision.selected) continue;

      const record: CorpusRecord = {
        ...candidate.record,
        id: `${config.corpusVersion}/${decision.textHash}`,
      };

      spool.add(decision.textHash, JSON.stringify(record));
      emittedBySource[candidate.source]++;
      splits[record.split]++;

      const population = classifyPopulation({
        intent: record.intent,
        speechAct: record.speechAct ?? "none",
        spans: record.spans,
      });

      populations[population]++;
      const author = record.provenance?.authorship ?? "legacy-unspecified";

      authorship[author] = (authorship[author] ?? 0) + 1;
    }

    replay.finish();
    corpus = await writeGzip(partial, spool.chunks());
  } finally {
    replay.close();
    spool.remove();
  }

  if (Object.values(splits).some((count) => count === 0)) {
    throw new Error(`Complete corpus has an empty split: ${JSON.stringify(splits)}`);
  }

  renameSync(partial, corpusPath);
  state.emit = { corpus, emittedBySource, splits, populations, authorship };
  replaceJson(statePath, state);
}

async function expectationsPhase(
  state: BuildState,
  statePath: string,
  work: string,
): Promise<void> {
  if (!state.plan || !state.emit || state.expectations) {
    throw new Error("Expectations phase is not pending");
  }

  const expectations = await sourceExpectations(join(work, "corpus.jsonl.gz"));

  if (
    expectations.corpus.records !== state.plan.summary.uniqueTexts ||
    expectations.corpus.sha256 !== state.emit.corpus.sha256
  ) {
    throw new Error("JSONL expectations differ from the union output");
  }

  const path = join(work, "source-expectations.json");

  rmSync(path, { force: true });
  writeJsonExclusive(path, expectations);

  const expectedShardBytes = Object.values(expectations.splits).reduce(
    (total, split) =>
      total +
      split.sequenceMetadata.bytes +
      Object.values(split.fields).reduce((sum, field) => sum + field.bytes, 0),
    0,
  );

  state.expectations = { artifact: await artifact(path), expectedShardBytes };
  replaceJson(statePath, state);
}

function shardArtifacts(
  work: string,
  expectations: Expectations,
): Promise<Record<string, Artifact>> {
  return (async () => {
    const artifacts: Record<string, Artifact> = {};

    for (const split of SPLITS) {
      const prefix = `corpus-${split}`;

      artifacts[`${prefix}.json`] = await artifact(join(work, `${prefix}.json`));
      const expectedSplit = expectations.splits[split]!;
      const metadataName = `${prefix}${expectedSplit.sequenceMetadata.suffix}`;
      const metadataArtifact = await artifact(join(work, metadataName));

      if (
        metadataArtifact.bytes !== expectedSplit.sequenceMetadata.bytes ||
        metadataArtifact.sha256 !== expectedSplit.sequenceMetadata.sha256
      ) {
        throw new Error(`Sequence metadata differs from source expectations: ${metadataName}`);
      }

      artifacts[metadataName] = metadataArtifact;

      for (const [field, format] of Object.entries(SHARD_FORMAT)) {
        const name = `${prefix}${format.suffix}`;
        const expected = expectedSplit.fields[field as keyof typeof expectedSplit.fields];

        if (!expected || statSync(join(work, name)).size !== expected.bytes) {
          throw new Error(`Shard artifact differs from expectations: ${name}`);
        }

        artifacts[name] = expected;
      }
    }

    return artifacts;
  })();
}

async function shardsPhase(state: BuildState, statePath: string, work: string): Promise<void> {
  if (!state.plan || !state.emit || !state.expectations || state.shards) {
    throw new Error("Shards phase is not pending");
  }

  const expectations = JSON.parse(
    readFileSync(join(work, "source-expectations.json"), "utf8"),
  ) as Expectations;

  diskPreflight(work, state.expectations.expectedShardBytes + 64 * 1024 ** 2, 512 * 1024 ** 2);

  const summary = await featurizeKindsStreaming(
    join(work, "corpus.jsonl.gz"),
    join(work, "corpus"),
  );

  if (
    summary.records !== state.plan.summary.uniqueTexts ||
    Object.values(summary.splits).some(
      (split) => split.skippedEmpty !== 0 || split.skippedLong !== 0,
    )
  ) {
    throw new Error("Streaming featurization skipped complete-corpus records");
  }

  state.shards = { artifacts: await shardArtifacts(work, expectations) };
  replaceJson(statePath, state);
}

async function verifyPhase(state: BuildState, statePath: string, work: string): Promise<void> {
  if (!state.plan || !state.shards || state.verify) throw new Error("Verify phase is not pending");
  const report = join(work, "python-verification.json");

  rmSync(report, { force: true });
  const python = join(repo, "training/cron/.venv/bin/python");

  const result = spawnSync(
    python,
    [
      join(repo, "training/cron/verify_production_shards.py"),
      join(work, "corpus.jsonl.gz"),
      join(work, "corpus"),
      join(work, "source-expectations.json"),
      report,
    ],
    { stdio: "inherit" },
  );

  if (result.error || result.status !== 0) {
    throw new Error(`Python shard verification failed: ${result.error ?? result.status}`);
  }

  const parsed = JSON.parse(readFileSync(report, "utf8")) as {
    passed?: boolean;
    records?: number;
  };

  if (!parsed.passed || parsed.records !== state.plan.summary.uniqueTexts) {
    throw new Error("Python did not consume the complete corpus");
  }

  state.verify = { artifact: await artifact(report) };
  replaceJson(statePath, state);
}

function syncDirectory(path: string): void {
  const fd = openSync(path, "r");

  try {
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

async function publishPhase(
  state: BuildState,
  statePath: string,
  work: string,
  output: string,
): Promise<void> {
  if (!state.plan || !state.emit || !state.expectations || !state.shards || !state.verify) {
    throw new Error("Publish phase requires every prior phase");
  }

  const protectedText = loadProtectedText();
  const currentInputs = await inputHashes(state.config, protectedText);

  if (JSON.stringify(currentInputs) !== JSON.stringify(state.inputs)) {
    throw new Error("Production inputs changed since the plan phase");
  }

  for (const [name, hash] of Object.entries(
    hashDirectoryFiles(resolveTrainingPath(state.config.inputs.acceptedRealDirectory)),
  )) {
    if (state.inputs[name] !== hash) {
      throw new Error(`Production input appeared during generation: ${name}`);
    }
  }

  const artifacts: Record<string, Artifact> = {
    "corpus.jsonl.gz": state.emit.corpus,
    "source-expectations.json": state.expectations.artifact,
    "python-verification.json": state.verify.artifact,
    ...state.shards.artifacts,
  };

  const manifest = {
    version: 1,
    status: "budgeted-unique-corpus-and-verified-shards",
    corpusVersion: state.config.corpusVersion,
    policy: state.config.policy,
    config: state.config,
    configSha256: state.configSha256,
    sourceTraversalVersion: COMPLETE_SOURCE_VERSION,
    inputs: state.inputs,
    implementation: state.implementation,
    toolchain: state.toolchain,
    counts: {
      candidates: state.plan.summary.candidates,
      uniqueRecords: state.plan.summary.uniqueTexts,
      duplicatesRemoved: state.plan.summary.duplicates,
      candidatesBySource: state.plan.candidatesBySource,
      emittedBySource: state.emit.emittedBySource,
      splits: state.emit.splits,
      populations: state.emit.populations,
      authorship: state.emit.authorship,
    },
    union: {
      partitions: state.plan.summary.partitions,
      ingestSha256: state.plan.summary.ingestSha256,
      selectedReferencesSha256: state.plan.summary.selectedReferencesSha256,
      identity: "SHA-256 of runtime-normalized text",
      order: "ascending text hash (source-independent, deterministic)",
      conflictPolicy: "same wording with different model supervision fails",
      winnerPolicy: "source priority, then stable provenance reference",
    },
    protectedSources: protectedText.sources.map((source) => ({
      ...source,
      file: relative(repo, source.file),
    })),
    trainingInputPrefix: "corpus",
    splitPrefixes: Object.fromEntries(SPLITS.map((split) => [split, `corpus-${split}`])),
    artifacts,
    limitations: [
      ...(state.emit.emittedBySource["accepted-human"] === 0
        ? ["No accepted real-user rows currently carry explicit consented human provenance"]
        : []),
      "AI-authored constructions are labeled as AI-authored",
      "Sources are budgeted: config.manual sets generated rows and config.sampling " +
        "sets deterministic keep rates; unsampled wordings are not training evidence",
      "Corpus verification proves data integrity and loader consumption, not model quality",
    ],
  };

  rmSync(join(work, "union-plan"), { recursive: true, force: true });
  rmSync(join(work, "manifest.json"), { force: true });
  writeJsonExclusive(join(work, "manifest.json"), manifest);
  const manifestFd = openSync(join(work, "manifest.json"), "r");

  try {
    fsyncSync(manifestFd);
  } finally {
    closeSync(manifestFd);
  }

  syncDirectory(work);
  if (existsSync(output)) throw new Error(`Refusing to overwrite production version: ${output}`);
  renameSync(work, output);
  syncDirectory(dirname(output));
  rmSync(statePath, { force: true });
}

/** Advance preparation by its next pending phase. */
async function runProductionPhase(
  output: string,
  progress: (phase: string, count?: number) => void,
): Promise<ProductionPhase | "already-published"> {
  const paths = pathsFor(output);

  mkdirSync(paths.parent, { recursive: true });
  using _lock = acquirePreparationLock(paths.output);

  if (existsSync(paths.output)) {
    if (!existsSync(join(paths.output, "manifest.json"))) {
      throw new Error(`Output exists without a production manifest: ${paths.output}`);
    }

    rmSync(paths.state, { force: true });

    return "already-published";
  }

  const config = configSchema.parse(JSON.parse(readFileSync(PRODUCTION_CONFIG_PATH, "utf8")));
  const state = existsSync(paths.state) ? readState(paths.state) : undefined;

  if (state) assertState(state, paths.output, config);
  const phase = nextPhase(state);

  progress(`phase/${phase}`);

  if (phase === "plan") {
    await planPhase(paths.state, paths.work, paths.output, config, progress);
  } else {
    if (!state) throw new Error(`${phase} phase requires a completed plan phase`);
    if (phase === "emit") await emitPhase(state, paths.state, paths.work, config, progress);
    else if (phase === "expectations") await expectationsPhase(state, paths.state, paths.work);
    else if (phase === "shards") await shardsPhase(state, paths.state, paths.work);
    else if (phase === "verify") await verifyPhase(state, paths.state, paths.work);
    else await publishPhase(state, paths.state, paths.work, paths.output);
  }

  return phase;
}

/** Complete or resume preparation for the single training workflow.
 *
 * Input/code identity owns the directory: changing a source never silently
 * reuses old shards, and a published corpus is never overwritten.
 */
export async function prepareProductionCorpus(
  progress: (phase: string, count?: number) => void,
): Promise<string> {
  const config = configSchema.parse(JSON.parse(readFileSync(PRODUCTION_CONFIG_PATH, "utf8")));

  progress("source-integrity");
  await publicSourceHashes(config);
  fileHash(resolveTrainingPath(config.inputs.verifiedLlm));
  await prepareManualSource(config, progress);
  progress("corpus-input-integrity");
  const inputs = await inputHashes(config, loadProtectedText());
  const implementation = corpusCodeHashes(import.meta.url);

  const identity = createHash("sha256")
    .update(JSON.stringify({ config, inputs, implementation }))
    .digest("hex");

  const output = join(training, "data/synth", `${config.corpusVersion}-${identity.slice(0, 16)}`);

  for (;;) {
    const completed = await runProductionPhase(output, progress);

    if (completed === "publish" || completed === "already-published") break;
  }

  const manifest = JSON.parse(readFileSync(join(output, "manifest.json"), "utf8")) as {
    status?: string;
    inputs?: Record<string, string>;
    implementation?: Record<string, string>;
    artifacts?: Record<string, Artifact>;
  };

  if (
    manifest.status !== "budgeted-unique-corpus-and-verified-shards" ||
    JSON.stringify(manifest.inputs) !== JSON.stringify(inputs) ||
    JSON.stringify(manifest.implementation) !== JSON.stringify(implementation) ||
    !manifest.artifacts ||
    Object.keys(manifest.artifacts).length === 0
  ) {
    throw new Error("Published training corpus does not match the current source recipe");
  }

  progress("published-shard-integrity");

  for (const [name, expected] of Object.entries(manifest.artifacts)) {
    if (basename(name) !== name) throw new Error(`Invalid corpus artifact name: ${name}`);
    const path = join(output, name);

    if (
      !existsSync(path) ||
      statSync(path).size !== expected.bytes ||
      (await hashArtifact(path)) !== expected.sha256
    ) {
      throw new Error(`Published training artifact changed: ${path}`);
    }
  }

  const verified = JSON.parse(readFileSync(join(output, "python-verification.json"), "utf8")) as {
    passed?: boolean;
  };

  if (!verified.passed) {
    throw new Error("The production corpus has no passing Python consumption verification");
  }

  return join(output, "corpus");
}

if (import.meta.main) {
  throw new Error("Corpus preparation is internal. Run from the repository root: bun run train");
}
