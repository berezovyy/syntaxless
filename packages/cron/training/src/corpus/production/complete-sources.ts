/** Canonical source traversal for the budgeted, all-unique training corpus. */
import { createHash } from "node:crypto";

import { z } from "zod";

import { SPEECH_ACTS } from "../../../../src/model/speech-acts";
import { fingerprintExample, sha256Hex } from "../../fingerprint";
import { KINDS } from "../../kinds";
import { type AcceptedRealRecord, iterateAcceptedReal } from "../../real/accepted";
import { normalizeAnnotatedText } from "../../sentence";
import { jsonlByteChunks, jsonlReader } from "../jsonl-io";
import { type CorpusRecord, corpusRecordSchema, FAMILIES, iterateLlmRecords } from "../merge";
import { provenanceSchema } from "../provenance";
import { semanticCases } from "./cases";
import { catalogCandidates } from "./catalog";
import { loadConstructions, renderConstruction, verifyDiscarded } from "./constructions";
import type { CorpusUnionCandidateOptions } from "./global-union";
import { type ManualSource, streamManualSource } from "./manual-source";
import { corpusTextKey, loadProtectedText, type ProtectedText } from "./protected-text";

/** v2: per-source deterministic sampling rates. */
export const COMPLETE_SOURCE_VERSION = "complete-sources-v2";

/**
 * Keep rates for sources whose size is fixed by their inputs rather than by a
 * generation budget. 1 keeps everything; omitted sources keep everything.
 */
export interface SourceSampling {
  authored?: number;
  systematic?: number;
  public?: number;
}

/** Stable keep decision: the same key and rate always give the same answer. */
export function sampled(rate: number, ...key: readonly string[]): boolean {
  if (!(rate >= 0 && rate <= 1)) {
    throw new RangeError(`Source sampling rate must be within [0, 1]: ${rate}`);
  }

  if (rate === 1) return true;
  const value = createHash("sha256").update(key.join("\0")).digest().readUInt32BE(0);

  return value < rate * 2 ** 32;
}

export interface CompleteSourceOptions {
  /** Prepared artifact required whenever the manual source is traversed. */
  manualSource?: ManualSource;
  publicCorpusPath: string;
  verifiedLlmPath: string;
  acceptedRealDirectory: string;
  protectedText?: ProtectedText;
  sampling?: SourceSampling;
  /** Test/debug restriction. Production omits this and traverses every source. */
  sources?: readonly CompleteSourceName[];
  progress?: (source: CompleteSourceName, candidates: number) => void;
}

export type CompleteSourceName =
  "manual" | "authored" | "systematic" | "public" | "verified-llm" | "accepted-human";

export interface CompleteSourceCandidate {
  source: CompleteSourceName;
  record: CorpusRecord;
  union: CorpusUnionCandidateOptions;
}

const PRIORITY: Readonly<Record<CompleteSourceName, number>> = {
  "accepted-human": 0,
  public: 10,
  authored: 20,
  "verified-llm": 30,
  manual: 40,
  systematic: 50,
};

const acceptedHumanMetadataSchema = z
  .object({
    id: z.string().min(1),
    family: z.enum(FAMILIES),
    speechAct: z.enum(SPEECH_ACTS),
    trainingConsent: z
      .object({
        granted: z.literal(true),
        scope: z.literal("model-training"),
        recordedBy: z.string().min(1),
      })
      .strict(),
    provenance: provenanceSchema.refine(
      (value) => value.authorship === "human-authored",
      "accepted human records require human-authored provenance",
    ),
  })
  .passthrough();

function* acceptedHumanRecords(directory: string): Generator<CorpusRecord> {
  let index = 0;

  for (const input of iterateAcceptedReal(directory)) {
    yield acceptedHumanRecord(input, index++);
  }
}

function acceptedHumanRecord(input: AcceptedRealRecord, index: number): CorpusRecord {
  const metadata = acceptedHumanMetadataSchema.safeParse(input);

  if (!metadata.success) {
    throw new Error(
      `Accepted real record ${input.id || index} lacks explicit human provenance and model-training consent: ${metadata.error.message}`,
    );
  }

  if (input.spans.some((span) => span.actionable === undefined)) {
    throw new Error(`Accepted real record ${input.id} lacks explicit span actionability`);
  }

  const annotated = normalizeAnnotatedText(
    input.text,
    input.spans as Parameters<typeof normalizeAnnotatedText>[1],
  );

  const speechAct = metadata.data.speechAct;

  const fingerprint =
    input.intent === null
      ? sha256Hex(`negative\0${annotated.text}`)
      : fingerprintExample({ ...annotated, speechAct });

  return corpusRecordSchema.parse({
    id: `${COMPLETE_SOURCE_VERSION}/accepted-human/${input.id}`,
    split: "train",
    family: metadata.data.family,
    fingerprint,
    text: annotated.text,
    intent: input.intent === null ? null : structuredClone(input.intent),
    speechAct,
    spans: annotated.spans,
    template: "real/accepted",
    source: "llm",
    provenance: metadata.data.provenance,
  }) as CorpusRecord;
}

function permitted(guard: ProtectedText, record: CorpusRecord): boolean {
  return !guard.matches(record.text);
}

/**
 * Traverse every canonical input in a frozen order. Calling this twice with
 * unchanged inputs yields the identical sequence required by CorpusUnionPlan.
 */
export async function* completeSourceCandidates(
  options: CompleteSourceOptions,
): AsyncGenerator<CompleteSourceCandidate> {
  const guard = options.protectedText ?? loadProtectedText();

  const enabled = new Set<CompleteSourceName>(
    options.sources ?? [
      "manual",
      "authored",
      "systematic",
      "public",
      "verified-llm",
      "accepted-human",
    ],
  );

  const rate = {
    authored: options.sampling?.authored ?? 1,
    systematic: options.sampling?.systematic ?? 1,
    public: options.sampling?.public ?? 1,
  };

  const report = (source: CompleteSourceName, count: number): void => {
    if (count % 100_000 === 0) options.progress?.(source, count);
  };

  let count = 0;

  if (enabled.has("manual")) {
    if (options.manualSource === undefined) {
      throw new Error("A prepared manual source is required before corpus traversal");
    }

    for await (const record of streamManualSource(options.manualSource)) {
      count++;
      report("manual", count);

      if (permitted(guard, record)) {
        yield { source: "manual", record, union: { priority: PRIORITY.manual } };
      }
    }
  }

  if (enabled.has("authored")) {
    const constructions = loadConstructions();

    count = 0;

    for (const group of constructions.groups) {
      for (const semantic of semanticCases(group.caseSet)) {
        verifyDiscarded(semantic);

        for (const form of group.forms) {
          const record = renderConstruction(constructions, group, form, semantic);

          count++;
          report("authored", count);
          if (!sampled(rate.authored, "authored", record.id)) continue;

          if (permitted(guard, record)) {
            yield {
              source: "authored",
              record,
              union: { priority: PRIORITY.authored },
            };
          }
        }
      }
    }
  }

  if (enabled.has("systematic")) {
    count = 0;

    const include =
      rate.systematic === 1
        ? undefined
        : (constructionId: string, intentId: string) =>
            sampled(rate.systematic, "systematic", constructionId, intentId);

    for (const candidate of catalogCandidates(KINDS, undefined, include)) {
      const record: CorpusRecord = { ...candidate.record, split: "train" };

      count++;
      report("systematic", count);

      if (permitted(guard, record)) {
        yield {
          source: "systematic",
          record,
          union: { priority: PRIORITY.systematic },
        };
      }
    }
  }

  if (enabled.has("public")) {
    count = 0;
    await using reader = jsonlReader(jsonlByteChunks(options.publicCorpusPath));

    for await (const line of reader.lines) {
      if (!line.trim()) continue;
      count++;
      let record: CorpusRecord;

      try {
        record = corpusRecordSchema.parse(JSON.parse(line)) as CorpusRecord;
      } catch (error) {
        throw new Error(`Invalid public corpus record ${count} in ${options.publicCorpusPath}`, {
          cause: error,
        });
      }

      report("public", count);
      if (!sampled(rate.public, "public", record.id)) continue;

      if (permitted(guard, record)) {
        yield { source: "public", record, union: { priority: PRIORITY.public } };
      }
    }
  }

  if (enabled.has("verified-llm")) {
    count = 0;

    for (const input of iterateLlmRecords(options.verifiedLlmPath)) {
      if (corpusTextKey(input.text) !== input.text) {
        throw new Error(`Verified LLM record ${input.id} is not runtime-normalized`);
      }

      const record: CorpusRecord = {
        ...input,
        template: input.family === "negative" ? "easy-no" : input.family,
        provenance: {
          authorship: "ai-authored",
          sourceId: "verified-llm-v1",
          groupId: input.angleId ?? input.id,
          constructionId: input.angleId ?? "verified-llm",
          sourceRecordId: input.id,
          attribution: `${input.model ?? "unknown model"}; run ${input.runId ?? "unknown"}`,
        },
      };

      count++;
      report("verified-llm", count);

      if (permitted(guard, record)) {
        yield {
          source: "verified-llm",
          record,
          union: { priority: PRIORITY["verified-llm"] },
        };
      }
    }
  }

  if (enabled.has("accepted-human")) {
    count = 0;

    for (const record of acceptedHumanRecords(options.acceptedRealDirectory)) {
      count++;
      report("accepted-human", count);

      if (permitted(guard, record)) {
        yield {
          source: "accepted-human",
          record,
          union: { priority: PRIORITY["accepted-human"] },
        };
      }
    }
  }
}
