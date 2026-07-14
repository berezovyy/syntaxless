/** Canonical source traversal for the complete, all-unique training corpus. */
import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";

import { z } from "zod";

import { SPEECH_ACTS } from "../../../../src/model/speech-acts";
import { BACKWARD_SEED } from "../../backward";
import { fingerprintExample, sha256Hex } from "../../fingerprint";
import { KINDS } from "../../kinds";
import { iterateMix } from "../../mix/draw";
import { loadAcceptedReal } from "../../real/accepted";
import { normalizeAnnotatedText } from "../../sentence";
import { type CorpusRecord, corpusRecordSchema, FAMILIES, readLlmRecords } from "../merge";
import { provenanceSchema } from "../provenance";
import { semanticCases } from "./cases";
import { catalogCandidates } from "./catalog";
import { loadConstructions, renderConstruction, verifyDiscarded } from "./constructions";
import type { CorpusUnionCandidateOptions } from "./global-union";
import { manualCorpusRecord, streamManualSource } from "./manual-source";
import { corpusTextKey, loadProtectedText, type ProtectedText } from "./protected-text";

export const COMPLETE_SOURCE_VERSION = "complete-sources-v1";

export interface CompleteSourceOptions {
  manualRecords?: number;
  manualSeed?: number;
  manualKernelsPerKind?: number;
  /** Reusable manual artifact. Both paths must be supplied together. */
  manualCorpusPath?: string;
  manualManifestPath?: string;
  publicCorpusPath: string;
  verifiedLlmPath: string;
  acceptedRealDirectory: string;
  protectedText?: ProtectedText;
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

function acceptedHumanRecords(directory: string): CorpusRecord[] {
  return loadAcceptedReal(directory).map((input, index) => {
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
  });
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

  const report = (source: CompleteSourceName, count: number): void => {
    if (count % 100_000 === 0) options.progress?.(source, count);
  };

  let count = 0;

  if (enabled.has("manual")) {
    if ((options.manualCorpusPath === undefined) !== (options.manualManifestPath === undefined)) {
      throw new Error("manualCorpusPath and manualManifestPath must be supplied together");
    }

    const records =
      options.manualCorpusPath !== undefined && options.manualManifestPath !== undefined
        ? streamManualSource({
            corpusPath: options.manualCorpusPath,
            manifestPath: options.manualManifestPath,
            ...(options.manualRecords === undefined ? {} : { records: options.manualRecords }),
            ...(options.manualSeed === undefined ? {} : { seed: options.manualSeed }),
            ...(options.manualKernelsPerKind === undefined
              ? {}
              : { kernelsPerKind: options.manualKernelsPerKind }),
          })
        : (async function* (): AsyncGenerator<CorpusRecord> {
            let ordinal = 0;

            for (const example of iterateMix({
              count: options.manualRecords ?? 20_000_000,
              seed: options.manualSeed ?? BACKWARD_SEED,
              perKindKernels: options.manualKernelsPerKind ?? 2_048,
            })) {
              yield manualCorpusRecord(example, ordinal);
              ordinal++;
            }
          })();

    for await (const record of records) {
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

    for (const candidate of catalogCandidates(KINDS)) {
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

    const lines = createInterface({
      input: createReadStream(options.publicCorpusPath, { encoding: "utf8" }),
      crlfDelay: Number.POSITIVE_INFINITY,
    });

    for await (const line of lines) {
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

      if (permitted(guard, record)) {
        yield { source: "public", record, union: { priority: PRIORITY.public } };
      }
    }
  }

  if (enabled.has("verified-llm")) {
    count = 0;

    for (const input of readLlmRecords(options.verifiedLlmPath)) {
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
