import { createHash } from "node:crypto";

/**
 * Sample R / H / E to quota bands. Does not walk catalog order.
 */
import type { SpeechAct } from "../../../src/model/speech-acts";
import { corpusTextKey, loadProtectedText, textKey } from "../corpus/production/protected-text";
import { fingerprintExample, sha256Hex } from "../fingerprint";
import type { Split } from "../generate";
import type { KindExample } from "../kinds";
import { iterateScheduleKernels, type ScheduleKernel } from "../kinds";
import { createRng, type Rng } from "../rng";
import { normalizeAnnotatedText } from "../sentence";
import { compose } from "../slots/compose";
import { humanEvalTexts } from "../slots/leak";
import { sampleSlotDraw } from "../slots/sample";
import { SPEECH_FRAMES } from "../slots/speech";
import { assertEasyRecord, easyKindExample } from "./easy";
import { Hash256Set } from "./hash-set";
import { bandFailures, targetCounts } from "./quotas";
import { type MixReport, mixReportFromExamples } from "./report";
import { type EasySourceKind, loadEasySource } from "./tatoeba";

const REQUEST_SPEECH = SPEECH_FRAMES.filter(
  (frame) => frame.act === "request" || frame.act === "correction",
);

const HARD_SPEECH = SPEECH_FRAMES.filter(
  (frame) => frame.act !== "request" && frame.act !== "correction",
);

export interface DrawMixOptions {
  count: number;
  seed: number;
  /** If set, every row gets this split. Otherwise 75/12.5/12.5 with quotas in each. */
  split?: Split;
  perKindKernels?: number;
}

export interface DrawMixResult {
  examples: KindExample[];
  report: MixReport;
  easySource: EasySourceKind;
}

export interface MixAccumulator {
  total: number;
  counts: Record<"R" | "H" | "E", number>;
  uniqueE: Set<string>;
  uniqueH: Set<string>;
}

function kernelPool(perKind: number): ScheduleKernel[] {
  return [...iterateScheduleKernels(perKind)];
}

function wrapKernel(row: ScheduleKernel, population: "R" | "H", rng: Rng): KindExample {
  const speech = population === "R" ? REQUEST_SPEECH : HARD_SPEECH;

  for (let attempt = 0; attempt < 128; attempt++) {
    const draw = sampleSlotDraw(rng, {
      speech,
      allowEmptySpeech: population === "R",
    });

    const act = draw.speech.act;

    if (population === "H" && (act === "request" || act === "correction")) continue;
    const composed = compose(row.hole, draw);

    if (composed === null) continue;

    if (population === "H") {
      if (composed.speechAct === "request" || composed.speechAct === "correction") continue;
      if (composed.holeActionable) continue;
      if (!composed.sentence.spans.some((span) => span.label !== "O")) continue;
    }

    const template = `${row.kernel.kind}/mix-${population}-${draw.speech.id}-${draw.position}-${draw.scene.id}-${draw.hedge.id}-${draw.reason.id}-${draw.tail.id}`;

    return {
      ...row.kernel,
      template,
      example: {
        ...normalizeAnnotatedText(composed.sentence.text, composed.sentence.spans),
        template,
        style: `mix-${population}`,
        speechAct: composed.speechAct,
      },
    };
  }

  throw new Error(`mix: could not wrap kernel ${row.kernel.template} as ${population}`);
}

function partitionEasy(sentences: readonly string[]): Record<Split, readonly string[]> {
  if (sentences.length < 3) {
    throw new Error("mix: need at least 3 easy-no sentences to partition splits");
  }

  const train = Math.floor(sentences.length * 0.75);
  const validation = Math.floor(sentences.length * 0.125);

  return {
    train: sentences.slice(0, train),
    validation: sentences.slice(train, train + validation),
    heldout: sentences.slice(train + validation),
  };
}

export function emptyMixAccumulator(): MixAccumulator {
  return {
    total: 0,
    counts: { R: 0, H: 0, E: 0 },
    uniqueE: new Set(),
    uniqueH: new Set(),
  };
}

export function noteMixExample(acc: MixAccumulator, example: KindExample): void {
  acc.total += 1;
  const hasHole = example.example.spans.some((span) => span.label !== "O");

  if (!hasHole) {
    acc.counts.E += 1;
    acc.uniqueE.add(example.example.text);

    return;
  }

  if (example.example.speechAct === "request" || example.example.speechAct === "correction") {
    acc.counts.R += 1;

    return;
  }

  acc.counts.H += 1;
  acc.uniqueH.add(example.template);
}

function assignBucketSplit(example: KindExample, split: Split): KindExample {
  return { ...example, split };
}

/** Same hash `toRecord` in backward.ts uses, so featurize split disjointness holds. */
export function exampleFingerprint(example: KindExample): string {
  if (example.intent === null) return sha256Hex(`negative\0${example.example.text}`);

  return fingerprintExample(example.example);
}

function splitSizes(count: number, split: Split | undefined): Array<readonly [Split, number]> {
  if (split !== undefined) return [[split, count]];
  const train = Math.floor(count * 0.75);
  const validation = Math.floor(count * 0.125);

  return [
    ["train", train],
    ["validation", validation],
    ["heldout", count - train - validation],
  ];
}

/**
 * Stream mix rows. R/H fingerprints are unique across splits. Easy-no texts
 * are partitioned across splits and may repeat inside a split (Tatoeba is
 * 50k; a 20M mix cannot have 2M unique E).
 */
export function* iterateMix(options: DrawMixOptions): Generator<KindExample> {
  const rng = createRng(options.seed);
  const kernels = kernelPool(options.perKindKernels ?? 48);

  if (kernels.length === 0) throw new Error("mix: no schedule kernels");
  const blocked = new Set(humanEvalTexts());
  const source = loadEasySource({ blocked });
  const protectedText = loadProtectedText();

  // Partition normalized identities, not source lines: casing/punctuation
  // variants must not create distinct train and held-out examples.
  const easySentences = [
    ...new Map(
      source.sentences
        .filter((text) => !protectedText.matches(text))
        .map((text) => [textKey(text), text]),
    ).values(),
  ];

  const easyBySplit = partitionEasy(easySentences);
  const sizes = splitSizes(options.count, options.split);

  const expectedSchedules = sizes.reduce((total, [, n]) => {
    const target = targetCounts(n);

    return total + target.R + target.H;
  }, 0);

  const seen = new Hash256Set(expectedSchedules);
  const seenTexts = new Hash256Set(expectedSchedules);

  for (const [split, n] of sizes) {
    if (n === 0) continue;
    const easy = easyBySplit[split];

    if (easy.length === 0) throw new Error(`mix: no easy-no sentences for ${split}`);
    const remaining = { ...targetCounts(n) };
    let easyIndex = 0;

    while (remaining.R + remaining.H + remaining.E > 0) {
      const population = rng.weighted([
        ["R", remaining.R],
        ["H", remaining.H],
        ["E", remaining.E],
      ] as const);

      if (population === "E") {
        const text = easy[easyIndex % easy.length]!;

        easyIndex += 1;
        const example = easyKindExample(text, easyIndex);

        assertEasyRecord(example);
        remaining.E -= 1;
        yield assignBucketSplit(example, split);
        continue;
      }

      let next: KindExample | undefined;

      for (let attempt = 0; attempt < 256; attempt++) {
        const row = kernels[rng.int(0, kernels.length - 1)]!;
        const wrapped = wrapKernel(row, population, rng);
        const fingerprint = Buffer.from(exampleFingerprint(wrapped), "hex");
        const text = createHash("sha256").update(corpusTextKey(wrapped.example.text)).digest();

        if (seen.has(fingerprint) || seenTexts.has(text)) continue;
        if (protectedText.matches(wrapped.example.text)) continue;
        seen.add(fingerprint);
        seenTexts.add(text);
        next = assignBucketSplit(wrapped, split);
        break;
      }

      if (next === undefined) {
        throw new Error(`mix: could not draw unique ${population} after 256 tries`);
      }

      remaining[population] -= 1;
      yield next;
    }
  }
}

export function drawMix(options: DrawMixOptions): DrawMixResult {
  const examples = [...iterateMix(options)];
  const blocked = new Set(humanEvalTexts());
  const source = loadEasySource({ blocked });
  const report = mixReportFromExamples(examples, source.kind);
  const failures = bandFailures(report.shares);

  if (failures.length > 0) throw new Error(`mix quotas missed: ${failures.join("; ")}`);

  return { examples, report, easySource: source.kind };
}

export function isRefuseSpeechAct(act: SpeechAct): boolean {
  return act !== "request" && act !== "correction";
}
