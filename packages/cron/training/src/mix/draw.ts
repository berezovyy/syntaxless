/**
 * Sample R / H / E to quota bands. Does not walk catalog order.
 */
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { isScheduleLabel } from "../../../src/model/labels";
import type { Split } from "../corpus/merge";
import { corpusTextKey, loadProtectedText, textKey } from "../corpus/production/protected-text";
import { fingerprintExample, sha256Hex } from "../fingerprint";
import type { KindExample } from "../kinds";
import { iterateScheduleKernels, type ScheduleKernel } from "../kinds";
import { spellingVariant } from "../noise";
import { createRng, type Rng } from "../rng";
import { normalizeAnnotatedText } from "../sentence";
import { compose } from "../slots/compose";
import { humanEvalTexts } from "../slots/leak";
import { sampleSlotDraw } from "../slots/sample";
import { SPEECH_FRAMES } from "../slots/speech";
import { DiskHash256Set } from "./disk-hash-set";
import { assertEasyRecord, easyKindExample } from "./easy";
import { scheduleTargetCounts, targetCounts } from "./quotas";
import { loadTatoebaSentences } from "./tatoeba";

const REQUEST_SPEECH = SPEECH_FRAMES.filter(
  (frame) => frame.act === "request" || frame.act === "correction",
);

const HARD_SPEECH = SPEECH_FRAMES.filter(
  (frame) => frame.act !== "request" && frame.act !== "correction",
);

export interface MixOptions {
  count: number;
  seed: number;
  /** If set, every row gets this split. Otherwise 75/12.5/12.5 with quotas in each. */
  split?: Split;
  perKindKernels?: number;
  /**
   * Draw the E population from the 50k-line Tatoeba list (default). Production
   * disables it: its negatives come from the public source, which already
   * contains every one of these sentences and wins the union.
   */
  easyNegatives?: boolean;
  /** Share of R/H rows replaced by a meaning-preserving spelling variant. */
  spellingNoise?: number;
  /**
   * Directory for the disk-backed uniqueness index (two SQLite files, removed
   * when the generator finishes). Production passes a directory on the data
   * disk. When omitted, a fresh directory under the OS temp directory is used,
   * which may be RAM-backed tmpfs: acceptable only for small mixes.
   */
  dedupDirectory?: string;
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
      if (!composed.sentence.spans.some((span) => isScheduleLabel(span.label))) continue;
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

function assignBucketSplit(example: KindExample, split: Split): KindExample {
  return { ...example, split };
}

/** One typo in a non-literal token; the clean wording stays drawable. */
function noisyWrap(wrapped: KindExample, seed: number): KindExample {
  const noisy = spellingVariant(wrapped.example, `mix-noise-${seed}`, { force: true });

  if (noisy === undefined) return wrapped;
  const template = `${wrapped.template}/noise-${noisy.noise}`;

  return {
    ...wrapped,
    template,
    example: {
      ...noisy.example,
      ...normalizeAnnotatedText(noisy.example.text, noisy.example.spans),
      template,
      style: `${wrapped.example.style}-noise`,
    },
  };
}

/** Match the canonical manual-record fingerprint used by corpus preparation. */
function exampleFingerprint(example: KindExample): string {
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
 * 50k; a 20M mix cannot have 2M unique E). Without easy negatives the R/H
 * ratio is unchanged and every row is unique.
 */
export function* iterateMix(options: MixOptions): Generator<KindExample> {
  const easyNegatives = options.easyNegatives ?? true;
  const spellingNoise = options.spellingNoise ?? 0;

  if (!(spellingNoise >= 0 && spellingNoise <= 1)) {
    throw new RangeError(`mix: spellingNoise must be within [0, 1]: ${spellingNoise}`);
  }

  const quotas = easyNegatives ? targetCounts : scheduleTargetCounts;
  const rng = createRng(options.seed);
  const kernels = kernelPool(options.perKindKernels ?? 48);

  if (kernels.length === 0) throw new Error("mix: no schedule kernels");
  const protectedText = loadProtectedText();

  // Partition normalized identities, not source lines: casing/punctuation
  // variants must not create distinct train and held-out examples.
  const easyBySplit: Record<Split, readonly string[]> = easyNegatives
    ? partitionEasy([
        ...new Map(
          loadTatoebaSentences({ blocked: new Set(humanEvalTexts()) })
            .filter((text) => !protectedText.matches(text))
            .map((text) => [textKey(text), text]),
        ).values(),
      ])
    : { train: [], validation: [], heldout: [] };

  const sizes = splitSizes(options.count, options.split);

  // Exact online uniqueness needs every earlier digest. Keep them on disk so
  // generator memory does not grow with `count` (see disk-hash-set.ts).
  const ownedDirectory =
    options.dedupDirectory === undefined
      ? mkdtempSync(join(tmpdir(), "syntaxless-mix-"))
      : undefined;

  const dedupDirectory = options.dedupDirectory ?? ownedDirectory!;

  mkdirSync(dedupDirectory, { recursive: true });
  let seen: DiskHash256Set | undefined;
  let seenTexts: DiskHash256Set | undefined;

  try {
    seen = new DiskHash256Set(join(dedupDirectory, "mix-fingerprints.sqlite"));
    seenTexts = new DiskHash256Set(join(dedupDirectory, "mix-texts.sqlite"));

    for (const [split, n] of sizes) {
      if (n === 0) continue;
      const easy = easyBySplit[split];

      if (easyNegatives && easy.length === 0) {
        throw new Error(`mix: no easy-no sentences for ${split}`);
      }

      const remaining = { ...quotas(n) };
      let easyIndex = 0;

      while (remaining.R + remaining.H + remaining.E > 0) {
        const population = rng.weighted<"R" | "H" | "E">([
          ["R", remaining.R],
          ["H", remaining.H],
          ...(easyNegatives ? ([["E", remaining.E]] as const) : []),
        ]);

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
          const clean = wrapKernel(row, population, rng);

          // No rng draw at share 0 keeps noise-free mixes byte-identical.
          const wrapped =
            spellingNoise > 0 && rng.next() < spellingNoise
              ? noisyWrap(clean, options.seed)
              : clean;

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
  } finally {
    seen?.close();
    seenTexts?.close();
    if (ownedDirectory !== undefined) rmSync(ownedDirectory, { recursive: true, force: true });
  }
}
