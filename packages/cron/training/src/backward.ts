/** Catalog frame splitting and source identity used by corpus preparation. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { isScheduleLabel } from "../../src/model/labels";
import type { Split } from "./corpus/merge";
import { coarseText, framePattern } from "./fingerprint";
import type { ExampleSplit, KindExample, KindId } from "./kinds";

/** Provenance seed stored on every record; the expansion is not random. */
export const BACKWARD_SEED = 20260913;

/** Versioned split salt. Never depend on row order, values, or selected kinds. */
export const SPLIT_VERSION = "kinds-frames-v3"; // v3: per-kind workers use the pure global frame hash, no repair pass

const SPLITS: readonly Split[] = ["train", "validation", "heldout"];

const GENERATOR_SOURCES: readonly string[] = [
  "../../src/parser/lexicon.ts",
  "backward.ts",
  "correction.ts",
  "corpus/production/protected-text.ts",
  "discourse.ts",
  "fingerprint.ts",
  "kinds.ts",
  "mix/draw.ts",
  "mix/easy.ts",
  "mix/filter.ts",
  "mix/hash-set.ts",
  "mix/quotas.ts",
  "mix/tatoeba.ts",
  "noise.ts",
  "sentence.ts",
  "slots/compose.ts",
  "slots/hedge.ts",
  "slots/reason.ts",
  "slots/sample.ts",
  "slots/scene.ts",
  "slots/speech.ts",
  "slots/tail.ts",
  "slots/types.ts",
];

export function generatorSha256(): string {
  const base = dirname(fileURLToPath(import.meta.url));
  const digest = createHash("sha256");

  for (const name of [...GENERATOR_SOURCES].sort()) {
    digest.update(name, "utf8");
    digest.update("\0", "utf8");
    digest.update(readFileSync(join(base, name)));
    digest.update("\0", "utf8");
  }

  return digest.digest("hex");
}

/** Carrier wording cannot turn a known schedule construction into a new frame.
 * Negatives have no schedule slots; group them by normalized prose instead. */
export function frameKey(example: KindExample): string {
  const splitExample = example.splitExample ?? example.example;
  const hasScheduleLabels = splitExample.spans.some((span) => isScheduleLabel(span.label));

  if (example.intent === null && !hasScheduleLabels) {
    return `negative:${coarseText(splitExample.text)}`;
  }

  const speechAct = splitExample.speechAct ?? (example.intent === null ? "none" : "request");
  const pattern = framePattern(splitExample.text, splitExample.spans, speechAct);

  if (example.intent === null) return JSON.stringify(pattern);

  return JSON.stringify(pattern.filter((part) => part !== "O"));
}

/** Assign entire frames, globally and deterministically, rather than rows.
 * Roughly 75% of frames train, 12.5% validation, 12.5% held-out. */
export function assignSplits(examples: readonly KindExample[]): KindExample[] {
  const plan = splitPlan(examples);

  return examples.map((example) => assignSplit(example, plan));
}

type SplitPlan = ReadonlyMap<KindId, ReadonlyMap<string, ExampleSplit>>;

function splitBucket(key: string): number {
  return createHash("sha256").update(`${SPLIT_VERSION}\0${key}`).digest().readUInt32BE(0) % 8;
}

/** Assign equal structural frames globally, then repair sparse kinds so train,
 * validation, and held-out all contain at least one distinct frame. A frame can
 * occur in more than one kind (for example an explicit day-of-month and a
 * named day rule can have the same labeled surface shape), so assigning it
 * independently per kind would leak that frame across splits. */
function splitPlan(examples: Iterable<KindExample>): SplitPlan {
  const keys = new Map<KindId, Set<string>>();

  for (const example of examples) {
    const set = keys.get(example.kind) ?? new Set<string>();

    set.add(frameKey(example));
    keys.set(example.kind, set);
  }

  const orderedByKind = new Map<KindId, string[]>();

  for (const [kind, values] of keys) {
    const ordered = [...values].sort((left, right) => {
      const leftHash = createHash("sha256").update(left).digest("hex");
      const rightHash = createHash("sha256").update(right).digest("hex");

      return leftHash.localeCompare(rightHash);
    });

    if (ordered.length < SPLITS.length) {
      throw new Error(`${kind}: at least ${SPLITS.length} distinct frames are required`);
    }

    orderedByKind.set(kind, ordered);
  }

  const assignments = new Map<string, ExampleSplit>();

  for (const ordered of orderedByKind.values()) {
    for (const key of ordered) {
      if (assignments.has(key)) continue;
      const bucket = splitBucket(key);

      assignments.set(key, bucket === 6 ? "validation" : bucket === 7 ? "heldout" : "train");
    }
  }

  const kindsForKey = new Map<string, KindId[]>();

  for (const [kind, ordered] of orderedByKind) {
    for (const key of ordered) {
      const owners = kindsForKey.get(key) ?? [];

      owners.push(kind);
      kindsForKey.set(key, owners);
    }
  }

  const count = (kind: KindId, split: ExampleSplit): number =>
    orderedByKind.get(kind)!.filter((key) => assignments.get(key) === split).length;

  // Moving a shared frame is safe only when none of its owner kinds loses its
  // last frame in the donor split. Prefer moves that fill the same missing
  // split for several owner kinds at once.
  for (let pass = 0; pass < orderedByKind.size * SPLITS.length; pass++) {
    let changed = false;

    for (const [kind, ordered] of orderedByKind) {
      for (const split of SPLITS) {
        if (count(kind, split) !== 0) continue;

        const candidates = ordered
          .filter((key) => {
            const donor = assignments.get(key)!;

            return (
              donor !== split && kindsForKey.get(key)!.every((owner) => count(owner, donor) > 1)
            );
          })
          .sort((left, right) => {
            const leftGain = kindsForKey
              .get(left)!
              .filter((owner) => count(owner, split) === 0).length;

            const rightGain = kindsForKey
              .get(right)!
              .filter((owner) => count(owner, split) === 0).length;

            return rightGain - leftGain || left.localeCompare(right);
          });

        const chosen = candidates[0];

        if (chosen === undefined) {
          throw new Error(`${kind}: cannot assign a disjoint ${split} frame`);
        }

        assignments.set(chosen, split);
        changed = true;
      }
    }

    const complete = [...orderedByKind].every(([kind]) =>
      SPLITS.every((split) => count(kind, split) > 0),
    );

    if (complete) break;
    if (!changed) throw new Error("cannot make all kind splits structurally disjoint");
  }

  const plan = new Map<KindId, Map<string, ExampleSplit>>();

  for (const [kind, ordered] of orderedByKind) {
    plan.set(kind, new Map(ordered.map((key) => [key, assignments.get(key)!])));
  }

  return plan;
}

export function assignSplit(example: KindExample, plan?: SplitPlan): KindExample {
  const planned = plan?.get(example.kind)?.get(frameKey(example));

  if (planned !== undefined) return { ...example, split: planned };
  const bucket = splitBucket(frameKey(example));
  const split: ExampleSplit = bucket === 6 ? "validation" : bucket === 7 ? "heldout" : "train";

  return { ...example, split };
}
