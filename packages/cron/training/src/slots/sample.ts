/**
 * Seeded draws from production bags. The product is sampled, never dumped.
 */
import { createRng, type Rng } from "../rng";
import { Sentence } from "../sentence";
import { compose, type ComposeResult } from "./compose";
import { slotFingerprint } from "./fingerprint";
import { HEDGE_PARTS } from "./hedge";
import { REASON_PARTS } from "./reason";
import { SCENE_PARTS } from "./scene";
import { SPEECH_FRAMES } from "./speech";
import { TAIL_PARTS } from "./tail";
import {
  type BagPart,
  EMPTY_PART,
  EMPTY_SPEECH,
  type HolePosition,
  POSITIONS,
  type SlotDraw,
  type SpeechFrame,
} from "./types";

export const SLOT_WRAPS_PER_KERNEL = 8;

const OPTIONAL_SLOTS = ["scene", "hedge", "reason", "tail"] as const;

type OptionalSlot = (typeof OPTIONAL_SLOTS)[number];

/**
 * Humans rarely stack hedge+scene+reason+tail. Fill 1 extra, sometimes 2.
 * Never emit the bare hole here — iterateKinds already yields kernel-only rows.
 */
export function sampleSlotDraw(
  rng: Rng,
  bags?: {
    speech?: readonly SpeechFrame[];
    scene?: readonly BagPart[];
    hedge?: readonly BagPart[];
    reason?: readonly BagPart[];
    tail?: readonly BagPart[];
    positions?: readonly HolePosition[];
    /** Default true: 35% empty speech (bare kernel as a request). Hard-no wraps set false. */
    allowEmptySpeech?: boolean;
  },
): SlotDraw {
  const speech = bags?.speech ?? SPEECH_FRAMES;
  const scene = bags?.scene ?? SCENE_PARTS;
  const hedge = bags?.hedge ?? HEDGE_PARTS;
  const reason = bags?.reason ?? REASON_PARTS;
  const tail = bags?.tail ?? TAIL_PARTS;
  const positions = bags?.positions ?? POSITIONS;
  const extras = rng.bool(0.3) ? 2 : 1;
  const filled = new Set<OptionalSlot>();

  while (filled.size < extras && filled.size < OPTIONAL_SLOTS.length) {
    filled.add(rng.pick(OPTIONAL_SLOTS));
  }

  const allowEmpty = bags?.allowEmptySpeech !== false;

  return {
    speech: (allowEmpty && rng.bool(0.35)) || speech.length === 0 ? EMPTY_SPEECH : rng.pick(speech),
    scene: filled.has("scene") && scene.length > 0 ? rng.pick(scene) : EMPTY_PART,
    hedge: filled.has("hedge") && hedge.length > 0 ? rng.pick(hedge) : EMPTY_PART,
    reason: filled.has("reason") && reason.length > 0 ? rng.pick(reason) : EMPTY_PART,
    tail: filled.has("tail") && tail.length > 0 ? rng.pick(tail) : EMPTY_PART,
    position: rng.pick(positions),
  };
}

export function seedFrom(parts: readonly string[]): number {
  let hash = 2166136261;

  for (const part of parts) {
    for (let index = 0; index < part.length; index++) {
      hash = Math.imul(hash ^ part.charCodeAt(index), 16777619);
    }
  }

  return hash >>> 0;
}

export function wrapsForKernel(
  hole: Sentence,
  seedParts: readonly string[],
  k = SLOT_WRAPS_PER_KERNEL,
): ComposeResult[] {
  const rng = createRng(seedFrom(seedParts));
  const out: ComposeResult[] = [];
  const seen = new Set<string>();
  let guard = 0;

  while (out.length < k && guard < k * 32) {
    guard += 1;
    const draw = sampleSlotDraw(rng);
    const fingerprint = slotFingerprint(draw);

    if (seen.has(fingerprint)) continue;
    seen.add(fingerprint);
    const composed = compose(hole, draw);

    if (composed === null) continue;
    out.push(composed);
  }

  return out;
}
