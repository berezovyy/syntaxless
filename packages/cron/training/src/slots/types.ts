/**
 * Slot-bag parts glued around a schedule hole. Diversity is the product of
 * bags, not hand-written full sentences.
 */
import type { SpeechAct } from "../../../src/model/speech-acts";

export const EMPTY_ID = "empty";

export type HolePosition = "left" | "mid" | "right";

export interface BagPart {
  id: string;
  text: string;
  /** Locative preposition for scene NPs (`at` / `in` / `on` / `by` / `with`). */
  prep?: string;
}

export interface SpeechFrame {
  id: string;
  act: SpeechAct;
  prefix?: string;
  suffix?: string;
  quoted?: boolean;
}

export const EMPTY_PART: BagPart = { id: EMPTY_ID, text: "" };

/** Empty speech still names a request: the kernel itself is the ask. */
export const EMPTY_SPEECH: SpeechFrame = { id: EMPTY_ID, act: "request" };

export const POSITIONS: readonly HolePosition[] = ["left", "mid", "right"];

export interface SlotDraw {
  speech: SpeechFrame;
  scene: BagPart;
  hedge: BagPart;
  reason: BagPart;
  tail: BagPart;
  position: HolePosition;
}

export function holeActionableFor(act: SpeechAct): boolean {
  return act === "request" || act === "correction";
}
