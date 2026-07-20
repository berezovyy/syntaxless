/** Fingerprint the wrapper's speech act, placement and context. */
import type { SlotDraw } from "./types";

export function slotFingerprint(draw: SlotDraw): string {
  return [
    draw.speech.act,
    draw.position,
    draw.scene.id,
    draw.reason.id,
    draw.hedge.id,
    draw.tail.id,
  ].join("|");
}
