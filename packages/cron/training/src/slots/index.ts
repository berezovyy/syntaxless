export { compose, locativeScene, type ComposeResult } from "./compose";

export {
  FINGERPRINT_MAX_SHARE,
  exceedsFingerprintCap,
  fingerprintReport,
  slotFingerprint,
} from "./fingerprint";

export { allBagTexts, humanEvalTexts, intraBagTrigramCollisions, leakHits } from "./leak";

export {
  SLOT_WRAPS_PER_KERNEL,
  sampleFingerprints,
  sampleSlotDraw,
  seedFrom,
  wrapsForKernel,
} from "./sample";

export { HEDGE_PARTS } from "./hedge";
export { REASON_PARTS } from "./reason";
export { SCENE_PARTS } from "./scene";
export { SPEECH_FRAMES } from "./speech";
export { TAIL_PARTS } from "./tail";

export {
  EMPTY_ID,
  EMPTY_PART,
  EMPTY_SPEECH,
  POSITIONS,
  holeActionableFor,
  type BagPart,
  type HolePosition,
  type SlotDraw,
  type SpeechFrame,
} from "./types";
