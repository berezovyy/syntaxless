/** Internal build entry used by the artifact promotion gate.
 *
 * This file is not part of the package export map. It exposes the exact
 * code-split tokenizer and inference functions used by the public parser so a
 * candidate weights module can be evaluated without a second implementation.
 */
export { inferTokens } from "./inference.js";
export { LABELS } from "./labels.js";
export { tokenize } from "./tokenizer.js";
export { weights } from "./weights.gen.js";

import { decodeIntent } from "../parser/decode.js";
import type { OverlayMode } from "../parser/decode.js";
import { normalize } from "../parser/normalize.js";
import { inferTokens } from "./inference.js";
import { SCHEDULE_KINDS } from "./schedule-kinds.js";
import { speechActFromId } from "./speech-acts.js";
import { tokenize } from "./tokenizer.js";

/** Decode before the public confidence floor. Internal evaluation only: this
 * exposes whether a rejection hid a correct proposal without weakening the
 * public parser's fail-closed contract. Public decode is heads-only: a missing
 * speech-act head fails closed rather than falling back to overlay regex. */
export function decodeRaw(text: string, overlay: OverlayMode = "off") {
  const tokens = tokenize(normalize(text));
  const predictions = inferTokens(tokens);

  const scheduleKind =
    predictions.scheduleKind === undefined || predictions.scheduleKindScore === undefined
      ? undefined
      : {
          kind: SCHEDULE_KINDS[predictions.scheduleKind]!,
          score: predictions.scheduleKindScore,
        };

  if (predictions.speechAct === undefined) {
    throw new RangeError("decodeRaw: heads-only requires a speech-act head");
  }

  const speechAct = speechActFromId(predictions.speechAct);

  return decodeIntent(
    tokens.map((token, index) => ({
      token,
      role: predictions.labels[index]!,
      boundary: predictions.boundaries[index]! === 1,
      score: predictions.scores[index]!,
      boundaryScore: predictions.boundaryScores[index]!,
      ...(predictions.actionable === undefined
        ? {}
        : { actionable: predictions.actionable[index] === 1 }),
    })),
    scheduleKind,
    text,
    overlay,
    speechAct,
  );
}
