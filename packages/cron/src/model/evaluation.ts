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
import { normalize } from "../parser/normalize.js";
import { inferTokens } from "./inference.js";
import { SCHEDULE_KINDS } from "./schedule-kinds.js";
import { tokenize } from "./tokenizer.js";

/** Decode before the public confidence floor. Internal evaluation only: this
 * exposes whether a rejection hid a correct proposal without weakening the
 * public parser's fail-closed contract. */
export function decodeRaw(text: string) {
  const tokens = tokenize(normalize(text));
  const predictions = inferTokens(tokens);

  const scheduleKind =
    predictions.scheduleKind === undefined || predictions.scheduleKindScore === undefined
      ? undefined
      : {
          kind: SCHEDULE_KINDS[predictions.scheduleKind]!,
          score: predictions.scheduleKindScore,
        };

  return decodeIntent(
    tokens.map((token, index) => ({
      token,
      role: predictions.labels[index]!,
      boundary: predictions.boundaries[index]! === 1,
      score: predictions.scores[index]!,
      boundaryScore: predictions.boundaryScores[index]!,
    })),
    scheduleKind,
    text,
  );
}
