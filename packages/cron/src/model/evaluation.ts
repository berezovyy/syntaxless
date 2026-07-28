/** Internal build entry for scoring artifacts with the shipped parser and compiler. */
import { compile } from "../compiler/cron.js";
import { decodeIntent } from "../parser/decode.js";
import { labelTokensFromPredictions, parsePrediction } from "../parser/model.js";
import { normalize } from "../parser/normalize.js";
import type { WeightsArtifact } from "./decode.js";
import { createInference } from "./inference.js";
import { LABELS } from "./labels.js";
import { SCHEDULE_KINDS } from "./schedule-kinds.js";
import { tokenize } from "./tokenizer.js";
import { weights } from "./weights.gen.js";

export { compile, LABELS, tokenize, weights };

/** One artifact supplies inference, calibrated parsing, and raw proposal decoding. */
export function createEvaluation(artifact: WeightsArtifact) {
  const engine = createInference(artifact);

  return {
    weights: artifact,
    LABELS,
    tokenize,
    compile,
    inferTokens: engine.inferTokens,
    predict(text: string) {
      const normalized = normalize(text);
      const tokens = tokenize(normalized);
      const predictions = engine.inferTokens(tokens, true);
      const labeled = labelTokensFromPredictions(tokens, predictions);

      const scheduleKind =
        predictions.scheduleKind === undefined || predictions.scheduleKindScore === undefined
          ? undefined
          : {
              kind: SCHEDULE_KINDS[predictions.scheduleKind]!,
              score: predictions.scheduleKindScore,
            };

      return {
        normalized,
        tokens,
        predictions,
        parse: () => parsePrediction(normalized, engine, predictions, labeled),
        // Raw inspection uses the full speech head and original text, before calibration.
        decodeRaw() {
          if (predictions.speechAct === undefined) {
            throw new RangeError("decodeRaw: heads-only requires a speech-act head");
          }

          return decodeIntent(
            labeled,
            scheduleKind,
            text,
            engine.predictedSpeechAct(predictions.speechAct),
          );
        },
      };
    },
  };
}
