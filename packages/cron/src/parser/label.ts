/**
 * Synchronous labeling helper.
 *
 * This is the only module in `src/parser` that statically imports the inference
 * engine, so it is deliberately kept out of the public entry's import graph.
 * Callers that want lazy initialization use {@link parseWithModel} instead.
 */
import { inferTokens } from "../model/inference.js";
import type { RawToken } from "../model/tokenizer.js";
import type { LabeledToken } from "./decode.js";
import { labelTokensFromPredictions } from "./model.js";

/** Run one forward pass and build the decoder's input. */
export function labelTokens(tokens: readonly RawToken[]): LabeledToken[] {
  return labelTokensFromPredictions(tokens, inferTokens(tokens));
}
