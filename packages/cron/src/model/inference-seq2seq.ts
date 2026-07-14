/**
 * CPU greedy decode under the Track B grammar mask.
 * Research branch until the 100k prototype beats the tagger.
 */
import { greedyStep } from "./grammar.js";
import type { GrammarToken } from "./grammar.js";

export function greedyDecode(logitsSteps: readonly (readonly number[])[]): GrammarToken[] {
  const prefix: GrammarToken[] = [];

  for (const logits of logitsSteps) {
    const next = greedyStep(prefix, logits);

    prefix.push(next);
    if (next === ")") break;
  }

  return prefix;
}
