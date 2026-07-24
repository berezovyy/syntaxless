import { createInference as create } from "../../../shared/model/inference.js";
import type { WeightsArtifact } from "./decode.js";
import { LABELS } from "./labels.js";

export type { Predictions } from "../../../shared/model/inference.js";

export function createInference(artifact: WeightsArtifact) {
  return create(artifact, LABELS);
}
