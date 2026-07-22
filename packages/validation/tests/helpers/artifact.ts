import type { WeightsArtifact, WeightSegment } from "../../src/model/decode.js";
/** A small numerical fixture, independent of active model quality or local runs. */
import { LABELS } from "../../src/model/labels.js";
import { EMBEDDING_ROWS, FEATURE_COUNT } from "../../src/model/tokenizer.js";

export function fixtureArtifact(): WeightsArtifact {
  const shapes: Record<string, number[]> = {
    "embedding.weight": [EMBEDDING_ROWS + 1, 1],
    "depthwise.weight": [1, 1, 3],
    "depthwise.bias": [1],
    "neighbor.weight": [1, 1],
    "neighbor.bias": [1],
    "combine.weight": [1, 2],
    "combine.bias": [1],
    "context_norm.weight": [1],
    "context_norm.bias": [1],
    "head_gate.weight": [1, 2],
    "head_gate.bias": [1],
    "role_head.weight": [LABELS.length, 1],
    "role_head.bias": [LABELS.length],
    "boundary_head.weight": [1, 1],
    "boundary_head.bias": [1],
  };

  for (const direction of ["forward", "backward"]) {
    for (const layer of ["proposal", "gate"]) {
      shapes[`${direction}_recurrence.${layer}.weight`] = [1, 1];
      shapes[`${direction}_recurrence.${layer}.bias`] = [1];
    }
  }

  let offset = 0;

  const segments: WeightSegment[] = Object.entries(shapes).map(([name, shape]) => {
    const length = shape.reduce((a, b) => a * b, 1);
    const segment = { name, shape, length, offset, scale: 0.1 };

    offset += length;

    return segment;
  });

  return {
    version: 2,
    hidden: 1,
    featureRows: EMBEDDING_ROWS,
    rowsPerToken: FEATURE_COUNT,
    storage: "alphabet",
    roleClasses: LABELS.length,
    labels: LABELS,
    boundaryThreshold: 0,
    q: "f".repeat(offset),
    segments,
  };
}
