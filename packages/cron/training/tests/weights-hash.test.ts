import { expect, test } from "bun:test";

import { weights } from "../../src/model/weights.gen";
import { artifactSha256 } from "../src/weights-hash";

test("weight identity covers vocabulary, layout, scales and calibration", () => {
  const original = artifactSha256(weights);

  for (const changed of [
    { ...weights, labels: [...weights.labels].reverse() },
    { ...weights, boundaryThreshold: weights.boundaryThreshold + 0.1 },
    { ...weights, energyThreshold: 123 },
    { ...weights, mahalanobisMean: [1, 2, 3] },
    {
      ...weights,
      segments: weights.segments.map((segment, index) =>
        index === 0 ? { ...segment, scale: segment.scale * 2 } : segment,
      ),
    },
    { ...weights, segments: [...weights.segments].reverse() },
  ]) {
    expect(changed.q).toBe(weights.q);
    expect(artifactSha256(changed)).not.toBe(original);
  }

  const reordered = Object.fromEntries(Object.entries(weights).reverse());

  expect(artifactSha256(reordered as unknown as typeof weights)).toBe(original);
});
