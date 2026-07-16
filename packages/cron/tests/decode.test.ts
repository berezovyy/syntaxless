import { describe, expect, test } from "bun:test";

import { decodeWeights } from "../src/model/decode.ts";
import type { WeightsArtifact } from "../src/model/decode.ts";
import { weights } from "../src/model/weights.gen.ts";

interface MutableSegment {
  name: string;
  offset: number;
  length: number;
  shape: number[];
  scale: number;
}

interface MutableArtifact {
  version: number;
  hidden: number;
  featureRows: number;
  rowsPerToken: number;
  storage: string;
  roleClasses: number;
  boundaryThreshold: number;
  labels: string[];
  q: string;
  segments: MutableSegment[];
}

const artifact: WeightsArtifact = weights;

function cloneArtifact(): MutableArtifact {
  return structuredClone(artifact) as unknown as MutableArtifact;
}

describe("decodeWeights rejects malformed artifacts", () => {
  test("unknown alphabet symbol", () => {
    const malformed = cloneArtifact();

    malformed.q = `!${malformed.q.slice(1)}`;
    expect(() => decodeWeights(malformed)).toThrow(RangeError);
  });

  test("finite scales cannot produce non-finite Float32 tensors", () => {
    for (const q of ["+", "A"]) {
      const malformed = {
        ...artifact,
        q,
        segments: [{ name: "overflow", offset: 0, length: 1, shape: [1], scale: 1e38 }],
      };

      expect(() => decodeWeights(malformed)).toThrow("Float32");
    }
  });
});
