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

function shapeSize(shape: readonly number[]): number {
  return shape.reduce((product, dimension) => product * dimension, 1);
}

describe("generated weights artifact", () => {
  test("segments tile the q payload without gaps or overlaps", () => {
    let covered = 0;

    for (const segment of artifact.segments) {
      expect(segment.offset).toBe(covered);
      expect(segment.length).toBe(shapeSize(segment.shape));
      covered += segment.length;
    }

    expect(covered).toBe(artifact.q.length);
  });

  test("every decoded tensor has its declared shape", () => {
    const decoded = decodeWeights(artifact);

    expect(decoded.size).toBe(artifact.segments.length);

    for (const segment of artifact.segments) {
      const tensor = decoded.get(segment.name);

      expect(tensor).toBeInstanceOf(Float32Array);
      expect(tensor).toBeDefined();
      expect(tensor!.length).toBe(shapeSize(segment.shape));
      expect(tensor!.length).toBe(segment.length);
    }
  });

  test("decoded values stay inside the int6 range for their scale", () => {
    const decoded = decodeWeights(artifact);

    for (const segment of artifact.segments) {
      for (const value of decoded.get(segment.name)!) {
        expect(Math.abs(value)).toBeLessThanOrEqual(31 * segment.scale + 1e-6);
      }
    }
  });
});

describe("decodeWeights rejects malformed artifacts", () => {
  test("out-of-bounds offset", () => {
    const malformed = cloneArtifact();

    malformed.segments[0]!.offset = malformed.q.length + 1;
    expect(() => decodeWeights(malformed)).toThrow(RangeError);
  });

  test("unknown alphabet symbol", () => {
    const malformed = cloneArtifact();

    malformed.q = `!${malformed.q.slice(1)}`;
    expect(() => decodeWeights(malformed)).toThrow(RangeError);
  });

  test("shape that does not match its length", () => {
    const malformed = cloneArtifact();
    const first = malformed.segments[0]!;

    first.shape = [first.length + 1];
    expect(() => decodeWeights(malformed)).toThrow(RangeError);
  });

  test("non-positive scale", () => {
    const malformed = cloneArtifact();

    malformed.segments[0]!.scale = 0;
    expect(() => decodeWeights(malformed)).toThrow(RangeError);
  });

  test("duplicate segment name", () => {
    const malformed = cloneArtifact();

    malformed.segments[1]!.name = malformed.segments[0]!.name;
    expect(() => decodeWeights(malformed)).toThrow();
  });

  test("unsupported storage", () => {
    const malformed = cloneArtifact();

    malformed.storage = "raw";
    expect(() => decodeWeights(malformed)).toThrow(RangeError);
  });
});
