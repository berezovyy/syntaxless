import { expect, test } from "bun:test";

import { LABELS } from "../src/model/labels.ts";
import { fixtureArtifact } from "./helpers/artifact.ts";

test("legacy formats and reordered vocabularies fail before inference", async () => {
  const { getEngine } = await import("../src/model/engine.ts");
  const legacy = { ...fixtureArtifact(), version: 1 };

  await expect(
    getEngine({ weights: new TextEncoder().encode(JSON.stringify(legacy)) }),
  ).rejects.toThrow("unsupported-weights-version");

  const reordered = { ...fixtureArtifact(), labels: [...LABELS].reverse() };

  await expect(
    getEngine({ weights: new TextEncoder().encode(JSON.stringify(reordered)) }),
  ).rejects.toThrow("label vocabulary");
});
