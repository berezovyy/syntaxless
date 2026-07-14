/**
 * Concatenate per-kind featurize-kinds prefixes into one Dataset prefix.
 * Offsets and frame ids are rebased; other bins are byte-appended.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";

import { LABEL_COUNT, LABELS } from "../../src/model/labels.js";
import { SPEECH_ACT_COUNT, SPEECH_ACTS } from "../../src/model/speech-acts.js";
import { EMBEDDING_ROWS, FEATURE_COUNT } from "../../src/model/tokenizer.js";
import { SHARD_FORMAT } from "./featurize";
import type { FeaturizeManifest } from "./featurize";

const SIMPLE_KEYS = [
  "rows",
  "labels",
  "boundaries",
  "kinds",
  "neighbors",
  "speech",
  "actionable",
] as const;

function loadManifest(prefix: string): FeaturizeManifest {
  return JSON.parse(readFileSync(`${prefix}.json`, "utf8")) as FeaturizeManifest;
}

function concatBuffers(paths: string[]): Buffer {
  const parts = paths.filter((path) => existsSync(path)).map((path) => readFileSync(path));

  return parts.length === 0 ? Buffer.alloc(0) : Buffer.concat(parts);
}

function readUint32(path: string): Uint32Array {
  if (!existsSync(path)) return new Uint32Array();
  const buf = readFileSync(path);

  return new Uint32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

export function concatKindPrefixes(prefixes: readonly string[], out: string): void {
  const present = prefixes.filter((prefix) => existsSync(`${prefix}.json`));

  if (present.length === 0) throw new RangeError(`concat: no kind shards for ${out}`);

  const manifests = present.map(loadManifest);
  const fingerprints = [...new Set(manifests.flatMap((manifest) => manifest.fingerprints))].sort();
  const labelCounts: Record<string, number> = Object.fromEntries(LABELS.map((label) => [label, 0]));
  const frameNames: string[] = [];
  const frameCounts: number[] = [];
  const frameBase: number[] = [];
  const actionableCounts: Record<0 | 1, number> = { 0: 0, 1: 0 };
  let framesSoFar = 0;

  for (const manifest of manifests) {
    frameBase.push(framesSoFar);

    for (let index = 0; index < manifest.frameNames.length; index++) {
      frameNames.push(manifest.frameNames[index]!);
      frameCounts.push(manifest.frameCounts[index] ?? 0);
    }

    framesSoFar += manifest.frameNames.length;
    for (const label of LABELS) labelCounts[label]! += manifest.labelCounts[label] ?? 0;
    actionableCounts[0] += manifest.actionableCounts[0] ?? 0;
    actionableCounts[1] += manifest.actionableCounts[1] ?? 0;
  }

  for (const key of SIMPLE_KEYS) {
    const suffix = SHARD_FORMAT[key].suffix;

    writeFileSync(`${out}${suffix}`, concatBuffers(present.map((prefix) => `${prefix}${suffix}`)));
  }

  const offsetChunks: number[] = [0];
  let tokens = 0;

  for (const prefix of present) {
    const offsets = readUint32(`${prefix}${SHARD_FORMAT.offsets.suffix}`);

    for (let index = 1; index < offsets.length; index++) {
      offsetChunks.push(tokens + offsets[index]!);
    }

    tokens += offsets.length > 0 ? offsets[offsets.length - 1]! : 0;
  }

  writeFileSync(
    `${out}${SHARD_FORMAT.offsets.suffix}`,
    Buffer.from(new Uint32Array(offsetChunks).buffer),
  );

  const frameParts: number[] = [];

  for (let shard = 0; shard < present.length; shard++) {
    const frames = readUint32(`${present[shard]!}${SHARD_FORMAT.frames.suffix}`);
    const base = frameBase[shard]!;

    for (const id of frames) frameParts.push(id + base);
  }

  writeFileSync(
    `${out}${SHARD_FORMAT.frames.suffix}`,
    Buffer.from(new Uint32Array(frameParts).buffer),
  );

  const merged: FeaturizeManifest = {
    fingerprints,
    roleClasses: manifests[0]?.roleClasses ?? LABEL_COUNT,
    rowsPerToken: manifests[0]?.rowsPerToken ?? FEATURE_COUNT,
    featureRows: manifests[0]?.featureRows ?? EMBEDDING_ROWS,
    labelCounts,
    frameNames,
    frameCounts,
    speechActClasses: manifests[0]?.speechActClasses ?? SPEECH_ACT_COUNT,
    speechActNames: manifests[0]?.speechActNames ?? SPEECH_ACTS,
    actionableCounts: { 0: actionableCounts[0], 1: actionableCounts[1] },
    format: SHARD_FORMAT,
  };

  writeFileSync(`${out}.json`, `${JSON.stringify(merged, null, 2)}\n`);
}
