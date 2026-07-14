/**
 * Concatenate per-kind featurize-kinds prefixes into one Dataset prefix.
 * Offsets and frame ids are rebased; other bins are byte-appended.
 */
import {
  closeSync,
  existsSync,
  openSync,
  readFileSync,
  readSync,
  writeFileSync,
  writeSync,
} from "node:fs";

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
  "populations",
] as const;

function loadManifest(prefix: string): FeaturizeManifest {
  return JSON.parse(readFileSync(`${prefix}.json`, "utf8")) as FeaturizeManifest;
}

const COPY_CHUNK = 64 * 1024 * 1024;

/** Byte-append `paths` into `out` in 64 MiB chunks; the uncapped pool's
 * rows.bin is ~18 GB, so the files must never be resident at once. */
function appendFiles(paths: string[], out: string): void {
  const dst = openSync(out, "w");
  const chunk = Buffer.allocUnsafe(COPY_CHUNK);

  try {
    for (const path of paths) {
      if (!existsSync(path)) continue;
      const src = openSync(path, "r");

      try {
        let read: number;

        while ((read = readSync(src, chunk, 0, COPY_CHUNK, null)) > 0) {
          writeSync(dst, chunk, 0, read);
        }
      } finally {
        closeSync(src);
      }
    }
  } finally {
    closeSync(dst);
  }
}

function readUint32(path: string): Uint32Array {
  if (!existsSync(path)) return new Uint32Array();
  const buf = readFileSync(path);

  return new Uint32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

/** Rebase per-kind uint32 shards (offsets, frame ids) into `out`, one kind
 * at a time, so the merged array is streamed rather than built in memory. */
function writeRebasedUint32(
  paths: string[],
  out: string,
  rebase: (values: Uint32Array, shard: number) => Uint32Array,
  header?: Uint32Array,
): void {
  const dst = openSync(out, "w");

  try {
    if (header !== undefined && header.length > 0) {
      writeSync(dst, Buffer.from(header.buffer, header.byteOffset, header.byteLength));
    }

    for (let shard = 0; shard < paths.length; shard++) {
      const values = rebase(readUint32(paths[shard]!), shard);

      if (values.length > 0) {
        writeSync(dst, Buffer.from(values.buffer, values.byteOffset, values.byteLength));
      }
    }
  } finally {
    closeSync(dst);
  }
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
  const populationCounts = { R: 0, H: 0, E: 0 };
  const hasPopulations = manifests.every((manifest) => manifest.format.populations !== undefined);
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

    for (const population of ["R", "H", "E"] as const) {
      populationCounts[population] += manifest.populationCounts?.[population] ?? 0;
    }
  }

  for (const key of SIMPLE_KEYS) {
    if (key === "populations" && !hasPopulations) continue;
    const suffix = SHARD_FORMAT[key].suffix;

    appendFiles(
      present.map((prefix) => `${prefix}${suffix}`),
      `${out}${suffix}`,
    );
  }

  // offsets: [0, ...] with every kind's offsets shifted by the tokens before it.
  let tokens = 0;

  writeRebasedUint32(
    present.map((prefix) => `${prefix}${SHARD_FORMAT.offsets.suffix}`),
    `${out}${SHARD_FORMAT.offsets.suffix}`,
    (offsets) => {
      const shifted = new Uint32Array(Math.max(0, offsets.length - 1));

      for (let index = 1; index < offsets.length; index++) {
        shifted[index - 1] = tokens + offsets[index]!;
      }

      tokens += offsets.length > 0 ? offsets[offsets.length - 1]! : 0;

      return shifted;
    },
    new Uint32Array([0]),
  );

  // frames: per-kind frame ids shifted by the kind's frame base.
  writeRebasedUint32(
    present.map((prefix) => `${prefix}${SHARD_FORMAT.frames.suffix}`),
    `${out}${SHARD_FORMAT.frames.suffix}`,
    (frames, shard) => {
      const base = frameBase[shard]!;
      const shifted = new Uint32Array(frames.length);

      for (let index = 0; index < frames.length; index++) shifted[index] = frames[index]! + base;

      return shifted;
    },
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
    ...(hasPopulations ? { populationCounts } : {}),
    format: hasPopulations
      ? SHARD_FORMAT
      : (Object.fromEntries(
          Object.entries(SHARD_FORMAT).filter(([key]) => key !== "populations"),
        ) as FeaturizeManifest["format"]),
  };

  writeFileSync(`${out}.json`, `${JSON.stringify(merged, null, 2)}\n`);
}
