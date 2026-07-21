/** Private binary contract shared by the bounded tagger trainers. */
import { closeSync, mkdirSync, openSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

export const SHARD_FORMAT = {
  rows: { suffix: ".rows.bin", dtype: "uint16" },
  labels: { suffix: ".labels.bin", dtype: "uint8" },
  boundaries: { suffix: ".boundaries.bin", dtype: "uint8" },
  kinds: { suffix: ".kinds.bin", dtype: "uint8" },
  neighbors: { suffix: ".neighbors.bin", dtype: "int16" },
  offsets: { suffix: ".offsets.bin", dtype: "uint32" },
  frames: { suffix: ".frames.bin", dtype: "uint32" },
} as const;

export interface Sequence {
  rows: Uint16Array;
  labels: Uint8Array;
  boundaries: Uint8Array;
  kinds: Uint8Array;
  neighbors: Int16Array;
  fingerprint: string;
  frame: string;
}

export interface Vocabulary {
  labels: readonly string[];
  paddingLabel: number;
  rowsPerToken: number;
  featureRows: number;
}

export function writeShards(prefix: string, sequences: Iterable<Sequence>, vocabulary: Vocabulary) {
  const capacity = 8192;

  const buffers = {
    rows: new Uint16Array(capacity * vocabulary.rowsPerToken),
    labels: new Uint8Array(capacity),
    boundaries: new Uint8Array(capacity),
    kinds: new Uint8Array(capacity),
    neighbors: new Int16Array(capacity * 2),
  };

  const widths = {
    rows: vocabulary.rowsPerToken,
    labels: 1,
    boundaries: 1,
    kinds: 1,
    neighbors: 2,
  };

  const files: Partial<Record<keyof typeof buffers, number>> = {};

  const offsets = [0];
  const frames: number[] = [];
  const fingerprints = new Set<string>();

  const frameNames: string[] = [];
  const frameCounts: number[] = [];
  const frameIds = new Map<string, number>();

  const labelCounts = Object.fromEntries(vocabulary.labels.map((label) => [label, 0]));
  let buffered = 0;

  function flush() {
    if (!buffered) return;

    for (const name of Object.keys(buffers) as (keyof typeof buffers)[]) {
      const view = buffers[name].subarray(0, buffered * widths[name]);

      writeFileSync(files[name]!, Buffer.from(view.buffer, view.byteOffset, view.byteLength));
    }

    buffered = 0;
  }

  mkdirSync(dirname(prefix), { recursive: true });

  try {
    for (const name of Object.keys(buffers) as (keyof typeof buffers)[]) {
      files[name] = openSync(prefix + SHARD_FORMAT[name].suffix, "wx");
    }

    for (const sequence of sequences) {
      if (frames.length === 100000) throw Error("training corpus exceeds 100000 sequences");
      const count = sequence.labels.length;

      if (
        !count ||
        count > 128 ||
        sequence.rows.length !== count * vocabulary.rowsPerToken ||
        sequence.neighbors.length !== count * 2 ||
        sequence.kinds.length !== count ||
        sequence.boundaries.length !== count
      ) {
        throw Error("invalid training sequence dimensions");
      }

      if (buffered + count > capacity) flush();

      for (let index = 0; index < count; index++) {
        const label = sequence.labels[index]!;

        if (sequence.kinds[index] === 3) {
          if (label !== vocabulary.paddingLabel) {
            throw Error("whitespace requires the padding label");
          }
        } else {
          const name = vocabulary.labels[label];

          if (name === undefined) throw Error("unknown or missing training role");
          labelCounts[name]!++;
        }
      }

      for (const name of Object.keys(buffers) as (keyof typeof buffers)[]) {
        buffers[name].set(sequence[name], buffered * widths[name]);
      }

      buffered += count;
      offsets.push(offsets.at(-1)! + count);
      fingerprints.add(sequence.fingerprint);
      let frame = frameIds.get(sequence.frame);

      if (frame === undefined) {
        frame = frameNames.length;
        frameIds.set(sequence.frame, frame);
        frameNames.push(sequence.frame);
        frameCounts.push(0);
      }

      frames.push(frame);
      frameCounts[frame]!++;
    }

    flush();
  } finally {
    for (const file of Object.values(files)) closeSync(file);
  }

  writeFileSync(
    prefix + SHARD_FORMAT.offsets.suffix,
    Buffer.from(Uint32Array.from(offsets).buffer),
    { flag: "wx" },
  );

  writeFileSync(prefix + SHARD_FORMAT.frames.suffix, Buffer.from(Uint32Array.from(frames).buffer), {
    flag: "wx",
  });

  const manifest = {
    fingerprints: [...fingerprints].sort(),
    roleClasses: vocabulary.labels.length,
    rowsPerToken: vocabulary.rowsPerToken,
    featureRows: vocabulary.featureRows,
    labelCounts,
    frameNames,
    frameCounts,
    format: SHARD_FORMAT,
  };

  writeFileSync(prefix + ".json", JSON.stringify(manifest, null, 2) + "\n", {
    flag: "wx",
  });

  return { sequences: frames.length, tokens: offsets.at(-1)!, manifest };
}
