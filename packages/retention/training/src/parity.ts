/** Compare the actual exported TypeScript engine with frozen PyTorch outputs. */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { getEngine } from "../../src/model/engine.js";
import { LABEL_COUNT } from "../../src/model/labels.js";
import { FEATURE_COUNT, PADDING_ROW } from "../../src/model/tokenizer.js";

function binary<T>(path: string, ctor: new (buffer: ArrayBuffer) => T): T {
  const bytes = readFileSync(path);

  return new ctor(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
  );
}

export async function verifyParity(candidate: string) {
  const artifact = readFileSync(join(candidate, "weights.json"));
  const engine = await getEngine({ weights: artifact });
  const rows = binary(join(candidate, "parity/parity.rows.bin"), Uint16Array);
  const offsets = binary(join(candidate, "parity/parity.offsets.bin"), Uint32Array);
  const logits = binary(join(candidate, "parity/parity.logits.bin"), Float32Array);
  const boundaries = binary(join(candidate, "parity/parity.boundaries.bin"), Float32Array);
  const roleClasses = LABEL_COUNT;

  if (
    rows.length !== offsets.at(-1)! * FEATURE_COUNT ||
    logits.length !== offsets.at(-1)! * roleClasses ||
    boundaries.length !== offsets.at(-1)
  ) {
    throw new Error("parity fixture geometry mismatch");
  }

  let roleError = 0;
  let boundaryError = 0;
  let paddingError = 0;

  for (let i = 0; i < offsets.length - 1; i++) {
    const start = offsets[i]!;
    const end = offsets[i + 1]!;
    const count = end - start;

    const input = rows.slice(start * FEATURE_COUNT, end * FEATURE_COUNT);
    const result = engine.inferRows(input, count, true);
    const padded = new Uint16Array(input.length + FEATURE_COUNT * 3).fill(PADDING_ROW);

    padded.set(input);
    const extra = engine.inferRows(padded, count, true);

    for (let j = 0; j < count * roleClasses; j++) {
      roleError = Math.max(
        roleError,
        Math.abs(result.logits![j]! - logits[start * roleClasses + j]!),
      );

      paddingError = Math.max(paddingError, Math.abs(result.logits![j]! - extra.logits![j]!));
    }

    for (let j = 0; j < count; j++) {
      boundaryError = Math.max(
        boundaryError,
        Math.abs(result.boundaryLogits![j]! - boundaries[start + j]!),
      );
    }
  }

  if (
    ![roleError, boundaryError, paddingError].every(
      (error) => Number.isFinite(error) && error <= 1e-4,
    )
  ) {
    throw new Error(
      `inference parity failed: ${JSON.stringify({ roleError, boundaryError, paddingError })}`,
    );
  }

  const report = {
    sequences: offsets.length - 1,
    roleError,
    boundaryError,
    paddingError,
    passed: true,
  };

  writeFileSync(join(candidate, "parity-report.json"), JSON.stringify(report, null, 2) + "\n");

  return report;
}

if (import.meta.main) console.log(await verifyParity(process.argv[2]!));
