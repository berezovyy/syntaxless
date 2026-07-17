import { createHash } from "node:crypto";

import type { WeightsArtifact } from "../../src/model/decode";

/**
 * Identity of the entire wire artifact, independent of JSON spelling/key order.
 * Keep this encoding in sync with artifact_sha256 in training/cron/export.py:
 * domain prefix; z/null, t/true, f/false, n/float64 BE, s/UTF-8, a/array, o/object.
 * Strings and containers have an ASCII byte/item count followed by ':'. Object
 * keys sort by UTF-8 bytes; array order is retained. Signed zero becomes +0.
 * This runs only in tooling, never in the dependency-free browser runtime.
 */
export function artifactSha256(artifact: WeightsArtifact): string {
  const digest = createHash("sha256").update("syntaxless-weights-v1\n");
  const number = Buffer.allocUnsafe(8);

  function hash(value: unknown): void {
    if (value === null) {
      digest.update("z");
    } else if (typeof value === "boolean") {
      digest.update(value ? "t" : "f");
    } else if (typeof value === "number") {
      if (!Number.isFinite(value)) throw new RangeError("Non-finite weight metadata");
      number.writeDoubleBE(value === 0 ? 0 : value);
      digest.update("n").update(number);
    } else if (typeof value === "string") {
      const bytes = Buffer.from(value, "utf8");

      digest.update(`s${bytes.length}:`).update(bytes);
    } else if (Array.isArray(value)) {
      digest.update(`a${value.length}:`);
      for (const item of value) hash(item);
    } else if (typeof value === "object") {
      const entries = Object.entries(value).sort(([a], [b]) =>
        Buffer.compare(Buffer.from(a, "utf8"), Buffer.from(b, "utf8")),
      );

      digest.update(`o${entries.length}:`);

      for (const [key, item] of entries) {
        hash(key);
        hash(item);
      }
    } else {
      throw new TypeError("Weight artifacts must contain only JSON values");
    }
  }

  hash(artifact);

  return digest.digest("hex");
}
