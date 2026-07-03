import type { WeightsArtifact } from "./decode.js";
/**
 * Model asset policy.
 *
 * The default bytes are the bundled Phase 15 `weights.gen.ts` artifact, so the
 * package works offline with no third-party CDN. A caller that self-hosts the
 * weights can override the default with `config.weights`, a `Uint8Array` that
 * decodes to a {@link WeightsArtifact}. Every resolved artifact is checked
 * against {@link WEIGHTS_FORMAT_VERSION}; a mismatch is a hard error and never
 * a silent mis-decode.
 *
 * Data only: no session, no fetch, no browser globals, no top-level await.
 */
import { weights as bundledWeights } from "./weights.gen.js";

/** Version of the weight artifact this runtime understands. */
export const WEIGHTS_FORMAT_VERSION = 1;

/**
 * A resolved weight artifact does not match the runtime's supported format.
 * The error name is the machine-readable code and the message names both the
 * expected and the received version.
 */
export class UnsupportedWeightsVersionError extends Error {
  readonly expected: number;
  readonly actual: number;

  constructor(expected: number, actual: number) {
    super(
      `unsupported-weights-version: runtime expects ${expected} ` +
        `but the artifact declares ${actual}`,
    );

    this.name = "unsupported-weights-version";
    this.expected = expected;
    this.actual = actual;
  }
}

/** Throw {@link UnsupportedWeightsVersionError} when the artifact is not understood. */
export function assertFormatVersion(artifact: { version: number }): void {
  if (artifact.version !== WEIGHTS_FORMAT_VERSION) {
    throw new UnsupportedWeightsVersionError(WEIGHTS_FORMAT_VERSION, artifact.version);
  }
}

/**
 * Decode caller-provided self-hosted bytes. The canonical byte form of an
 * artifact is its UTF-8 JSON representation, so a self-hoster can serve the
 * same object the generator emits.
 */
export function decodeWeightsBytes(bytes: Uint8Array): WeightsArtifact {
  let parsed: unknown;

  try {
    parsed = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new TypeError("weights bytes are not valid UTF-8 JSON");
  }

  if (parsed === null || typeof parsed !== "object") {
    throw new TypeError("weights bytes must decode to an artifact object");
  }

  const artifact = parsed as Partial<WeightsArtifact>;

  if (typeof artifact.version !== "number") {
    throw new TypeError("weights artifact is missing a numeric version");
  }

  return artifact as WeightsArtifact;
}

/**
 * Resolve the artifact to load: the bundled default, or the caller's bytes.
 * The format version is validated here so a mismatch fails before any decode
 * work happens.
 */
export function resolveWeights(override?: Uint8Array): WeightsArtifact {
  const artifact = override === undefined ? bundledWeights : decodeWeightsBytes(override);

  assertFormatVersion(artifact);

  return artifact;
}
