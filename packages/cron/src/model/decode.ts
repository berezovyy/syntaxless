/**
 * Typed-array decoder for the hash-verified int6 weight artifact.
 *
 * The generated `weights.gen.ts` module is a pure data literal. This decoder
 * validates its layout and turns the packed symbol string into one
 * `Float32Array` per parameter. A malformed artifact throws; it never silently
 * substitutes zeros or skips a tensor.
 */

export interface WeightSegment {
  readonly name: string;
  readonly offset: number;
  readonly length: number;
  readonly shape: readonly number[];
  readonly scale: number;
}

export interface WeightsArtifact {
  readonly version: number;
  readonly hidden: number;
  readonly featureRows: number;
  readonly rowsPerToken: number;
  readonly storage: string;
  readonly roleClasses: number;
  readonly scheduleKindClasses?: number;
  readonly scheduleKindLabels?: readonly string[];
  readonly speechActClasses?: number;
  readonly speechActLabels?: readonly string[];
  readonly boundaryThreshold: number;
  readonly energyThreshold?: number;
  readonly mahalanobisMean?: readonly number[] | number;
  readonly mahalanobisVariance?: readonly number[] | number;
  readonly mahalanobisThreshold?: number;
  readonly labels: readonly string[];
  readonly q: string;
  readonly segments: readonly WeightSegment[];
}

/** int6 symbols encode `value + SYMBOL_BIAS`; 0..62 covers -31..31. */
const SYMBOL_BIAS = 31;
const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

const SYMBOLS: ReadonlyMap<string, number> = new Map(
  [...ALPHABET].map((symbol, index) => [symbol, index]),
);

const EXPECTED_VERSION = 1;
const EXPECTED_STORAGE = "alphabet";

function shapeSize(shape: readonly number[]): number {
  return shape.reduce((size, dimension) => size * dimension, 1);
}

/**
 * Decode the packed payload into a map from parameter name to values.
 *
 * Every segment must be a unique, in-bounds, non-overlapping range whose shape
 * product equals its declared length, and every symbol must belong to the
 * alphabet. Any violation is an error.
 */
export function decodeWeights(weights: WeightsArtifact): Map<string, Float32Array> {
  if (weights.version !== EXPECTED_VERSION) {
    throw new RangeError(`unsupported weights version: ${weights.version}`);
  }

  if (weights.storage !== EXPECTED_STORAGE) {
    throw new RangeError(`unsupported weights storage: ${weights.storage}`);
  }

  if (typeof weights.q !== "string") throw new TypeError("weights.q must be a string");

  const decoded = new Map<string, Float32Array>();
  const used = new Uint8Array(weights.q.length);

  for (const segment of weights.segments) {
    const { name, offset, length, shape, scale } = segment;

    if (typeof name !== "string" || name.length === 0) {
      throw new TypeError("segment name must be a non-empty string");
    }

    if (decoded.has(name)) throw new Error(`duplicate segment: ${name}`);

    if (!Number.isInteger(offset) || offset < 0) {
      throw new RangeError(`segment ${name}: offset must be a non-negative integer`);
    }

    if (!Number.isInteger(length) || length < 0) {
      throw new RangeError(`segment ${name}: length must be a non-negative integer`);
    }

    if (offset + length > weights.q.length) {
      throw new RangeError(`segment ${name}: range exceeds the q payload`);
    }

    if (!Number.isFinite(scale) || scale <= 0) {
      throw new RangeError(`segment ${name}: scale must be finite and positive`);
    }

    if (
      !Array.isArray(shape) ||
      shape.some((dimension) => !Number.isInteger(dimension) || dimension < 0)
    ) {
      throw new TypeError(`segment ${name}: shape must be non-negative integers`);
    }

    if (shapeSize(shape) !== length) {
      throw new RangeError(
        `segment ${name}: shape [${shape.join(", ")}] does not match length ${length}`,
      );
    }

    const values = new Float32Array(length);
    let index = 0;

    for (const symbol of weights.q.slice(offset, offset + length)) {
      const symbolIndex = SYMBOLS.get(symbol);

      if (symbolIndex === undefined) {
        throw new RangeError(`segment ${name}: unknown symbol ${JSON.stringify(symbol)}`);
      }

      if (used[offset + index] === 1) {
        throw new RangeError(`segment ${name}: overlaps an earlier segment`);
      }

      used[offset + index] = 1;
      values[index] = (symbolIndex - SYMBOL_BIAS) * scale;
      index += 1;
    }

    decoded.set(name, values);
  }

  return decoded;
}
