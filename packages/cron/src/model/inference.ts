/**
 * Typed-array CPU inference for packed int6 taggers.
 * Each engine owns its decoded tensors and calibration. Importing this module
 * does not load an artifact; operator order matches the Python parity fixtures.
 */
import { decodeWeights } from "./decode.js";
import type { WeightsArtifact } from "./decode.js";
import { isTypedVocabulary, LABEL_COUNT } from "./labels.js";
import { lexiconClassId } from "./lexical.js";
import { SCHEDULE_KINDS } from "./schedule-kinds.js";
import { DECISION_ACTS, isDecisionVocabulary, SPEECH_ACTS } from "./speech-acts.js";
import type { SpeechAct } from "./speech-acts.js";
import {
  CONSONANT_HASH_COUNT,
  CONSONANT_HASH_ROW0,
  featureRows,
  lexicalBucket,
  NUMBER_BUCKET_COUNT,
  NUMBER_BUCKET_ROW0,
  PADDING_ROW,
  WORD_HASH_COUNT,
  WORD_HASH_ROW0,
} from "./tokenizer.js";
import type { RawToken } from "./tokenizer.js";

/** Per-token model output. Padding positions are included for `inferRows`. */
export interface Predictions {
  labels: Uint8Array;
  boundaries: Uint8Array;
  scores: Float32Array;
  /** Distance from the calibrated boundary decision in `[0, 1]`. */
  boundaryScores: Float32Array;
  logits?: Float32Array;
  /** Mean-pooled context state (hidden,). */
  pooled?: Float32Array;
  boundaryLogits?: Float32Array;
  scheduleKind?: number;
  scheduleKindScore?: number;
  scheduleKindLogits?: Float32Array;
  speechAct?: number;
  speechActScore?: number;
  speechActLogits?: Float32Array;
  actionable?: Uint8Array;
  actionableScores?: Float32Array;
  actionableLogits?: Float32Array;
}

export interface InferenceEngine {
  readonly calibration: {
    readonly energyThreshold: number | undefined;
    readonly mahalanobisMean: readonly number[] | undefined;
    readonly mahalanobisVariance: readonly number[] | undefined;
    readonly mahalanobisThreshold: number | undefined;
  };
  readonly weightsMatchLabels: boolean;
  readonly decisionHead: boolean;
  predictedSpeechAct(id: number): SpeechAct;
  inferTokens(tokens: readonly RawToken[], debug?: boolean): Predictions;
  inferRows(
    rows: Uint16Array | readonly number[],
    count: number,
    debug?: boolean,
    lexicalIds?: Uint16Array,
    lexiconIds?: Uint8Array,
  ): Predictions;
}

/** Validate the model's tensor geometry and keep its state local to one engine. */
export function createInference(artifact: WeightsArtifact): InferenceEngine {
  const tensors = decodeWeights(artifact);
  const HIDDEN = artifact.hidden;
  const FEATURE_ROWS = artifact.featureRows;
  const ROWS_PER_TOKEN = artifact.rowsPerToken;
  const ROLE_CLASSES = artifact.roleClasses;
  const SCHEDULE_KIND_CLASSES = artifact.scheduleKindClasses ?? 0;
  const SPEECH_ACT_CLASSES = artifact.speechActClasses ?? 0;

  for (const [name, value, minimum] of [
    ["hidden", HIDDEN, 1],
    ["featureRows", FEATURE_ROWS, 1],
    ["rowsPerToken", ROWS_PER_TOKEN, 1],
    ["roleClasses", ROLE_CLASSES, 1],
    ["scheduleKindClasses", SCHEDULE_KIND_CLASSES, 0],
    ["speechActClasses", SPEECH_ACT_CLASSES, 0],
  ] as const) {
    if (!Number.isInteger(value) || value < minimum) {
      throw new RangeError(`weights ${name} must be an integer >= ${minimum}`);
    }
  }

  const BOUNDARY_THRESHOLD = artifact.boundaryThreshold;

  if (!Number.isFinite(BOUNDARY_THRESHOLD)) {
    throw new RangeError("weights boundaryThreshold must be finite");
  }

  for (const name of ["energyThreshold", "mahalanobisThreshold"] as const) {
    const value = artifact[name];

    if (value !== undefined && !Number.isFinite(value)) {
      throw new RangeError(`weights ${name} must be finite when present`);
    }
  }

  const LAYER_NORM_EPSILON = 1e-5;

  function calibrationVector(
    name: "mahalanobisMean" | "mahalanobisVariance",
  ): readonly number[] | undefined {
    const value = artifact[name];

    if (value === undefined) return undefined;

    // Legacy exports used finite scalars as inactive vector placeholders.
    if (typeof value === "number") {
      if (!Number.isFinite(value)) throw new RangeError(`weights ${name} must be finite`);

      return undefined;
    }

    if (!Array.isArray(value) || value.length !== HIDDEN) {
      throw new RangeError(`weights ${name} must have ${HIDDEN} coordinates`);
    }

    const nonNegative = name === "mahalanobisVariance";

    for (const coordinate of value) {
      if (!Number.isFinite(coordinate) || (nonNegative && coordinate < 0)) {
        const qualifier = nonNegative ? "non-negative " : "";

        throw new RangeError(`weights ${name} must contain finite ${qualifier}numbers`);
      }
    }

    return value.slice();
  }

  const ENERGY_THRESHOLD = artifact.energyThreshold;
  const MAHALANOBIS_MEAN = calibrationVector("mahalanobisMean");
  const MAHALANOBIS_VARIANCE = calibrationVector("mahalanobisVariance");
  const MAHALANOBIS_THRESHOLD = artifact.mahalanobisThreshold;

  /** Typed ids are stable; the shipped 77-class vocabulary predates context numbers. */
  const WEIGHTS_MATCH_LABELS = isTypedVocabulary(artifact.labels, ROLE_CLASSES);

  if ((ROLE_CLASSES === LABEL_COUNT || ROLE_CLASSES === 77) && !WEIGHTS_MATCH_LABELS) {
    throw new Error("weights typed role vocabulary does not match runtime");
  }

  if (
    SCHEDULE_KIND_CLASSES > 0 &&
    (SCHEDULE_KIND_CLASSES !== SCHEDULE_KINDS.length ||
      artifact.scheduleKindLabels?.some((label, id) => label !== SCHEDULE_KINDS[id]) ||
      artifact.scheduleKindLabels?.length !== SCHEDULE_KINDS.length)
  ) {
    throw new Error("weights schedule-kind vocabulary does not match runtime");
  }

  const SPEECH_LABELS = artifact.speechActLabels ?? [];

  const DECISION_HEAD =
    SPEECH_ACT_CLASSES === DECISION_ACTS.length && isDecisionVocabulary(SPEECH_LABELS);

  if (
    SPEECH_ACT_CLASSES > 0 &&
    !DECISION_HEAD &&
    (SPEECH_ACT_CLASSES !== SPEECH_ACTS.length ||
      SPEECH_LABELS.some((label, id) => label !== SPEECH_ACTS[id]) ||
      SPEECH_LABELS.length !== SPEECH_ACTS.length)
  ) {
    throw new Error("weights speech-act vocabulary does not match runtime");
  }

  const shapes = new Map(artifact.segments.map(({ name, shape }) => [name, shape]));
  const FACTORIZED = tensors.has("word_lexical.weight");

  function requireTensor(name: string, ...expected: number[]): Float32Array {
    const values = tensors.get(name);

    if (values === undefined) throw new Error(`missing weight tensor: ${name}`);
    const actual = shapes.get(name)!;

    if (actual.length !== expected.length || actual.some((value, i) => value !== expected[i])) {
      throw new RangeError(
        `weight tensor ${name}: expected shape [${expected}], received [${actual}]`,
      );
    }

    tensors.delete(name);

    return values;
  }

  function requireEmbedding(name: string): Float32Array {
    const rows = shapes.get(name)?.[0] ?? 0;

    if (rows < 1) throw new RangeError(`weight tensor ${name}: empty embedding table`);

    return requireTensor(name, rows, HIDDEN);
  }

  const EMBEDDING = requireTensor("embedding.weight", FEATURE_ROWS + 1, HIDDEN);
  const DEPTHWISE_WEIGHT = requireTensor("depthwise.weight", HIDDEN, FACTORIZED ? HIDDEN : 1, 3);
  const DEPTHWISE_BIAS = requireTensor("depthwise.bias", HIDDEN);
  const NEIGHBOR_WEIGHT = requireTensor("neighbor.weight", HIDDEN, HIDDEN);
  const NEIGHBOR_BIAS = requireTensor("neighbor.bias", HIDDEN);
  const RECURRENCE_INPUT = FACTORIZED ? HIDDEN * 2 : HIDDEN;

  const FORWARD_PROPOSAL_WEIGHT = requireTensor(
    "forward_recurrence.proposal.weight",
    HIDDEN,
    RECURRENCE_INPUT,
  );

  const FORWARD_PROPOSAL_BIAS = requireTensor("forward_recurrence.proposal.bias", HIDDEN);

  const FORWARD_GATE_WEIGHT = requireTensor(
    "forward_recurrence.gate.weight",
    HIDDEN,
    RECURRENCE_INPUT,
  );

  const FORWARD_GATE_BIAS = requireTensor("forward_recurrence.gate.bias", HIDDEN);

  const BACKWARD_PROPOSAL_WEIGHT = requireTensor(
    "backward_recurrence.proposal.weight",
    HIDDEN,
    RECURRENCE_INPUT,
  );

  const BACKWARD_PROPOSAL_BIAS = requireTensor("backward_recurrence.proposal.bias", HIDDEN);

  const BACKWARD_GATE_WEIGHT = requireTensor(
    "backward_recurrence.gate.weight",
    HIDDEN,
    RECURRENCE_INPUT,
  );

  const BACKWARD_GATE_BIAS = requireTensor("backward_recurrence.gate.bias", HIDDEN);
  const COMBINE_WEIGHT = requireTensor("combine.weight", HIDDEN, HIDDEN * 2);
  const COMBINE_BIAS = requireTensor("combine.bias", HIDDEN);
  const CONTEXT_NORM_WEIGHT = requireTensor("context_norm.weight", HIDDEN);
  const CONTEXT_NORM_BIAS = requireTensor("context_norm.bias", HIDDEN);
  const HEAD_GATE_WEIGHT = requireTensor("head_gate.weight", HIDDEN, HIDDEN * 2);
  const HEAD_GATE_BIAS = requireTensor("head_gate.bias", HIDDEN);
  const ROLE_HEAD_WEIGHT = requireTensor("role_head.weight", ROLE_CLASSES, HIDDEN);
  const ROLE_HEAD_BIAS = requireTensor("role_head.bias", ROLE_CLASSES);
  const BOUNDARY_HEAD_WEIGHT = requireTensor("boundary_head.weight", 1, HIDDEN);
  const BOUNDARY_HEAD_BIAS = requireTensor("boundary_head.bias", 1);

  const SCHEDULE_KIND_HEAD_WEIGHT =
    SCHEDULE_KIND_CLASSES === 0
      ? undefined
      : requireTensor("schedule_kind_head.weight", SCHEDULE_KIND_CLASSES, HIDDEN);

  const SCHEDULE_KIND_HEAD_BIAS =
    SCHEDULE_KIND_CLASSES === 0
      ? undefined
      : requireTensor("schedule_kind_head.bias", SCHEDULE_KIND_CLASSES);

  const SPEECH_ACT_HEAD_WEIGHT =
    SPEECH_ACT_CLASSES === 0
      ? undefined
      : requireTensor("speech_act_head.weight", SPEECH_ACT_CLASSES, HIDDEN);

  const SPEECH_ACT_HEAD_BIAS =
    SPEECH_ACT_CLASSES === 0
      ? undefined
      : requireTensor("speech_act_head.bias", SPEECH_ACT_CLASSES);

  const HAS_ACTIONABLE = tensors.has("actionable_head.weight");

  const ACTIONABLE_HEAD_WEIGHT = HAS_ACTIONABLE
    ? requireTensor("actionable_head.weight", 1, HIDDEN)
    : undefined;

  const ACTIONABLE_HEAD_BIAS = HAS_ACTIONABLE
    ? requireTensor("actionable_head.bias", 1)
    : undefined;

  const WORD_LEXICAL = FACTORIZED ? requireEmbedding("word_lexical.weight") : undefined;

  const CONSONANT_LEXICAL = FACTORIZED
    ? requireTensor("consonant_lexical.weight", CONSONANT_HASH_COUNT, HIDDEN)
    : undefined;

  const NUMBER_LEXICAL = FACTORIZED
    ? requireTensor("number_lexical.weight", NUMBER_BUCKET_COUNT, HIDDEN)
    : undefined;

  const LEXICON_CLASS_WEIGHT =
    FACTORIZED && tensors.has("lexicon_class.weight")
      ? requireEmbedding("lexicon_class.weight")
      : undefined;

  const LEXICAL_BUCKETS = WORD_LEXICAL === undefined ? 0 : WORD_LEXICAL.length / HIDDEN;

  if (tensors.size > 0) {
    throw new Error(`unexpected weight tensors: ${[...tensors.keys()].join(", ")}`);
  }

  function sigmoid(value: number): number {
    return 1 / (1 + Math.exp(-value));
  }

  /**
   * Derive each token's previous/next non-space neighbour from the kind feature,
   * which `featureRows` always emits first. Padding positions are excluded so a
   * neighbour never crosses into a padded tail.
   */
  function deriveNeighbors(
    rows: Uint16Array | readonly number[],
    width: number,
    validCount: number,
  ): { previous: Int32Array; next: Int32Array } {
    const previous = new Int32Array(width).fill(-1);
    const next = new Int32Array(width).fill(-1);
    let last = -1;

    for (let token = 0; token < width; token++) {
      previous[token] = last;

      if (token < validCount && ((rows[token * ROWS_PER_TOKEN] ?? FEATURE_ROWS) & 3) !== 3) {
        last = token;
      }
    }

    last = -1;

    for (let token = width - 1; token >= 0; token--) {
      next[token] = last;

      if (token < validCount && ((rows[token * ROWS_PER_TOKEN] ?? FEATURE_ROWS) & 3) !== 3) {
        last = token;
      }
    }

    return { previous, next };
  }

  function recurrence(
    states: Float32Array,
    width: number,
    validCount: number,
    proposalWeight: Float32Array,
    proposalBias: Float32Array,
    gateWeight: Float32Array,
    gateBias: Float32Array,
    reversed: boolean,
  ): Float32Array {
    const output = new Float32Array(width * HIDDEN);
    const state = new Float32Array(HIDDEN);

    for (let step = 0; step < width; step++) {
      const token = reversed ? width - 1 - step : step;
      const stepMask = token < validCount ? 1 : 0;

      for (let out = 0; out < HIDDEN; out++) {
        let proposal = proposalBias[out]!;
        let gate = gateBias[out]!;

        for (let input = 0; input < HIDDEN; input++) {
          const value = states[token * HIDDEN + input]!;

          proposal += proposalWeight[out * HIDDEN + input]! * value;
          gate += gateWeight[out * HIDDEN + input]! * value;
        }

        const candidate = (1 - sigmoid(gate)) * state[out]! + sigmoid(gate) * Math.tanh(proposal);

        state[out] = state[out]! + stepMask * (candidate - state[out]!);
      }

      output.set(state, token * HIDDEN);
    }

    return output;
  }

  /** Map a head id to the speech act decode branches on. 3-way `other` refuses. */
  function predictedSpeechAct(id: number): SpeechAct {
    if (DECISION_HEAD) {
      const name = DECISION_ACTS[id];

      if (name === "request" || name === "correction") return name;
      if (name === "other") return "none";
      throw new RangeError(`decision-act id out of range: ${id}`);
    }

    const act = SPEECH_ACTS[id];

    if (act === undefined) throw new RangeError(`speech-act id out of range: ${id}`);

    return act;
  }

  /**
   * Run the token tagger over `count` valid rows and a possibly padded tail.
   * `rows` is a flat `[width, rowsPerToken]` feature-row table.
   */
  function inferRows(
    rows: Uint16Array | readonly number[],
    count: number,
    debug = false,
    lexicalIds?: Uint16Array,
    lexiconIds?: Uint8Array,
  ): Predictions {
    const width = Math.floor(rows.length / ROWS_PER_TOKEN);
    const validCount = Math.max(0, Math.min(Math.trunc(count), width));
    const mask = (token: number): number => (token < validCount ? 1 : 0);

    const rowAt = (token: number, slot: number): number =>
      rows[token * ROWS_PER_TOKEN + slot] ?? FEATURE_ROWS;

    const states = new Float32Array(width * HIDDEN);

    for (let token = 0; token < width; token++) {
      for (let slot = 0; slot < ROWS_PER_TOKEN; slot++) {
        const row = rowAt(token, slot);

        if (row === FEATURE_ROWS || row === PADDING_ROW || (FACTORIZED && identityRow(row))) {
          continue;
        }

        if (row < 0 || row >= FEATURE_ROWS) {
          throw new RangeError(`embedding row out of range: ${row}`);
        }

        addEmbedding(states, token, EMBEDDING, row);
      }

      if (!FACTORIZED) continue;
      const word = rowAt(token, 4);

      if (LEXICAL_BUCKETS === WORD_HASH_COUNT) {
        if (word >= WORD_HASH_ROW0 && word < WORD_HASH_ROW0 + WORD_HASH_COUNT) {
          addEmbedding(states, token, WORD_LEXICAL!, word - WORD_HASH_ROW0);
        }
      } else {
        const id = lexicalIds?.[token];

        if (id === undefined) throw new Error("12-bit lexical ids are required for this artifact");
        addEmbedding(states, token, WORD_LEXICAL!, id);
      }

      const consonant = rowAt(token, 5);

      if (
        consonant >= CONSONANT_HASH_ROW0 &&
        consonant < CONSONANT_HASH_ROW0 + CONSONANT_HASH_COUNT
      ) {
        addEmbedding(states, token, CONSONANT_LEXICAL!, consonant - CONSONANT_HASH_ROW0);
      }

      const bucket = rowAt(token, 8);

      if (bucket >= NUMBER_BUCKET_ROW0 && bucket < NUMBER_BUCKET_ROW0 + NUMBER_BUCKET_COUNT) {
        addEmbedding(states, token, NUMBER_LEXICAL!, bucket - NUMBER_BUCKET_ROW0);
      }

      if (LEXICON_CLASS_WEIGHT !== undefined) {
        addEmbedding(states, token, LEXICON_CLASS_WEIGHT, lexiconIds?.[token] ?? 0);
      }
    }

    const { previous, next } = deriveNeighbors(rows, width, validCount);
    const neighborSums = new Float32Array(width * HIDDEN);

    for (let token = 0; token < width; token++) {
      const before = previous[token]!;
      const after = next[token]!;

      if (before >= 0) {
        for (let hidden = 0; hidden < HIDDEN; hidden++) {
          neighborSums[token * HIDDEN + hidden] =
            neighborSums[token * HIDDEN + hidden]! + states[before * HIDDEN + hidden]!;
        }
      }

      if (after >= 0) {
        for (let hidden = 0; hidden < HIDDEN; hidden++) {
          neighborSums[token * HIDDEN + hidden] =
            neighborSums[token * HIDDEN + hidden]! + states[after * HIDDEN + hidden]!;
        }
      }
    }

    for (let token = 0; token < width; token++) {
      for (let out = 0; out < HIDDEN; out++) {
        let sum = NEIGHBOR_BIAS[out]!;

        for (let input = 0; input < HIDDEN; input++) {
          sum += NEIGHBOR_WEIGHT[out * HIDDEN + input]! * neighborSums[token * HIDDEN + input]!;
        }

        states[token * HIDDEN + out] = states[token * HIDDEN + out]! + sum;
      }
    }

    for (let token = 0; token < width; token++) {
      if (mask(token) === 0) states.fill(0, token * HIDDEN, (token + 1) * HIDDEN);
    }

    const convolved = new Float32Array(width * HIDDEN);

    if (FACTORIZED) {
      for (let token = 0; token < width; token++) {
        for (let out = 0; out < HIDDEN; out++) {
          let sum = DEPTHWISE_BIAS[out]!;

          for (let input = 0; input < HIDDEN; input++) {
            for (let kernel = 0; kernel < 3; kernel++) {
              const neighbour = token + kernel - 1;

              if (neighbour < 0 || neighbour >= width) continue;

              sum +=
                DEPTHWISE_WEIGHT[(out * HIDDEN + input) * 3 + kernel]! *
                states[neighbour * HIDDEN + input]!;
            }
          }

          convolved[token * HIDDEN + out] = sum;
        }
      }
    } else {
      for (let token = 0; token < width; token++) {
        for (let hidden = 0; hidden < HIDDEN; hidden++) {
          let sum = DEPTHWISE_BIAS[hidden]!;

          for (let kernel = 0; kernel < 3; kernel++) {
            const neighbour = token + kernel - 1;

            if (neighbour < 0 || neighbour >= width) continue;
            sum += DEPTHWISE_WEIGHT[hidden * 3 + kernel]! * states[neighbour * HIDDEN + hidden]!;
          }

          convolved[token * HIDDEN + hidden] = sum;
        }
      }
    }

    for (let token = 0; token < width; token++) {
      for (let hidden = 0; hidden < HIDDEN; hidden++) {
        states[token * HIDDEN + hidden] =
          (states[token * HIDDEN + hidden]! + convolved[token * HIDDEN + hidden]!) * mask(token);
      }
    }

    const recurrent = FACTORIZED ? statefulRecurrence : recurrence;

    const forward = recurrent(
      states,
      width,
      validCount,
      FORWARD_PROPOSAL_WEIGHT,
      FORWARD_PROPOSAL_BIAS,
      FORWARD_GATE_WEIGHT,
      FORWARD_GATE_BIAS,
      false,
    );

    const backward = recurrent(
      states,
      width,
      validCount,
      BACKWARD_PROPOSAL_WEIGHT,
      BACKWARD_PROPOSAL_BIAS,
      BACKWARD_GATE_WEIGHT,
      BACKWARD_GATE_BIAS,
      true,
    );

    const combined = new Float32Array(width * HIDDEN);

    for (let token = 0; token < width; token++) {
      for (let out = 0; out < HIDDEN; out++) {
        let sum = COMBINE_BIAS[out]!;

        for (let input = 0; input < HIDDEN; input++) {
          sum += COMBINE_WEIGHT[out * HIDDEN * 2 + input]! * forward[token * HIDDEN + input]!;

          sum +=
            COMBINE_WEIGHT[out * HIDDEN * 2 + HIDDEN + input]! * backward[token * HIDDEN + input]!;
        }

        combined[token * HIDDEN + out] = sum;
      }
    }

    for (let token = 0; token < width; token++) {
      let mean = 0;

      for (let hidden = 0; hidden < HIDDEN; hidden++) mean += combined[token * HIDDEN + hidden]!;
      mean /= HIDDEN;
      let variance = 0;

      for (let hidden = 0; hidden < HIDDEN; hidden++) {
        const centered = combined[token * HIDDEN + hidden]! - mean;

        variance += centered * centered;
      }

      variance /= HIDDEN;
      const inverse = 1 / Math.sqrt(variance + LAYER_NORM_EPSILON);

      for (let hidden = 0; hidden < HIDDEN; hidden++) {
        combined[token * HIDDEN + hidden] =
          (combined[token * HIDDEN + hidden]! - mean) * inverse * CONTEXT_NORM_WEIGHT[hidden]! +
          CONTEXT_NORM_BIAS[hidden]!;
      }
    }

    const pooled = new Float32Array(HIDDEN);

    for (let token = 0; token < validCount; token++) {
      for (let hidden = 0; hidden < HIDDEN; hidden++) {
        pooled[hidden] = pooled[hidden]! + combined[token * HIDDEN + hidden]!;
      }
    }

    const denominator = Math.max(validCount, 1);

    for (let hidden = 0; hidden < HIDDEN; hidden++) pooled[hidden] = pooled[hidden]! / denominator;

    const gated = new Float32Array(width * HIDDEN);

    for (let token = 0; token < width; token++) {
      for (let out = 0; out < HIDDEN; out++) {
        let sum = HEAD_GATE_BIAS[out]!;

        for (let input = 0; input < HIDDEN; input++) {
          sum += HEAD_GATE_WEIGHT[out * HIDDEN * 2 + input]! * combined[token * HIDDEN + input]!;
          sum += HEAD_GATE_WEIGHT[out * HIDDEN * 2 + HIDDEN + input]! * pooled[input]!;
        }

        gated[token * HIDDEN + out] = sigmoid(sum) * combined[token * HIDDEN + out]!;
      }
    }

    const labels = new Uint8Array(width);
    const boundaries = new Uint8Array(width);
    const scores = new Float32Array(width);
    const logits = new Float32Array(width * ROLE_CLASSES);
    // Factorized confidence used unrounded logits before Float32 storage.
    const exactRoleLogits = FACTORIZED ? new Float64Array(ROLE_CLASSES) : undefined;

    const outside =
      FACTORIZED && SPEECH_ACT_HEAD_WEIGHT !== undefined ? new Float32Array(width) : undefined;

    const boundaryLogits = new Float32Array(width);
    const boundaryScores = new Float32Array(width);

    for (let token = 0; token < width; token++) {
      let best = -Infinity;
      let bestIndex = 0;
      let second = -Infinity;

      for (let role = 0; role < ROLE_CLASSES; role++) {
        let value = ROLE_HEAD_BIAS[role]!;

        for (let hidden = 0; hidden < HIDDEN; hidden++) {
          value += ROLE_HEAD_WEIGHT[role * HIDDEN + hidden]! * gated[token * HIDDEN + hidden]!;
        }

        logits[token * ROLE_CLASSES + role] = value;
        if (exactRoleLogits !== undefined) exactRoleLogits[role] = value;

        if (value > best) {
          second = best;
          best = value;
          bestIndex = role;
        } else if (value > second) {
          second = value;
        }
      }

      labels[token] = bestIndex;

      let boundary = BOUNDARY_HEAD_BIAS[0]!;

      for (let hidden = 0; hidden < HIDDEN; hidden++) {
        boundary += BOUNDARY_HEAD_WEIGHT[hidden]! * gated[token * HIDDEN + hidden]!;
      }

      boundaryLogits[token] = boundary;
      boundaries[token] = Number(boundary >= BOUNDARY_THRESHOLD);
      boundaryScores[token] = 1 - Math.exp(-Math.abs(boundary - BOUNDARY_THRESHOLD));

      let total = 0;

      for (let role = 0; role < ROLE_CLASSES; role++) {
        total += Math.exp((exactRoleLogits?.[role] ?? logits[token * ROLE_CLASSES + role]!) - best);
      }

      if (outside !== undefined) outside[token] = Math.exp(exactRoleLogits![0]! - best) / total;
      const secondValue = second === -Infinity ? best : second;

      scores[token] = 1 / total - Math.exp(secondValue - best) / total;
    }

    let scheduleKind: number | undefined;
    let scheduleKindScore: number | undefined;
    let scheduleKindLogits: Float32Array | undefined;

    if (
      SCHEDULE_KIND_CLASSES > 0 &&
      SCHEDULE_KIND_HEAD_WEIGHT !== undefined &&
      SCHEDULE_KIND_HEAD_BIAS !== undefined
    ) {
      scheduleKindLogits = new Float32Array(SCHEDULE_KIND_CLASSES);
      let best = -Infinity;
      let second = -Infinity;
      let bestIndex = 0;

      for (let kind = 0; kind < SCHEDULE_KIND_CLASSES; kind++) {
        let value = SCHEDULE_KIND_HEAD_BIAS[kind]!;

        for (let hidden = 0; hidden < HIDDEN; hidden++) {
          value += SCHEDULE_KIND_HEAD_WEIGHT[kind * HIDDEN + hidden]! * pooled[hidden]!;
        }

        scheduleKindLogits[kind] = value;

        if (value > best) {
          second = best;
          best = value;
          bestIndex = kind;
        } else if (value > second) {
          second = value;
        }
      }

      let total = 0;

      for (const value of scheduleKindLogits) total += Math.exp(value - best);
      scheduleKind = bestIndex;
      scheduleKindScore = 1 / total - Math.exp(second - best) / total;
    }

    let speechPool = pooled;

    if (outside !== undefined) {
      let speechMass = 0;

      speechPool = new Float32Array(HIDDEN);

      for (let token = 0; token < validCount; token++) {
        if ((rowAt(token, 0) & 3) === 3) continue;
        const weight = outside[token]!;

        speechMass += weight;

        for (let hidden = 0; hidden < HIDDEN; hidden++) {
          speechPool[hidden] = speechPool[hidden]! + weight * combined[token * HIDDEN + hidden]!;
        }
      }

      if (speechMass < 1e-4) {
        speechMass = 0;
        speechPool.fill(0);

        for (let token = 0; token < validCount; token++) {
          if ((rowAt(token, 0) & 3) === 3) continue;
          speechMass += 1;

          for (let hidden = 0; hidden < HIDDEN; hidden++) {
            speechPool[hidden] = speechPool[hidden]! + combined[token * HIDDEN + hidden]!;
          }
        }
      }

      const speechDenom = Math.max(speechMass, 1);

      for (let hidden = 0; hidden < HIDDEN; hidden++) {
        speechPool[hidden] = speechPool[hidden]! / speechDenom;
      }
    }

    let speechAct: number | undefined;
    let speechActScore: number | undefined;
    let speechActLogits: Float32Array | undefined;

    if (
      SPEECH_ACT_CLASSES > 0 &&
      SPEECH_ACT_HEAD_WEIGHT !== undefined &&
      SPEECH_ACT_HEAD_BIAS !== undefined
    ) {
      speechActLogits = new Float32Array(SPEECH_ACT_CLASSES);
      let best = -Infinity;
      let second = -Infinity;
      let bestIndex = 0;

      for (let act = 0; act < SPEECH_ACT_CLASSES; act++) {
        let value = SPEECH_ACT_HEAD_BIAS[act]!;

        for (let hidden = 0; hidden < HIDDEN; hidden++) {
          value += SPEECH_ACT_HEAD_WEIGHT[act * HIDDEN + hidden]! * speechPool[hidden]!;
        }

        speechActLogits[act] = value;

        if (value > best) {
          second = best;
          best = value;
          bestIndex = act;
        } else if (value > second) {
          second = value;
        }
      }

      let total = 0;

      for (const value of speechActLogits) total += Math.exp(value - best);
      speechAct = bestIndex;
      speechActScore = 1 / total - Math.exp(second - best) / total;
    }

    let actionable: Uint8Array | undefined;
    let actionableScores: Float32Array | undefined;
    let actionableLogits: Float32Array | undefined;

    if (ACTIONABLE_HEAD_WEIGHT !== undefined && ACTIONABLE_HEAD_BIAS !== undefined) {
      actionable = new Uint8Array(width);
      actionableScores = new Float32Array(width);
      actionableLogits = new Float32Array(width);

      for (let token = 0; token < width; token++) {
        let value = ACTIONABLE_HEAD_BIAS[0]!;

        for (let hidden = 0; hidden < HIDDEN; hidden++) {
          value += ACTIONABLE_HEAD_WEIGHT[hidden]! * gated[token * HIDDEN + hidden]!;
        }

        actionableLogits[token] = value;
        actionable[token] = Number(value >= 0);
        actionableScores[token] = sigmoid(value);
      }
    }

    return {
      labels,
      boundaries,
      scores,
      boundaryScores,
      ...(scheduleKind === undefined || scheduleKindScore === undefined
        ? {}
        : { scheduleKind, scheduleKindScore }),
      ...(speechAct === undefined || speechActScore === undefined
        ? {}
        : { speechAct, speechActScore }),
      ...(actionable === undefined || actionableScores === undefined
        ? {}
        : { actionable, actionableScores }),
      logits,
      pooled,
      ...(debug ? { boundaryLogits } : {}),
      ...(debug && scheduleKindLogits !== undefined ? { scheduleKindLogits } : {}),
      ...(debug && speechActLogits !== undefined ? { speechActLogits } : {}),
      ...(debug && actionableLogits !== undefined ? { actionableLogits } : {}),
    };
  }

  function identityRow(row: number): boolean {
    return (
      (row >= WORD_HASH_ROW0 && row < WORD_HASH_ROW0 + WORD_HASH_COUNT) ||
      (row >= CONSONANT_HASH_ROW0 && row < CONSONANT_HASH_ROW0 + CONSONANT_HASH_COUNT) ||
      (row >= NUMBER_BUCKET_ROW0 && row < NUMBER_BUCKET_ROW0 + NUMBER_BUCKET_COUNT)
    );
  }

  function addEmbedding(
    states: Float32Array,
    token: number,
    table: Float32Array,
    id: number,
  ): void {
    const base = id * HIDDEN;
    const into = token * HIDDEN;

    for (let hidden = 0; hidden < HIDDEN; hidden++) {
      states[into + hidden] = states[into + hidden]! + table[base + hidden]!;
    }
  }

  function statefulRecurrence(
    states: Float32Array,
    width: number,
    validCount: number,
    proposalWeight: Float32Array,
    proposalBias: Float32Array,
    gateWeight: Float32Array,
    gateBias: Float32Array,
    reversed: boolean,
  ): Float32Array {
    const output = new Float32Array(width * HIDDEN);
    const state = new Float32Array(HIDDEN);
    const mixed = new Float32Array(HIDDEN * 2);

    for (let step = 0; step < width; step++) {
      const token = reversed ? width - 1 - step : step;
      const stepMask = token < validCount ? 1 : 0;

      mixed.set(states.subarray(token * HIDDEN, (token + 1) * HIDDEN), 0);
      mixed.set(state, HIDDEN);

      for (let out = 0; out < HIDDEN; out++) {
        let proposal = proposalBias[out]!;
        let gate = gateBias[out]!;

        for (let input = 0; input < HIDDEN * 2; input++) {
          const value = mixed[input]!;

          proposal += proposalWeight[out * HIDDEN * 2 + input]! * value;
          gate += gateWeight[out * HIDDEN * 2 + input]! * value;
        }

        const open = sigmoid(gate);
        const candidate = (1 - open) * state[out]! + open * Math.tanh(proposal);

        state[out] = state[out]! + stepMask * (candidate - state[out]!);
      }

      output.set(state, token * HIDDEN);
    }

    return output;
  }

  /** Run the tagger over raw tokens without padding. */
  function inferTokens(tokens: readonly RawToken[], debug = false): Predictions {
    const rows = new Uint16Array(tokens.length * ROWS_PER_TOKEN).fill(FEATURE_ROWS);
    const lexical = FACTORIZED ? new Uint16Array(tokens.length) : undefined;

    const lexicon =
      FACTORIZED && LEXICON_CLASS_WEIGHT !== undefined ? new Uint8Array(tokens.length) : undefined;

    let token = 0;

    for (const item of tokens) {
      rows.set(
        featureRows(item, {
          featureRows: FEATURE_ROWS,
          rowsPerToken: ROWS_PER_TOKEN,
        }),
        token * ROWS_PER_TOKEN,
      );

      if (lexical !== undefined) lexical[token] = lexicalBucket(item.text);
      if (lexicon !== undefined) lexicon[token] = lexiconClassId(item.text);
      token += 1;
    }

    return inferRows(rows, tokens.length, debug, lexical, lexicon);
  }

  return {
    inferTokens,
    inferRows,
    predictedSpeechAct,
    weightsMatchLabels: WEIGHTS_MATCH_LABELS,
    decisionHead: DECISION_HEAD,
    calibration: {
      energyThreshold: ENERGY_THRESHOLD,
      mahalanobisMean: MAHALANOBIS_MEAN,
      mahalanobisVariance: MAHALANOBIS_VARIANCE,
      mahalanobisThreshold: MAHALANOBIS_THRESHOLD,
    },
  };
}
