/**
 * Typed-array CPU inference for the exported int6 token tagger.
 *
 * Data only: importing this module decodes the committed weight artifact once
 * and nothing else. There is no session, no fetch, no top-level await, and no
 * dependency outside `src/model`. The forward pass mirrors the Phase 14 Tagger
 * operator order so the frozen Phase 15 parity fixtures agree to within float32
 * accumulation error.
 */

import { decodeWeights } from "./decode.js";
import type { WeightsArtifact } from "./decode.js";
import { LABEL_COUNT, LABELS } from "./labels.js";
import { SCHEDULE_KINDS } from "./schedule-kinds.js";
import { SPEECH_ACTS } from "./speech-acts.js";
import { featureRows, PADDING_ROW } from "./tokenizer.js";
import type { RawToken } from "./tokenizer.js";
import { weights } from "./weights.gen.js";

const ARTIFACT: WeightsArtifact = weights;
const HIDDEN = ARTIFACT.hidden;
const FEATURE_ROWS = ARTIFACT.featureRows;
const ROWS_PER_TOKEN = ARTIFACT.rowsPerToken;
const ROLE_CLASSES = ARTIFACT.roleClasses;
const SCHEDULE_KIND_CLASSES = ARTIFACT.scheduleKindClasses ?? 0;
const SPEECH_ACT_CLASSES = ARTIFACT.speechActClasses ?? 0;
const BOUNDARY_THRESHOLD = ARTIFACT.boundaryThreshold;
const LAYER_NORM_EPSILON = 1e-5;

/** Track F conformal energy threshold; absent until calibrate.py writes it. */
export const ENERGY_THRESHOLD: number | undefined = ARTIFACT.energyThreshold;

export const MAHALANOBIS_MEAN: readonly number[] | undefined = Array.isArray(
  ARTIFACT.mahalanobisMean,
)
  ? ARTIFACT.mahalanobisMean
  : undefined;

export const MAHALANOBIS_VARIANCE: readonly number[] | undefined = Array.isArray(
  ARTIFACT.mahalanobisVariance,
)
  ? ARTIFACT.mahalanobisVariance
  : undefined;

export const MAHALANOBIS_THRESHOLD: number | undefined = ARTIFACT.mahalanobisThreshold;

/** False until an A1 export ships weights for the typed label set. */
export const WEIGHTS_MATCH_LABELS =
  ROLE_CLASSES === LABEL_COUNT &&
  ARTIFACT.labels.length === LABELS.length &&
  ARTIFACT.labels.every((label, id) => label === LABELS[id]);

if (
  SCHEDULE_KIND_CLASSES > 0 &&
  (SCHEDULE_KIND_CLASSES !== SCHEDULE_KINDS.length ||
    ARTIFACT.scheduleKindLabels?.some((label, id) => label !== SCHEDULE_KINDS[id]) ||
    ARTIFACT.scheduleKindLabels?.length !== SCHEDULE_KINDS.length)
) {
  throw new Error("weights schedule-kind vocabulary does not match runtime");
}

if (
  SPEECH_ACT_CLASSES > 0 &&
  (SPEECH_ACT_CLASSES !== SPEECH_ACTS.length ||
    ARTIFACT.speechActLabels?.some((label, id) => label !== SPEECH_ACTS[id]) ||
    ARTIFACT.speechActLabels?.length !== SPEECH_ACTS.length)
) {
  throw new Error("weights speech-act vocabulary does not match runtime");
}

const tensors = decodeWeights(ARTIFACT);

function requireTensor(name: string): Float32Array {
  const values = tensors.get(name);

  if (values === undefined) throw new Error(`missing weight tensor: ${name}`);

  return values;
}

const EMBEDDING = requireTensor("embedding.weight");
const DEPTHWISE_WEIGHT = requireTensor("depthwise.weight");
const DEPTHWISE_BIAS = requireTensor("depthwise.bias");
const NEIGHBOR_WEIGHT = requireTensor("neighbor.weight");
const NEIGHBOR_BIAS = requireTensor("neighbor.bias");
const FORWARD_PROPOSAL_WEIGHT = requireTensor("forward_recurrence.proposal.weight");
const FORWARD_PROPOSAL_BIAS = requireTensor("forward_recurrence.proposal.bias");
const FORWARD_GATE_WEIGHT = requireTensor("forward_recurrence.gate.weight");
const FORWARD_GATE_BIAS = requireTensor("forward_recurrence.gate.bias");
const BACKWARD_PROPOSAL_WEIGHT = requireTensor("backward_recurrence.proposal.weight");
const BACKWARD_PROPOSAL_BIAS = requireTensor("backward_recurrence.proposal.bias");
const BACKWARD_GATE_WEIGHT = requireTensor("backward_recurrence.gate.weight");
const BACKWARD_GATE_BIAS = requireTensor("backward_recurrence.gate.bias");
const COMBINE_WEIGHT = requireTensor("combine.weight");
const COMBINE_BIAS = requireTensor("combine.bias");
const CONTEXT_NORM_WEIGHT = requireTensor("context_norm.weight");
const CONTEXT_NORM_BIAS = requireTensor("context_norm.bias");
const HEAD_GATE_WEIGHT = requireTensor("head_gate.weight");
const HEAD_GATE_BIAS = requireTensor("head_gate.bias");
const ROLE_HEAD_WEIGHT = requireTensor("role_head.weight");
const ROLE_HEAD_BIAS = requireTensor("role_head.bias");
const BOUNDARY_HEAD_WEIGHT = requireTensor("boundary_head.weight");
const BOUNDARY_HEAD_BIAS = requireTensor("boundary_head.bias");

const SCHEDULE_KIND_HEAD_WEIGHT =
  SCHEDULE_KIND_CLASSES === 0 ? undefined : requireTensor("schedule_kind_head.weight");

const SCHEDULE_KIND_HEAD_BIAS =
  SCHEDULE_KIND_CLASSES === 0 ? undefined : requireTensor("schedule_kind_head.bias");

const SPEECH_ACT_HEAD_WEIGHT =
  SPEECH_ACT_CLASSES === 0 ? undefined : requireTensor("speech_act_head.weight");

const SPEECH_ACT_HEAD_BIAS =
  SPEECH_ACT_CLASSES === 0 ? undefined : requireTensor("speech_act_head.bias");

const ACTIONABLE_HEAD_WEIGHT = tensors.get("actionable_head.weight");
const ACTIONABLE_HEAD_BIAS = tensors.get("actionable_head.bias");

if (ACTIONABLE_HEAD_WEIGHT !== undefined && ACTIONABLE_HEAD_BIAS === undefined) {
  throw new Error("missing weight tensor: actionable_head.bias");
}

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

/**
 * Run the token tagger over `count` valid rows and a possibly padded tail.
 * `rows` is a flat `[width, rowsPerToken]` feature-row table.
 */
export function inferRows(
  rows: Uint16Array | readonly number[],
  count: number,
  debug = false,
): Predictions {
  const width = Math.floor(rows.length / ROWS_PER_TOKEN);
  const validCount = Math.max(0, Math.min(Math.trunc(count), width));
  const mask = (token: number): number => (token < validCount ? 1 : 0);

  const states = new Float32Array(width * HIDDEN);

  for (let token = 0; token < width; token++) {
    for (let slot = 0; slot < ROWS_PER_TOKEN; slot++) {
      const row = rows[token * ROWS_PER_TOKEN + slot];

      // Accept both the active artifact sentinel and the current tokenizer
      // sentinel so frozen legacy parity fixtures remain loadable.
      if (row === undefined || row === FEATURE_ROWS || row === PADDING_ROW) continue;

      if (row < 0 || row >= FEATURE_ROWS) {
        throw new RangeError(`embedding row out of range: ${row}`);
      }

      for (let hidden = 0; hidden < HIDDEN; hidden++) {
        states[token * HIDDEN + hidden] =
          states[token * HIDDEN + hidden]! + EMBEDDING[row * HIDDEN + hidden]!;
      }
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

  for (let token = 0; token < width; token++) {
    for (let hidden = 0; hidden < HIDDEN; hidden++) {
      states[token * HIDDEN + hidden] =
        (states[token * HIDDEN + hidden]! + convolved[token * HIDDEN + hidden]!) * mask(token);
    }
  }

  const forward = recurrence(
    states,
    width,
    validCount,
    FORWARD_PROPOSAL_WEIGHT,
    FORWARD_PROPOSAL_BIAS,
    FORWARD_GATE_WEIGHT,
    FORWARD_GATE_BIAS,
    false,
  );

  const backward = recurrence(
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
      total += Math.exp(logits[token * ROLE_CLASSES + role]! - best);
    }

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
        value += SPEECH_ACT_HEAD_WEIGHT[act * HIDDEN + hidden]! * pooled[hidden]!;
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

/** Run the tagger over raw tokens without padding. */
export function inferTokens(tokens: readonly RawToken[], debug = false): Predictions {
  const rows = new Uint16Array(tokens.length * ROWS_PER_TOKEN).fill(FEATURE_ROWS);
  let token = 0;

  for (const item of tokens) {
    rows.set(
      featureRows(item, {
        featureRows: FEATURE_ROWS,
        rowsPerToken: ROWS_PER_TOKEN,
      }),
      token * ROWS_PER_TOKEN,
    );

    token += 1;
  }

  return inferRows(rows, tokens.length, debug);
}
