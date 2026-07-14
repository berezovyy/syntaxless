/**
 * Track F: rejection as calibrated energy, not a pooled speech-act head.
 *
 * Energy is −logsumexp over content-token role logits, averaged. A conformal
 * threshold (exported with the weights) rejects when energy exceeds it.
 */
import { CONTENT_LABELS, labelFromId } from "../model/labels.js";

const CONTENT = new Set<string>(CONTENT_LABELS);

export function logSumExp(logits: readonly number[]): number {
  if (logits.length === 0) return Number.NEGATIVE_INFINITY;
  let max = logits[0]!;

  for (const value of logits) if (value > max) max = value;
  let sum = 0;

  for (const value of logits) sum += Math.exp(value - max);

  return max + Math.log(sum);
}

export function tokenEnergy(roleLogits: readonly number[]): number {
  return -logSumExp(roleLogits);
}

export function sequenceEnergy(
  perTokenLogits: readonly (readonly number[])[],
  roleIds: readonly number[],
): number {
  const energies: number[] = [];

  for (let index = 0; index < perTokenLogits.length; index++) {
    const role = roleIds[index];

    if (role === undefined) continue;
    let label: string;

    try {
      label = labelFromId(role);
    } catch {
      continue;
    }

    if (!CONTENT.has(label)) continue;
    energies.push(tokenEnergy(perTokenLogits[index]!));
  }

  if (energies.length === 0) return 0;

  return energies.reduce((sum, value) => sum + value, 0) / energies.length;
}

export function mahalanobis1d(value: number, mean: number, variance: number): number {
  if (variance <= 0) return 0;
  const delta = value - mean;

  return Math.sqrt((delta * delta) / variance);
}

export function mahalanobisDiag(
  vector: ArrayLike<number>,
  mean: ArrayLike<number>,
  variance: ArrayLike<number>,
): number {
  const n = Math.min(vector.length, mean.length, variance.length);
  let sum = 0;

  for (let index = 0; index < n; index++) {
    const spread = variance[index]!;

    if (spread <= 0) continue;
    const delta = vector[index]! - mean[index]!;

    sum += (delta * delta) / spread;
  }

  return Math.sqrt(sum);
}

export function rejectOnEnergy(energy: number, threshold: number | undefined): boolean {
  if (threshold === undefined) return false;

  return energy > threshold;
}

export function rejectOnAbstainScore(input: {
  energy: number;
  energyThreshold: number | undefined;
  mahalanobis?: number;
  mahalanobisThreshold?: number;
}): boolean {
  if (rejectOnEnergy(input.energy, input.energyThreshold)) return true;

  if (
    input.mahalanobis !== undefined &&
    input.mahalanobisThreshold !== undefined &&
    input.mahalanobis > input.mahalanobisThreshold
  ) {
    return true;
  }

  return false;
}
