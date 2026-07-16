/**
 * Energy is −logsumexp over content-token role logits, averaged. A conformal
 * threshold (exported with the weights) rejects when energy exceeds it.
 */
import { CONTENT_LABELS, labelId } from "../model/labels.js";

const CONTENT_ROLES = new Set(CONTENT_LABELS.map((label) => labelId[label]));

export function sequenceEnergy(
  logits: ArrayLike<number>,
  roleIds: ArrayLike<number>,
  classes: number,
): number {
  let energy = 0;
  let content = 0;

  for (let index = 0; index < roleIds.length; index++) {
    if (!CONTENT_ROLES.has(roleIds[index]!)) continue;
    const offset = index * classes;
    let max = logits[offset]!;

    for (let role = 1; role < classes; role++) {
      if (logits[offset + role]! > max) max = logits[offset + role]!;
    }

    let sum = 0;

    for (let role = 0; role < classes; role++) sum += Math.exp(logits[offset + role]! - max);
    energy -= max + Math.log(sum);
    content++;
  }

  return content === 0 ? 0 : energy / content;
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
