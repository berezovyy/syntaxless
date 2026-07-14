/**
 * Per-kind content hashes for incremental regen (STACK-PLAN S.2).
 *
 * A kind's artifact name is `<id>.<hash>.jsonl`. Hash inputs are the KindDef
 * templates + intents, SPLIT_VERSION, noise.ts bytes, and labels.ts bytes.
 * Unchanged kinds keep their on-disk jsonl/shards; a hash miss re-renders.
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { SPLIT_VERSION } from "./backward";
import { KINDS } from "./kinds";
import type { KindDef, KindId } from "./kinds";

function trainingSrc(): string {
  return dirname(fileURLToPath(import.meta.url));
}

export function fileSha256(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

export function noiseHash(root = resolve(trainingSrc(), "../../../..")): string {
  return fileSha256(join(root, "packages/cron/training/src/noise.ts"));
}

export function labelsHash(root = resolve(trainingSrc(), "../../../..")): string {
  return fileSha256(join(root, "packages/cron/src/model/labels.ts"));
}

export function hashKindDef(
  kind: KindDef,
  extras: { noise: string; labels: string; splitVersion?: string },
): string {
  const hash = createHash("sha256");

  hash.update(kind.id);
  hash.update("\0");
  hash.update(JSON.stringify(kind.templates));
  hash.update("\0");
  hash.update(JSON.stringify(kind.intents));
  hash.update("\0");
  hash.update(extras.splitVersion ?? SPLIT_VERSION);
  hash.update("\0");
  hash.update(extras.noise);
  hash.update("\0");
  hash.update(extras.labels);

  return hash.digest("hex").slice(0, 16);
}

export interface KindHashRecord {
  id: KindId;
  hash: string;
  jsonl: string;
  prefix: string;
}

export function kindArtifactNames(id: KindId, hash: string): { jsonl: string; prefix: string } {
  return {
    jsonl: `${id}.${hash}.jsonl`,
    prefix: `${id}.${hash}`,
  };
}

export function allKindHashes(
  dir: string,
  root = resolve(trainingSrc(), "../../../.."),
): KindHashRecord[] {
  const extras = { noise: noiseHash(root), labels: labelsHash(root) };

  return KINDS.map((kind) => {
    const hash = hashKindDef(kind, extras);
    const names = kindArtifactNames(kind.id, hash);

    return {
      id: kind.id,
      hash,
      jsonl: join(dir, names.jsonl),
      prefix: join(dir, names.prefix),
    };
  });
}

export interface KindRegenPlan {
  current: KindHashRecord[];
  stale: KindHashRecord[];
  ready: KindHashRecord[];
}

export function shardsReady(prefix: string): boolean {
  return (
    existsSync(`${prefix}-train.json`) &&
    existsSync(`${prefix}-validation.json`) &&
    existsSync(`${prefix}-heldout.json`)
  );
}

export function planKindRegen(
  dir: string,
  wanted?: readonly KindId[],
  root = resolve(trainingSrc(), "../../../.."),
): KindRegenPlan {
  const allow = wanted === undefined ? undefined : new Set(wanted);

  const current = allKindHashes(dir, root).filter(
    (record) => allow === undefined || allow.has(record.id),
  );

  const stale: KindHashRecord[] = [];
  const ready: KindHashRecord[] = [];

  for (const record of current) {
    if (existsSync(record.jsonl) && shardsReady(record.prefix)) ready.push(record);
    else stale.push(record);
  }

  return { current, stale, ready };
}
