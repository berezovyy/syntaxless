import { expect, test } from "bun:test";
import { writeFileSync } from "node:fs";
import { join } from "node:path";

import { tempDir } from "../tests/helpers/temp-dir";
import { SPLIT_VERSION } from "./backward";
import { hashKindDef, kindArtifactNames, labelsHash, noiseHash, planKindRegen } from "./kind-hash";
import { KINDS } from "./kinds";

test("each kind hash is stable and unique for the current catalog", () => {
  const extras = {
    noise: noiseHash(),
    labels: labelsHash(),
    splitVersion: SPLIT_VERSION,
  };

  const hashes = KINDS.map((kind) => hashKindDef(kind, extras));

  expect(hashes.every((value) => /^[0-9a-f]{16}$/.test(value))).toBe(true);
  expect(new Set(hashes).size).toBe(hashes.length);
  expect(hashKindDef(KINDS[0]!, extras)).toBe(hashes[0]!);
});

test("planKindRegen marks missing artifacts stale and present ones ready", () => {
  const dir = tempDir("cron-kind-hash-");
  const extras = { noise: noiseHash(), labels: labelsHash() };
  const kind = KINDS[0]!;
  const hash = hashKindDef(kind, extras);
  const names = kindArtifactNames(kind.id, hash);
  const jsonl = join(dir, names.jsonl);

  writeFileSync(jsonl, "{}\n");
  writeFileSync(join(dir, `${names.prefix}-train.json`), "{}\n");
  writeFileSync(join(dir, `${names.prefix}-validation.json`), "{}\n");
  writeFileSync(join(dir, `${names.prefix}-heldout.json`), "{}\n");
  const plan = planKindRegen(dir, [kind.id]);

  expect(plan.ready.map((record) => record.id)).toEqual([kind.id]);
  expect(plan.stale).toEqual([]);
  const other = KINDS[1]!;
  const missing = planKindRegen(dir, [other.id]);

  expect(missing.stale.map((record) => record.id)).toEqual([other.id]);
});
