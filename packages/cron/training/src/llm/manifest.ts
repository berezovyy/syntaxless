/**
 * Run manifest for an opencode corpus run.
 *
 * A manifest records what was asked of the model without recording anything the
 * model said: run id, model/provider ids, the angles and their prompt versions,
 * requested vs produced counts, cache hits, and a hash of the runner sources.
 * It carries no secret, no key, and no raw provider output. Written once when a
 * run finishes, under `data/llm/runs/<runId>/manifest.json`.
 */
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export interface ManifestAngle {
  angleId: string;
  promptId: string;
  promptVersion: number;
}

export interface RunManifest {
  runId: string;
  model: { providerID: string; modelID: string };
  angles: ManifestAngle[];
  /** Candidates requested (angles times count). */
  requested: number;
  /** Candidates that parsed into records. */
  produced: number;
  /** Candidate lines that failed to parse. */
  rejected: number;
  /** Angles served from the response cache. */
  cacheHits: number;
  /** Hash of the runner sources at run time. */
  runnerSha256: string;
}

/** Runner sources hashed into the manifest so drift is detectable. */
const RUNNER_SOURCES: readonly string[] = [
  "cache.ts",
  "client.ts",
  "manifest.ts",
  "runner.ts",
  "workflow.ts",
];

/** Hash the runner sources, path-tagged and sorted for stable output. */
export function runnerSha256(): string {
  const base = dirname(fileURLToPath(import.meta.url));
  const digest = createHash("sha256");

  for (const name of [...RUNNER_SOURCES].sort()) {
    digest.update(name, "utf8");
    digest.update("\0", "utf8");
    digest.update(readFileSync(join(base, name)));
    digest.update("\0", "utf8");
  }

  return digest.digest("hex");
}

/** Default runs directory, resolved from this module, not the working dir. */
export function defaultRunsDir(): string {
  return join(dirname(fileURLToPath(import.meta.url)), "..", "..", "data", "llm", "runs");
}

/** Write the manifest for a finished run. Returns the written path. */
export function writeManifest(manifest: RunManifest, dir = defaultRunsDir()): string {
  const path = join(dir, manifest.runId, "manifest.json");

  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(manifest, null, 2)}\n`);

  return path;
}
