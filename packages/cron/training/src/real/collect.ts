/**
 * Track C sources: hash-pinned real-text manifest. Collection writes
 * `data/real/manifest.json`; live fetches are optional.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export interface RealSource {
  id: string;
  license: string;
  url: string;
  role: "negative" | "schedule" | "paraphrase-of-human";
}

export const REAL_SOURCES: readonly RealSource[] = [
  {
    id: "tatoeba-en",
    license: "CC-BY 2.0 FR",
    url: "https://tatoeba.org",
    role: "negative",
  },
  {
    id: "recognizers-datetime",
    license: "MIT",
    url: "https://github.com/microsoft/Recognizers-Text",
    role: "schedule",
  },
  {
    id: "cron-github-so",
    license: "CC-BY-SA",
    url: "https://stackoverflow.com/questions/tagged/cron",
    role: "schedule",
  },
];

export interface RealManifest {
  sources: readonly RealSource[];
  generatedAt: string;
  sha256: string;
}

function repoRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "../../../../..");
}

export function writeRealManifest(
  dir = join(repoRoot(), "packages/cron/training/data/real"),
): RealManifest {
  mkdirSync(dir, { recursive: true });
  const sha256 = createHash("sha256").update(JSON.stringify(REAL_SOURCES)).digest("hex");

  const manifest: RealManifest = {
    sources: REAL_SOURCES,
    generatedAt: new Date().toISOString(),
    sha256,
  };

  writeFileSync(join(dir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);

  return manifest;
}

export function readRealManifest(
  dir = join(repoRoot(), "packages/cron/training/data/real"),
): RealManifest | undefined {
  const path = join(dir, "manifest.json");

  if (!existsSync(path)) return undefined;

  return JSON.parse(readFileSync(path, "utf8")) as RealManifest;
}
