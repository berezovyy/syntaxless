/** Read wording hashes only, so reserved labels never supply training examples. */
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { readJsonl } from "./jsonl.js";
import { fromRepository } from "./paths.js";

const hash = (bytes: string | Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const digestPattern = /^[a-f0-9]{64}$/u;

export function reservedHashes(metadataDirectory: string, consumedDirectory: string): Set<string> {
  const excluded = new Set<string>();
  const available = new Set<string>();
  const missing: string[] = [];

  const consumed = (existsSync(consumedDirectory) ? readdirSync(consumedDirectory) : [])
    .filter((name) => /^[a-f0-9]{64}\.json$/u.test(name))
    .map((name) => {
      const record = JSON.parse(readFileSync(join(consumedDirectory, name), "utf8"));

      if (record.sha256 + ".json" !== name) throw Error("consumed identity changed");

      if (record.archiveSha256) {
        const bytes = readFileSync(join(consumedDirectory, record.sha256 + ".archive.json"));

        if (hash(bytes) !== record.archiveSha256) throw Error("consumed archive changed");
        const archive = JSON.parse(bytes.toString());

        if (
          archive.sha256 !== record.sha256 ||
          !Array.isArray(archive.textHashes) ||
          archive.textHashes.some(
            (digest: unknown) => typeof digest !== "string" || !digestPattern.test(digest),
          )
        ) {
          throw Error("consumed wording archive is invalid");
        }

        for (const digest of archive.textHashes) excluded.add(digest);
        available.add(record.sha256);
      }

      const acceptance = record.acceptance && fromRepository(record.acceptance);

      if (
        acceptance &&
        existsSync(acceptance) &&
        hash(readFileSync(acceptance)) !== record.sha256
      ) {
        throw Error("consumed full acceptance changed");
      }

      return record;
    });

  for (const name of readdirSync(metadataDirectory).filter((name) =>
    name.endsWith(".jsonl.meta.json"),
  )) {
    const metadata = JSON.parse(readFileSync(join(metadataDirectory, name), "utf8"));
    const file = join(metadataDirectory, name.slice(0, -10));

    if (
      metadata.trainingEligible !== false ||
      typeof metadata.sha256 !== "string" ||
      !digestPattern.test(metadata.sha256)
    ) {
      throw Error("reserved dataset metadata is invalid");
    }

    if (!existsSync(file)) {
      missing.push(metadata.sha256);
      continue;
    }

    if (hash(readFileSync(file)) !== metadata.sha256) {
      throw Error("reserved dataset identity changed");
    }

    let count = 0;

    for (const value of readJsonl(file)) {
      const row = value as { text?: unknown; trainingEligible?: unknown };

      if (typeof row?.text !== "string" || row.trainingEligible !== false) {
        throw Error("reserved rows must be evaluation-only");
      }

      excluded.add(hash(row.text.trim().toLowerCase()));
      count++;
    }

    if (count !== metadata.rows) throw Error("reserved dataset row count changed");
    available.add(metadata.sha256);
  }

  if (missing.some((digest) => !available.has(digest))) {
    throw Error("unconsumed reserved dataset is missing");
  }

  if (consumed.some((record) => !available.has(record.sha256))) {
    throw Error("consumed wording is missing");
  }

  return excluded;
}
