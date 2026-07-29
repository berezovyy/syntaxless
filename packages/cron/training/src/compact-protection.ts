/** Protect available evaluation wording and report gaps in historical evidence. */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadProtectedText, ProtectedText } from "./corpus/protected-text.js";
import { sha256Hex } from "./fingerprint.js";

const UNAVAILABLE_RECORDS_LIMITATION =
  "Some consumed historical records are unavailable. " +
  "Available evaluation wording is excluded, " +
  "but exclusion of unavailable wording cannot be verified. " +
  "Fresh reserved acceptance is still required for promotion.";

type Partition = {
  partitionSha256: string;
  records: { partition: string; text: string; trainingEligible: false }[];
};

function partitionDigest(records: readonly unknown[]): string {
  return sha256Hex(records.map((record) => JSON.stringify(record)).join("\n") + "\n");
}

export function loadCompactProtection() {
  const base = loadProtectedText();
  const evaluation = fileURLToPath(new URL("../../evaluation/", import.meta.url));

  const ledgerPath = join(evaluation, "consumed.json");
  const ledgerBytes = readFileSync(ledgerPath, "utf8");

  const ledger = JSON.parse(ledgerBytes);

  if (
    ledger.version !== 1 ||
    !Array.isArray(ledger.acceptancePartitionSha256) ||
    ledger.acceptancePartitionSha256.some(
      (hash: unknown) => typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash),
    )
  ) {
    throw new Error("Invalid cron acceptance consumption ledger");
  }

  const available = new Set<string>();

  for (const source of base.sources.filter((source) => source.file.endsWith(".jsonl"))) {
    const records = readFileSync(source.file, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));

    const acceptance = records.filter((record) => record.partition === "acceptance");

    if (acceptance.length) available.add(partitionDigest(acceptance));
  }

  const archivePath = join(evaluation, "historical-acceptance.json");
  const archiveBytes = readFileSync(archivePath, "utf8");

  const archive = JSON.parse(archiveBytes) as {
    version: number;
    partitions: Partition[];
  };

  if (archive.version !== 1 || !Array.isArray(archive.partitions)) {
    throw new Error("Invalid cron historical acceptance archive");
  }

  const texts: string[] = [];

  for (const partition of archive.partitions) {
    if (
      !Array.isArray(partition.records) ||
      !partition.records.length ||
      partition.records.some(
        (record) =>
          record.partition !== "acceptance" ||
          record.trainingEligible !== false ||
          typeof record.text !== "string",
      ) ||
      partitionDigest(partition.records) !== partition.partitionSha256 ||
      !ledger.acceptancePartitionSha256.includes(partition.partitionSha256)
    ) {
      throw new Error("Cron historical acceptance identity changed");
    }

    available.add(partition.partitionSha256);
    texts.push(...partition.records.map((record) => record.text));
  }

  const missing = ledger.acceptancePartitionSha256.filter((hash: string) => !available.has(hash));
  const historical = new ProtectedText(texts);

  return {
    matches: (text: string) => base.matches(text) || historical.matches(text),
    sources: [
      ...base.sources,
      { file: archivePath, sha256: sha256Hex(archiveBytes), records: texts.length },
    ],
    consumption: {
      ledgerSha256: sha256Hex(ledgerBytes),
      ledgerPartitions: ledger.acceptancePartitionSha256.length,
      verifiedPartitions: ledger.acceptancePartitionSha256.length - missing.length,
      unavailablePartitionSha256: missing as string[],
      limitation: missing.length ? UNAVAILABLE_RECORDS_LIMITATION : null,
    },
  };
}
