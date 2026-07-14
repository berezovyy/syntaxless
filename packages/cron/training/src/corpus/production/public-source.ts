/** Complete pinned public-English source import. No row cap, bootstrap fallback, or model calls. */
import { Database } from "bun:sqlite";
import { spawn } from "node:child_process";
import {
  closeSync,
  existsSync,
  fsyncSync,
  openSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
  writeSync,
} from "node:fs";
import { isAbsolute, join, resolve } from "node:path";
import { StringDecoder } from "node:string_decoder";
import { fileURLToPath } from "node:url";

import { z } from "zod";

import { tokenize } from "../../../../src/model/tokenizer";
import { normalize } from "../../../../src/parser/normalize";
import { MAX_TOKENS } from "../../featurize";
import { sha256Hex } from "../../fingerprint";
import { EASY_FILTER_VERSION, easyNoDropReason } from "../../mix/filter";
import type { CorpusRecord } from "../merge";
import { CorpusAudit } from "./audit";
import { SourceGroups, sourceGroupSplit } from "./groups";
import { loadProtectedText, type ProtectedText, textKey } from "./protected-text";
import { hashArtifact, publishDirectory } from "./publication";
import { corpusCodeHashes, fileHash } from "./source-hashes";

const archiveSchema = z
  .object({
    file: z.string().min(1),
    url: z.string().url(),
    sha256: z.string().regex(/^[a-f0-9]{64}$/),
  })
  .strict();

const configSchema = z
  .object({
    version: z.literal(1),
    sourceId: z.string().min(1),
    snapshotDate: z.string(),
    retrievedOn: z.string(),
    license: z.literal("CC-BY-2.0-FR"),
    licenseURL: z.string().url(),
    documentationURL: z.string().url(),
    sentences: archiveSchema,
    links: archiveSchema,
    grouping: z
      .object({
        version: z.string(),
        relatedNames: z.array(z.string().regex(/^[a-z]+$/)),
        rules: z.array(z.string()),
        limitation: z.string(),
      })
      .strict(),
    transformation: z.string(),
    purpose: z.string(),
  })
  .strict();

export type PublicSourceConfig = z.infer<typeof configSchema>;

export const PUBLIC_CONFIG_PATH = new URL(
  "../../../data/production/tatoeba-v2.json",
  import.meta.url,
);

const training = fileURLToPath(new URL("../../../", import.meta.url));
const archivePath = (file: string): string => (isAbsolute(file) ? file : join(training, file));

/** Decompression failure is observed before the caller can publish any result. */
export async function visitBzipLines(
  path: string,
  visit: (line: string, number: number) => void,
): Promise<void> {
  const child = spawn("bunzip2", ["-dc", path], { stdio: ["ignore", "pipe", "pipe"] });
  let errorOutput = "";

  child.stderr!.on("data", (chunk: Buffer) => {
    if (errorOutput.length < 2048) {
      errorOutput += chunk.toString().slice(0, 2048 - errorOutput.length);
    }
  });

  const completed = new Promise<Error | undefined>((finish) => {
    child.once("error", (error) => finish(error));

    child.once("close", (code) =>
      finish(code === 0 ? undefined : new Error(`bunzip2 failed (${code}): ${errorOutput.trim()}`)),
    );
  });

  const decoder = new StringDecoder("utf8");

  let pending = "";
  let number = 0;

  const consume = () => {
    let start = 0;

    for (;;) {
      const end = pending.indexOf("\n", start);

      if (end < 0) break;
      const content = pending.slice(start, end);

      visit(content.endsWith("\r") ? content.slice(0, -1) : content, ++number);
      start = end + 1;
    }

    pending = pending.slice(start);
  };

  try {
    for await (const chunk of child.stdout!) {
      pending += decoder.write(chunk as Buffer);
      consume();
    }

    pending += decoder.end();
    consume();
    if (pending) visit(pending.endsWith("\r") ? pending.slice(0, -1) : pending, ++number);
    const error = await completed;

    if (error) throw error;
  } finally {
    if (child.exitCode === null) child.kill();
    await completed;
  }
}

function sourceId(text: string | undefined, where: string): number {
  const value = Number(text);

  if (!text || !/^[1-9]\d*$/.test(text) || !Number.isSafeInteger(value)) {
    throw new Error(`Invalid source ID at ${where}`);
  }

  return value;
}

export function publicValueFrame(text: string, names: readonly string[]): string {
  const people = new Set(names);

  return textKey(text)
    .split(" ")
    .map((word) => (people.has(word) ? "{person}" : word.replace(/\d+/g, "{number}")))
    .join(" ");
}

interface StoredSentence {
  id: number;
  text: string;
  original: string;
  owner: string | null;
  added: string;
  modified: string;
  group_id: string;
  split: CorpusRecord["split"];
  rank: string;
  first_word: string;
  word_bucket: number;
}

export function publicCorpusRecord(
  config: Pick<PublicSourceConfig, "sourceId" | "license">,
  row: Pick<StoredSentence, "id" | "text" | "owner" | "group_id" | "split">,
): CorpusRecord {
  return {
    id: `${config.sourceId}/${row.id}`,
    split: row.split,
    family: "negative",
    text: row.text,
    spans: [
      {
        start: 0,
        end: row.text.length,
        label: "O",
        actionable: false,
        clauseStart: false,
      },
    ],
    intent: null,
    speechAct: "none",
    source: "rule",
    template: "easy-no/tatoeba-v2",
    fingerprint: sha256Hex(`negative\0${row.text}`),
    provenance: {
      authorship: "public-source",
      sourceId: config.sourceId,
      sourceRecordId: String(row.id),
      groupId: row.group_id,
      constructionId: "ordinary-english",
      license: config.license,
      attribution: `Tatoeba contributors; ${row.owner ? `listed owner ${row.owner}` : "no listed owner"}; https://tatoeba.org/en/sentences/show/${row.id}; runtime-normalized`,
    },
  };
}

function writeAll(fd: number, bytes: Buffer): void {
  let offset = 0;

  while (offset < bytes.length) {
    const n = writeSync(fd, bytes, offset, bytes.length - offset);

    if (n <= 0) throw new Error("Writing source artifact made no progress");
    offset += n;
  }
}

const bump = (counts: Record<string, number>, key: string): void => {
  counts[key] = (counts[key] ?? 0) + 1;
};

export interface PublicSourceOptions {
  configPath?: string | URL;
  output: string;
  protectedText?: ProtectedText;
  onProgress?: (phase: string, rows: number) => void;
}

export async function buildPublicSource(options: PublicSourceOptions) {
  if (existsSync(options.output)) {
    throw new Error(`Refusing to overwrite published source: ${options.output}`);
  }

  const configPath = options.configPath ?? PUBLIC_CONFIG_PATH;
  const configHash = fileHash(configPath);
  const codeHashes = corpusCodeHashes();
  const config = configSchema.parse(JSON.parse(readFileSync(configPath, "utf8")));
  const protectedText = options.protectedText ?? loadProtectedText();

  const paths = {
    sentences: archivePath(config.sentences.file),
    links: archivePath(config.links.file),
  };

  for (const key of ["sentences", "links"] as const) {
    if ((await hashArtifact(paths[key])) !== config[key].sha256) {
      throw new Error(`Pinned ${key} archive checksum mismatch`);
    }
  }

  // Conservative room for source JSONL, relational index, audit index, and SQLite
  // temporary sorting/VACUUM. This is a preflight, not a promise against other writers.
  const estimate = Math.max(
    512 * 1024 ** 2,
    (statSync(paths.sentences).size + statSync(paths.links).size) * 200,
  );

  return publishDirectory(options.output, estimate, async (staging) => {
    const indexPath = join(staging, "source.sqlite");
    const db = new Database(indexPath);

    db.exec(`PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-65536;
      CREATE TABLE source_ids (id INTEGER PRIMARY KEY);
      CREATE TABLE identities (hash TEXT PRIMARY KEY, id INTEGER);
      CREATE TABLE frames (hash TEXT PRIMARY KEY, id INTEGER);
      CREATE TABLE sentences (id INTEGER PRIMARY KEY, text TEXT, original TEXT, owner TEXT, added TEXT, modified TEXT,
        group_id TEXT, split TEXT, rank TEXT, first_word TEXT, word_bucket INTEGER); BEGIN;`);

    const groups = new SourceGroups();
    const protectedIds: number[] = [];

    let inputRows = 0;
    let linkRows = 0;
    let eligibleBeforeGroupProtection = 0;
    let kept = 0;

    const dropped: Record<string, number> = {};
    const splits: Record<string, number> = { train: 0, validation: 0, heldout: 0 };
    const owners: Record<string, number> = {};
    const words: Record<string, number> = {};
    const buckets: Record<string, number> = {};
    let indexClosed = false;
    let corpusFd: number | undefined;
    let audit: CorpusAudit | undefined;

    try {
      await visitBzipLines(paths.links, (line, number) => {
        const fields = line.split("\t");

        if (fields.length !== 2) {
          throw new Error(`Malformed English-English link at line ${number}`);
        }

        groups.union(sourceId(fields[0], `link ${number}`), sourceId(fields[1], `link ${number}`));
        linkRows++;
      });

      const seenId = db.query("INSERT INTO source_ids VALUES (?)");
      const getFrame = db.query<{ id: number }, [string]>("SELECT id FROM frames WHERE hash=?");
      const putFrame = db.query("INSERT OR IGNORE INTO frames VALUES (?,?)");

      const getIdentity = db.query<{ id: number }, [string]>(
        "SELECT id FROM identities WHERE hash=?",
      );

      const putIdentity = db.query(
        "INSERT INTO identities VALUES (?,?) ON CONFLICT(hash) DO UPDATE SET id=excluded.id",
      );

      const insert = db.query("INSERT INTO sentences VALUES (?,?,?,?,?,?,'','',?,?,?)");
      const remove = db.query("DELETE FROM sentences WHERE id=?");

      await visitBzipLines(paths.sentences, (line, number) => {
        inputRows++;
        if (inputRows % 100_000 === 0) options.onProgress?.("filter", inputRows);
        const fields = line.split("\t");

        if (fields.length !== 6 || fields[1] !== "eng" || !fields[2]) {
          throw new Error(`Malformed English detailed source at line ${number}`);
        }

        const id = sourceId(fields[0], `sentence ${number}`);

        seenId.run(id); // Duplicate IDs are corruption, not additional examples.
        const original = fields[2];
        const isProtected = protectedText.matches(original);

        if (isProtected) protectedIds.push(id);
        const reason = easyNoDropReason(original);

        if (reason) {
          bump(dropped, reason);

          return;
        }

        const text = normalize(original);

        if (tokenize(text).length > MAX_TOKENS) {
          bump(dropped, "token-limit");

          return;
        }

        const frame = sha256Hex(publicValueFrame(text, config.grouping.relatedNames));
        const previousFrame = getFrame.get(frame);

        if (previousFrame) groups.union(id, previousFrame.id);
        putFrame.run(frame, id);

        // Frame unions include protected eligible records so their value twins
        // are excluded later, rather than leaked under different names/numbers.
        if (isProtected) {
          bump(dropped, "protected-wording");

          return;
        }

        const identity = sha256Hex(textKey(text));
        const previous = getIdentity.get(identity);

        if (previous) {
          groups.union(id, previous.id);
          bump(dropped, "normalized-duplicate");
          if (id > previous.id) return;
          remove.run(previous.id); // Deterministic minimum-ID representative.
        }

        putIdentity.run(identity, id);
        const owner = !fields[3] || fields[3] === "\\N" ? null : fields[3];
        const tokens = text.split(" ");

        insert.run(
          id,
          text,
          original,
          owner,
          fields[4]!,
          fields[5]!,
          sha256Hex(`${config.sourceId}\0${id}`),
          tokens[0]!,
          Math.floor(tokens.length / 5) * 5,
        );
      });

      const blockedGroups = new Set(protectedIds.map((id) => groups.find(id)));
      const assign = db.query("UPDATE sentences SET group_id=?,split=? WHERE id=?");

      for (const row of db
        .query<{ id: number }, []>("SELECT id FROM sentences ORDER BY id")
        .iterate()) {
        eligibleBeforeGroupProtection++;
        const root = groups.find(row.id);

        if (blockedGroups.has(root)) {
          remove.run(row.id);
          bump(dropped, "protected-related-group");
          continue;
        }

        const group = `component-${root}`;

        assign.run(group, sourceGroupSplit(config.sourceId, group), row.id);
      }

      db.exec(
        "COMMIT; CREATE INDEX split_rank ON sentences(split,rank); CREATE INDEX group_members ON sentences(group_id); CREATE INDEX owner_rank ON sentences(split,owner,rank); DROP TABLE source_ids; DROP TABLE identities; DROP TABLE frames; VACUUM;",
      );

      const corpusPath = join(staging, "corpus.jsonl");

      corpusFd = openSync(corpusPath, "wx");

      audit = new CorpusAudit({
        protectedText,
        enforceMix: false,
        indexPath: join(staging, "audit.sqlite"),
      });

      let pending = "";

      for (const row of db
        .query<StoredSentence, []>("SELECT * FROM sentences ORDER BY id")
        .iterate()) {
        const record = publicCorpusRecord(config, row);

        audit.add(record);
        pending += `${JSON.stringify(record)}\n`;

        if (pending.length >= 1024 * 1024) {
          writeAll(corpusFd, Buffer.from(pending));
          pending = "";
        }

        kept++;
        if (kept % 100_000 === 0) options.onProgress?.("validate", kept);
        bump(splits, row.split);
        bump(owners, row.owner ?? "<unowned>");
        bump(words, row.first_word);
        bump(buckets, String(row.word_bucket));
      }

      if (pending) writeAll(corpusFd, Buffer.from(pending));
      fsyncSync(corpusFd);
      closeSync(corpusFd);
      corpusFd = undefined;
      const integrity = audit.finish();

      audit.close();
      audit = undefined;
      rmSync(join(staging, "audit.sqlite"));

      if (!kept || !integrity.passed) {
        throw new Error(
          `Public source integrity failed: ${JSON.stringify({ violations: integrity.violations, examples: integrity.examples })}`,
        );
      }

      const groupStats = db
        .query<{ groups: number; maxSize: number }, []>(
          "SELECT COUNT(*) AS groups,MAX(n) AS maxSize FROM (SELECT COUNT(*) AS n FROM sentences GROUP BY group_id)",
        )
        .get()!;

      const largestGroups = db
        .query<{ groupId: string; split: string; records: number }, []>(
          "SELECT group_id AS groupId,split,COUNT(*) AS records FROM sentences GROUP BY group_id ORDER BY records DESC,group_id LIMIT 10",
        )
        .all();

      db.close();
      indexClosed = true;
      const reportPath = join(staging, "validation.json");

      writeFileSync(reportPath, `${JSON.stringify(integrity, null, 2)}\n`, {
        flag: "wx",
      });

      if (
        configHash !== fileHash(configPath) ||
        JSON.stringify(codeHashes) !== JSON.stringify(corpusCodeHashes()) ||
        protectedText.sources.some((source) => fileHash(source.file) !== source.sha256)
      ) {
        throw new Error("Inputs or implementation changed during public-source generation");
      }

      for (const key of ["sentences", "links"] as const) {
        if ((await hashArtifact(paths[key])) !== config[key].sha256) {
          throw new Error(`Pinned ${key} archive changed during generation`);
        }
      }

      const top = (counts: Record<string, number>, n: number) =>
        Object.entries(counts)
          .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], "en"))
          .slice(0, n);

      const artifacts: Record<string, { bytes: number; sha256: string }> = {};

      for (const file of ["corpus.jsonl", "source.sqlite", "validation.json"]) {
        const path = join(staging, file);

        artifacts[file] = {
          bytes: statSync(path).size,
          sha256: await hashArtifact(path),
        };

        const fd = openSync(path, "r");

        try {
          fsyncSync(fd);
        } finally {
          closeSync(fd);
        }
      }

      const manifest = {
        version: 1,
        status: "complete-public-source-pool-not-final-training-mix",
        sourceId: config.sourceId,
        config,
        configHash,
        codeHashes,
        filterVersion: EASY_FILTER_VERSION,
        toolchain: {
          bun: process.versions.bun,
          platform: process.platform,
          architecture: process.arch,
        },
        protectedSources: protectedText.sources,
        inputRows,
        linkRows,
        eligibleBeforeGroupProtection,
        records: kept,
        uniqueTexts: integrity.uniqueTexts,
        splits,
        dropped,
        grouping: {
          ...groupStats,
          protectedComponents: blockedGroups.size,
          largestGroups,
        },
        concentration: {
          listedOwnerBuckets: Object.keys(owners).length,
          unowned: owners["<unowned>"] ?? 0,
          topListedOwners: top(owners, 20),
          topFirstWords: top(words, 30),
          wordCountBuckets: buckets,
          note: "Listed owners are not verified original authors. Final selection must control concentration; this is the complete eligible pool.",
        },
        artifacts,
      };

      const manifestFd = openSync(join(staging, "manifest.json"), "wx");

      try {
        writeFileSync(manifestFd, `${JSON.stringify(manifest, null, 2)}\n`);
        fsyncSync(manifestFd);
      } finally {
        closeSync(manifestFd);
      }

      return manifest;
    } finally {
      if (corpusFd !== undefined) closeSync(corpusFd);
      audit?.close();
      if (!indexClosed) db.close();
    }
  });
}

if (import.meta.main) {
  const output = process.argv[2];

  if (!output || process.argv.length !== 3) {
    throw new Error("Usage: public-source.ts NEW_OUTPUT_DIRECTORY");
  }

  const manifest = await buildPublicSource({
    output: resolve(output),
    onProgress: (phase, n) => console.error(`${phase}: ${n.toLocaleString()} rows`),
  });

  console.log(
    JSON.stringify(
      {
        records: manifest.records,
        splits: manifest.splits,
        groups: manifest.grouping,
        output: resolve(output),
      },
      null,
      2,
    ),
  );
}
