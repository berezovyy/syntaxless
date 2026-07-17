/**
 * Exact SHA-256 membership set stored on disk with a fixed page cache.
 *
 * Online rejection sampling must ask "was this digest drawn before?" for
 * every candidate, which needs every earlier digest. Held in memory that is
 * 32+ bytes per generated row (`Hash256Set` reaches gigabytes at tens of
 * millions of rows). Here the digests live in a SQLite B-tree on disk and the
 * process keeps only `cacheBytes` of pages; the OS page cache may hold more,
 * which is reclaimable and not process working memory.
 *
 * Put the database on a real disk: a RAM-backed tmpfs directory would turn
 * the "disk" back into memory.
 */
import { Database, type Statement } from "bun:sqlite";
import { rmSync } from "node:fs";

/** Process-side SQLite page cache per set. */
const DEFAULT_CACHE_BYTES = 16 * 1024 ** 2;
/** Inserts per transaction; dirty pages beyond the cache spill to the file. */
const INSERTS_PER_TRANSACTION = 65_536;

export class DiskHash256Set {
  readonly #path: string;
  readonly #database: Database;
  readonly #has: Statement;
  readonly #insert: Statement;
  #size = 0;
  #uncommitted = 0;
  #closed = false;

  /** `path` is owned by the set: replaced on open and deleted by `close`. */
  constructor(path: string, cacheBytes = DEFAULT_CACHE_BYTES) {
    if (!Number.isSafeInteger(cacheBytes) || cacheBytes < 1024 * 1024) {
      throw new RangeError("DiskHash256Set cache must be at least 1 MiB");
    }

    this.#path = path;
    removeDatabase(path);
    this.#database = new Database(path, { create: true });

    try {
      // Scratch index: no journal or fsync; a crash discards the whole file.
      this.#database.exec(
        [
          "PRAGMA journal_mode=OFF",
          "PRAGMA synchronous=OFF",
          "PRAGMA locking_mode=EXCLUSIVE",
          "PRAGMA temp_store=FILE",
          "PRAGMA mmap_size=0",
          `PRAGMA cache_size=-${Math.floor(cacheBytes / 1024)}`,
          "CREATE TABLE digests (digest BLOB PRIMARY KEY) WITHOUT ROWID",
        ].join(";\n"),
      );

      this.#has = this.#database.prepare("SELECT 1 FROM digests WHERE digest = ?");
      this.#insert = this.#database.prepare("INSERT OR IGNORE INTO digests (digest) VALUES (?)");
    } catch (error) {
      this.#database.close();
      removeDatabase(path);
      throw error;
    }
  }

  get size(): number {
    return this.#size;
  }

  #check(digest: Buffer): void {
    if (this.#closed) throw new Error("DiskHash256Set is closed");

    if (digest.length !== 32) {
      throw new Error(`DiskHash256Set requires a 32-byte digest, got ${digest.length}`);
    }
  }

  has(digest: Buffer): boolean {
    this.#check(digest);

    return this.#has.get(digest) != null;
  }

  add(digest: Buffer): boolean {
    this.#check(digest);
    if (this.#uncommitted === 0) this.#database.exec("BEGIN");
    const added = this.#insert.run(digest).changes === 1;

    if (added) this.#size++;

    if (++this.#uncommitted >= INSERTS_PER_TRANSACTION) {
      this.#database.exec("COMMIT");
      this.#uncommitted = 0;
    }

    return added;
  }

  /** Close and delete the index. Safe to call more than once. */
  close(): void {
    if (this.#closed) return;
    this.#closed = true;

    try {
      if (this.#uncommitted > 0) this.#database.exec("COMMIT");
      this.#has.finalize();
      this.#insert.finalize();
    } finally {
      this.#database.close();
      removeDatabase(this.#path);
    }
  }

  [Symbol.dispose](): void {
    this.close();
  }
}

function removeDatabase(path: string): void {
  for (const suffix of ["", "-journal", "-wal", "-shm"]) {
    rmSync(`${path}${suffix}`, { force: true });
  }
}
