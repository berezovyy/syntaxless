/**
 * External merge sort that replays lines in ascending SHA-256 order.
 *
 * Source traversal groups rows by source. The trainer shuffles only inside
 * bounded locality blocks, so the published corpus must already be mixed.
 * Ordering by the union's text hash is deterministic and independent of
 * source order.
 *
 * Memory is bounded by `runCharacters` while adding and by `mergeFanIn` read
 * buffers while draining, whatever the corpus size: a full run is sorted and
 * written to disk, and runs are merged in as many passes as needed. Do not
 * replace this with "collect, sort, write" or with fixed hash buckets: the
 * emitted corpus is the largest artifact in the pipeline and a bucket grows
 * with it (see the training README's memory rule).
 */
import { closeSync, mkdirSync, openSync, readSync, rmSync, unlinkSync, writeSync } from "node:fs";
import { join } from "node:path";
import { StringDecoder } from "node:string_decoder";

/** Default run size in UTF-16 code units of queued `hash\tline` text. */
const DEFAULT_RUN_CHARACTERS = 16 * 1024 * 1024;
/** Default number of runs merged at once (one read buffer each). */
const DEFAULT_MERGE_FAN_IN = 64;
const READ_BUFFER_BYTES = 64 * 1024;
/** Run-file writes and output chunks are batched to about this many characters. */
const WRITE_CHARACTERS = 1024 * 1024;
const HASH = /^[0-9a-f]{64}$/;
const KEY_CHARACTERS = 64;

export interface HashOrderedSpoolOptions {
  /** Characters queued in memory before a sorted run is written to disk. */
  runCharacters?: number;
  /** Runs merged per pass. Tests lower it to force multi-pass merges. */
  mergeFanIn?: number;
}

/** Buffered, synchronous text writer for one run file. */
class RunWriter {
  readonly #fd: number;
  #pending: string[] = [];
  #characters = 0;

  constructor(path: string) {
    this.#fd = openSync(path, "wx");
  }

  line(text: string): void {
    this.#pending.push(text, "\n");
    this.#characters += text.length + 1;
    if (this.#characters >= WRITE_CHARACTERS) this.flush();
  }

  flush(): void {
    if (this.#characters === 0) return;
    const bytes = Buffer.from(this.#pending.join(""), "utf8");

    this.#pending = [];
    this.#characters = 0;
    let written = 0;

    while (written < bytes.length) {
      written += writeSync(this.#fd, bytes, written, bytes.length - written);
    }
  }

  close(): void {
    try {
      this.flush();
    } finally {
      closeSync(this.#fd);
    }
  }
}

/** Sequential line reader over one run file with one fixed read buffer. */
class RunReader {
  readonly #path: string;
  readonly #fd: number;
  readonly #buffer = Buffer.allocUnsafe(READ_BUFFER_BYTES);
  readonly #decoder = new StringDecoder("utf8");
  #lines: string[] = [];
  #index = 0;
  #pending = "";
  #done = false;
  #closed = false;

  constructor(path: string) {
    this.#path = path;
    this.#fd = openSync(path, "r");
  }

  next(): string | undefined {
    while (this.#index === this.#lines.length) {
      if (this.#done) return undefined;
      const read = readSync(this.#fd, this.#buffer, 0, this.#buffer.length, null);

      if (read === 0) {
        this.#done = true;

        if (this.#pending + this.#decoder.end()) {
          throw new Error(`Spool run is not newline-terminated: ${this.#path}`);
        }

        return undefined;
      }

      const text = this.#pending + this.#decoder.write(this.#buffer.subarray(0, read));

      this.#lines = text.split("\n");
      this.#pending = this.#lines.pop()!;
      this.#index = 0;
    }

    return this.#lines[this.#index++];
  }

  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    closeSync(this.#fd);
  }
}

/**
 * Every `hash\tline` of the sorted runs, in ascending order, one at a time.
 * Hashes are unique across all runs, so an equal key is a duplicate row.
 * Closing the generator early closes every run.
 */
function* mergeRuns(paths: readonly string[]): Generator<string> {
  const readers: RunReader[] = [];

  try {
    for (const path of paths) readers.push(new RunReader(path));
    const heads = readers.map((reader) => reader.next());
    const heap = heads.flatMap((head, index) => (head === undefined ? [] : [index]));
    const less = (a: number, b: number) => heads[a]! < heads[b]!;

    const siftDown = (start: number) => {
      let index = start;

      for (;;) {
        const left = index * 2 + 1;

        if (left >= heap.length) return;
        const right = left + 1;
        const child = right < heap.length && less(heap[right]!, heap[left]!) ? right : left;

        if (!less(heap[child]!, heap[index]!)) return;
        [heap[index], heap[child]] = [heap[child]!, heap[index]!];
        index = child;
      }
    };

    for (let index = (heap.length >> 1) - 1; index >= 0; index--) siftDown(index);
    let previous = "";

    while (heap.length > 0) {
      const source = heap[0]!;
      const entry = heads[source]!;
      const key = entry.slice(0, KEY_CHARACTERS);

      if (key === previous) throw new Error(`Duplicate spool hash: ${key}`);
      previous = key;
      yield entry;
      heads[source] = readers[source]!.next();

      if (heads[source] === undefined) {
        heap[0] = heap[heap.length - 1]!;
        heap.pop();
      }

      siftDown(0);
    }
  } finally {
    for (const reader of readers) reader.close();
  }
}

export class HashOrderedSpool {
  readonly #directory: string;
  readonly #runCharacters: number;
  readonly #mergeFanIn: number;
  /** The only in-memory rows: at most about `runCharacters` of them. */
  #run: string[] = [];
  #queuedCharacters = 0;
  /** Run file paths, about one per `runCharacters` of corpus text. */
  #runs: string[] = [];
  #files = 0;
  #draining = false;

  /** `directory` is owned by the spool and replaced if it exists. */
  constructor(directory: string, options: HashOrderedSpoolOptions = {}) {
    const runCharacters = options.runCharacters ?? DEFAULT_RUN_CHARACTERS;
    const mergeFanIn = options.mergeFanIn ?? DEFAULT_MERGE_FAN_IN;

    if (!Number.isSafeInteger(runCharacters) || runCharacters < 1) {
      throw new RangeError("Spool runCharacters must be a positive integer");
    }

    if (!Number.isInteger(mergeFanIn) || mergeFanIn < 2) {
      throw new RangeError("Spool mergeFanIn must be an integer of at least 2");
    }

    rmSync(directory, { recursive: true, force: true });
    mkdirSync(directory, { recursive: true });
    this.#directory = directory;
    this.#runCharacters = runCharacters;
    this.#mergeFanIn = mergeFanIn;
  }

  /** Characters held in memory; below the run budget after every `add`. */
  get queuedCharacters(): number {
    return this.#queuedCharacters;
  }

  /** Sorted runs currently on disk. */
  get runCount(): number {
    return this.#runs.length;
  }

  #nextPath(prefix: string): string {
    const name = `${prefix}-${String(this.#files++).padStart(6, "0")}.tsv`;

    return join(this.#directory, name);
  }

  /** Queue one line under its lowercase hex SHA-256 key. */
  add(hash: string, line: string): void {
    if (this.#draining) throw new Error("Hash-ordered spool is already draining");
    if (!HASH.test(hash)) throw new Error(`Invalid spool hash: ${hash}`);
    if (line.includes("\n")) throw new Error("Spool lines must not contain newlines");
    // The key is fixed-width, so plain string order is hash order.
    const entry = `${hash}\t${line}`;

    this.#run.push(entry);
    this.#queuedCharacters += entry.length;
    if (this.#queuedCharacters >= this.#runCharacters) this.#spill();
  }

  /** Take the in-memory run, sorted and checked for duplicate keys. */
  #takeSortedRun(): string[] {
    const run = this.#run;

    this.#run = [];
    this.#queuedCharacters = 0;
    run.sort();

    for (let index = 1; index < run.length; index++) {
      const key = run[index - 1]!.slice(0, KEY_CHARACTERS);

      if (run[index]!.startsWith(key)) throw new Error(`Duplicate spool hash: ${key}`);
    }

    return run;
  }

  #spill(): void {
    if (this.#run.length === 0) return;
    const run = this.#takeSortedRun();
    const path = this.#nextPath("run");
    const writer = new RunWriter(path);

    try {
      for (const entry of run) writer.line(entry);
    } finally {
      writer.close();
    }

    this.#runs.push(path);
  }

  /** Merge groups of runs until at most `mergeFanIn` remain. */
  #reduceRuns(): void {
    while (this.#runs.length > this.#mergeFanIn) {
      const next: string[] = [];

      for (let start = 0; start < this.#runs.length; start += this.#mergeFanIn) {
        const group = this.#runs.slice(start, start + this.#mergeFanIn);
        const path = this.#nextPath("merge");
        const writer = new RunWriter(path);

        try {
          for (const entry of mergeRuns(group)) writer.line(entry);
        } finally {
          writer.close();
        }

        for (const run of group) unlinkSync(run);
        next.push(path);
      }

      this.#runs = next;
    }
  }

  /**
   * Every queued line, newline-terminated, in ascending hash order. Rows are
   * pulled one at a time; at most one output chunk is built between yields.
   */
  async *chunks(): AsyncGenerator<Buffer> {
    if (this.#draining) throw new Error("Hash-ordered spool can only drain once");
    this.#draining = true;
    let entries: Iterable<string>;

    if (this.#runs.length === 0) {
      // Everything fit in one run: no disk round trip.
      entries = this.#takeSortedRun();
    } else {
      this.#spill();
      this.#reduceRuns();
      entries = mergeRuns(this.#runs);
    }

    let pending: string[] = [];
    let characters = 0;

    for (const entry of entries) {
      const line = entry.slice(KEY_CHARACTERS + 1);

      pending.push(line, "\n");
      characters += line.length + 1;

      if (characters >= WRITE_CHARACTERS) {
        const chunk = Buffer.from(pending.join(""), "utf8");

        pending = [];
        characters = 0;
        yield chunk;
      }
    }

    if (characters > 0) yield Buffer.from(pending.join(""), "utf8");
  }

  /** Delete the spool directory. Safe to call more than once. */
  remove(): void {
    rmSync(this.#directory, { recursive: true, force: true });
  }
}
