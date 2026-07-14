import { once } from "node:events";
/** Streaming JSONL I/O for plain files and canonical gzip artifacts. */
import { createReadStream, createWriteStream } from "node:fs";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { constants, createGunzip, createGzip } from "node:zlib";

export type ArtifactChunkVisitor = (chunk: Buffer) => void;
const GZIP_INPUT_BUFFER_BYTES = 4 * 1024 * 1024;

/**
 * Yield decompressed bytes while optionally observing the exact artifact bytes.
 * The observer therefore hashes compressed bytes for `.gz`, not an invented
 * uncompressed representation.
 */
export async function* jsonlByteChunks(
  path: string,
  visitArtifactChunk?: ArtifactChunkVisitor,
): AsyncGenerator<Buffer> {
  const source = createReadStream(path);

  if (!path.endsWith(".gz")) {
    for await (const value of source) {
      const chunk = Buffer.isBuffer(value) ? value : Buffer.from(value);

      visitArtifactChunk?.(chunk);
      yield chunk;
    }

    return;
  }

  if (visitArtifactChunk) {
    source.on("data", (value: Buffer | string) =>
      visitArtifactChunk(Buffer.isBuffer(value) ? value : Buffer.from(value)),
    );
  }

  const gunzip = createGunzip();

  source.once("error", (error) => gunzip.destroy(error));
  source.pipe(gunzip);
  for await (const value of gunzip) yield Buffer.isBuffer(value) ? value : Buffer.from(value);
}

/** A Node readable suitable for readline and other existing streaming APIs. */
export function createJsonlReadStream(path: string): Readable {
  return Readable.from(jsonlByteChunks(path));
}

/**
 * Direct-to-disk deterministic gzip writer. No uncompressed temporary exists.
 * `append` observes backpressure, so callers can stream arbitrarily large files.
 */
export class CanonicalGzipWriter {
  readonly path: string;
  readonly #gzip;
  readonly #done: Promise<void>;
  #chunks: Buffer[] = [];
  #bytes = 0;
  #closed = false;

  constructor(path: string) {
    if (!path.endsWith(".gz")) throw new Error(`Canonical gzip output must end in .gz: ${path}`);
    this.path = path;
    this.#gzip = createGzip({ level: constants.Z_BEST_SPEED });
    this.#done = pipeline(this.#gzip, createWriteStream(path, { flags: "wx" }));
  }

  append(value: string | Buffer): Promise<void> | undefined {
    if (this.#closed) throw new Error("Cannot append to a closed gzip writer");
    const chunk = Buffer.isBuffer(value) ? value : Buffer.from(value);

    this.#chunks.push(chunk);
    this.#bytes += chunk.length;
    if (this.#bytes >= GZIP_INPUT_BUFFER_BYTES) return this.#flush();
  }

  #flush(): Promise<void> | undefined {
    if (this.#bytes === 0) return;
    const chunk = Buffer.concat(this.#chunks, this.#bytes);

    this.#chunks = [];
    this.#bytes = 0;
    if (!this.#gzip.write(chunk)) return once(this.#gzip, "drain").then(() => undefined);
  }

  async close(): Promise<void> {
    if (this.#closed) throw new Error("Gzip writer is already closed");
    this.#closed = true;
    await this.#flush();
    this.#gzip.end();
    await this.#done;
  }

  async abort(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    this.#chunks = [];
    this.#bytes = 0;
    this.#gzip.destroy(new Error("Gzip writer aborted"));

    try {
      await this.#done;
    } catch {
      // Aborting is cleanup; the staging-directory owner removes the partial.
    }
  }
}
