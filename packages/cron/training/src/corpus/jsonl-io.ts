import { createHash } from "node:crypto";
import { once } from "node:events";
/**
 * Streaming JSONL I/O for plain files and canonical gzip artifacts.
 *
 * Every reader here holds one bounded buffer: corpus-sized files are never
 * read whole, decompressed whole, or split into a line array.
 */
import { closeSync, createReadStream, createWriteStream, openSync, readSync } from "node:fs";
import { createInterface } from "node:readline";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { StringDecoder } from "node:string_decoder";
import { constants, createGunzip, createGzip } from "node:zlib";

export type ArtifactChunkVisitor = (chunk: Buffer) => void;

/**
 * Lines of a plain (uncompressed) text file, read synchronously through one
 * fixed buffer. Use this instead of `readFileSync(...).split("\n")` for any
 * input whose size is not fixed by the code.
 */
export function* readLinesSync(path: string, bufferBytes = 64 * 1024): Generator<string> {
  const fd = openSync(path, "r");
  const buffer = Buffer.allocUnsafe(bufferBytes);
  const decoder = new StringDecoder("utf8");
  let pending = "";

  try {
    for (;;) {
      const read = readSync(fd, buffer, 0, buffer.length, null);

      if (read === 0) break;
      const lines = (pending + decoder.write(buffer.subarray(0, read))).split("\n");

      pending = lines.pop()!;
      yield* lines;
    }

    pending += decoder.end();
    if (pending) yield pending;
  } finally {
    closeSync(fd);
  }
}

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

  try {
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

    try {
      source.once("error", (error) => gunzip.destroy(error));
      source.pipe(gunzip);
      for await (const value of gunzip) yield Buffer.isBuffer(value) ? value : Buffer.from(value);
    } finally {
      gunzip.destroy();
    }
  } finally {
    // Iterator cancellation must finish closing the file before returning.
    const closed = source.closed
      ? undefined
      : new Promise<void>((resolve) => source.once("close", resolve));

    source.destroy();
    await closed;
  }
}

/** Own a line reader with `await using`; consume readline's iterator directly. */
export function jsonlReader(chunks: AsyncIterable<Buffer>) {
  const input = Readable.from(chunks);
  const lines = createInterface({ input, crlfDelay: Number.POSITIVE_INFINITY });
  // Register the iterator before input can emit lines or errors.
  const iterator = lines[Symbol.asyncIterator]();

  return {
    lines: iterator,
    async [Symbol.asyncDispose](): Promise<void> {
      lines.close();
      const closed = input.closed ? undefined : once(input, "close");

      input.destroy();
      await closed;
    },
  };
}

/** One awaited pipeline owns gzip output, backpressure, failure and file closure. */
export async function writeGzip(
  path: string,
  chunks: Iterable<Buffer> | AsyncIterable<Buffer>,
): Promise<{ bytes: number; sha256: string }> {
  const gzip = createGzip({ level: constants.Z_BEST_SPEED });
  const hash = createHash("sha256");
  let bytes = 0;

  gzip.on("data", (chunk: Buffer) => {
    hash.update(chunk);
    bytes += chunk.length;
  });

  await pipeline(
    Readable.from(chunks, { objectMode: false }),
    gzip,
    createWriteStream(path, { flags: "wx" }),
  );

  return { bytes, sha256: hash.digest("hex") };
}
