/** Memory-bounded exact set for SHA-256 digests. */

const WORDS_PER_HASH = 8;
const MAX_LOAD = 0.7;

function capacityFor(expectedEntries: number): number {
  if (!Number.isSafeInteger(expectedEntries) || expectedEntries < 0) {
    throw new Error(`Invalid expected hash-set size: ${expectedEntries}`);
  }

  const required = Math.max(2, Math.ceil(expectedEntries / MAX_LOAD));
  let capacity = 1;

  while (capacity < required) capacity *= 2;

  if (!Number.isSafeInteger(capacity * WORDS_PER_HASH)) {
    throw new Error(`Hash set is too large: ${expectedEntries}`);
  }

  return capacity;
}

/**
 * Open-addressed storage for complete 256-bit digests.
 *
 * JavaScript `Set<string>` retains tens of millions of separately allocated
 * strings and exceeds the production host's memory. This stores the same
 * identities as eight packed uint32 words plus one occupancy bit per slot.
 * Full digests are compared, so this does not weaken exact deduplication.
 */
export class Hash256Set {
  readonly #capacity: number;
  readonly #mask: number;
  readonly #words: Uint32Array;
  readonly #occupied: Uint32Array;
  #size = 0;

  constructor(expectedEntries: number) {
    this.#capacity = capacityFor(expectedEntries);
    this.#mask = this.#capacity - 1;
    this.#words = new Uint32Array(this.#capacity * WORDS_PER_HASH);
    this.#occupied = new Uint32Array(Math.ceil(this.#capacity / 32));
  }

  get size(): number {
    return this.#size;
  }

  #isOccupied(slot: number): boolean {
    return (this.#occupied[slot >>> 5]! & (1 << (slot & 31))) !== 0;
  }

  #markOccupied(slot: number): void {
    this.#occupied[slot >>> 5] = this.#occupied[slot >>> 5]! | (1 << (slot & 31));
  }

  #equals(slot: number, digest: Buffer): boolean {
    const offset = slot * WORDS_PER_HASH;

    for (let word = 0; word < WORDS_PER_HASH; word++) {
      if (this.#words[offset + word] !== digest.readUInt32LE(word * 4)) return false;
    }

    return true;
  }

  #slot(digest: Buffer): { slot: number; present: boolean } {
    if (digest.length !== 32) {
      throw new Error(`Hash256Set requires a 32-byte digest, got ${digest.length}`);
    }

    let slot = digest.readUInt32LE(0) & this.#mask;

    for (;;) {
      if (!this.#isOccupied(slot)) return { slot, present: false };
      if (this.#equals(slot, digest)) return { slot, present: true };
      slot = (slot + 1) & this.#mask;
    }
  }

  has(digest: Buffer): boolean {
    return this.#slot(digest).present;
  }

  add(digest: Buffer): boolean {
    const found = this.#slot(digest);

    if (found.present) return false;
    const offset = found.slot * WORDS_PER_HASH;

    for (let word = 0; word < WORDS_PER_HASH; word++) {
      this.#words[offset + word] = digest.readUInt32LE(word * 4);
    }

    this.#markOccupied(found.slot);
    this.#size++;

    return true;
  }
}
