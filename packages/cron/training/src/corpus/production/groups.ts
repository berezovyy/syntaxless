import { sha256Hex } from "../../fingerprint";
import type { Split } from "../../generate";

/** Sparse union-find; canonical root is always the smallest source ID. */
export class SourceGroups {
  readonly #parent = new Map<number, number>();
  find(id: number): number {
    const parent = this.#parent.get(id);

    if (parent === undefined || parent === id) return id;
    // Iterative compression avoids stack exhaustion on adversarial link order.
    let root = parent;

    while (this.#parent.has(root) && this.#parent.get(root) !== root) {
      root = this.#parent.get(root)!;
    }

    let current = id;

    while (current !== root) {
      const next = this.#parent.get(current) ?? root;

      this.#parent.set(current, root);
      current = next;
    }

    return root;
  }

  union(a: number, b: number): void {
    const left = this.find(a);
    const right = this.find(b);

    if (left !== right) this.#parent.set(Math.max(left, right), Math.min(left, right));
  }
}

/** Value variants and linked records are assigned together; no row-level resplit. */
export function sourceGroupSplit(sourceId: string, groupId: string): Split {
  const value = Number.parseInt(sha256Hex(`${sourceId}\0${groupId}`).slice(0, 8), 16);

  return value < 0xc0000000 ? "train" : value < 0xe0000000 ? "validation" : "heldout";
}
