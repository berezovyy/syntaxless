/** Preserve key priority and caller facts using predicted semantic roles. */
import { validate } from "./catalog.js";
import type { Role } from "./labels.js";
import type { Lexeme, Prepared } from "./prepare.js";
import { InputError } from "./types.js";
import type { Context, Direction, NullPlacement, SortIntent, SortKey } from "./types.js";

export function resolveRoles(
  prepared: Prepared,
  labels: readonly { role: Role; score: number }[],
  context: Context,
): { intent: SortIntent; confidence: number } {
  if (labels.length !== prepared.lexemes.length) {
    throw new Error("prediction length does not match input");
  }

  const groups: { lexeme: Lexeme; role: Role }[][] = [[]];
  let confidence = 1;

  for (const [index, lexeme] of prepared.lexemes.entries()) {
    const { role, score } = labels[index]!;

    if (!Number.isFinite(score) || score < 0 || score > 1) {
      throw new Error("invalid prediction confidence");
    }

    if (role === "UNSUPPORTED") {
      throw new InputError("invalid-input", "unsupported or non-actionable input");
    }

    if (
      (lexeme.field !== undefined && role !== "FIELD") ||
      (lexeme.direction !== undefined && role !== (lexeme.direction === "asc" ? "ASC" : "DESC"))
    ) {
      throw new InputError("invalid-input", "model did not preserve a caller field or direction");
    }

    if (role === "O") continue;
    confidence = Math.min(confidence, score);
    if (role === "GLUE") continue;

    if (role === "NEXT") {
      if (groups.at(-1)!.length) groups.push([]);
    } else {
      groups.at(-1)!.push({ lexeme, role });
    }
  }

  const pending: { field: string; direction: Direction; nulls?: NullPlacement }[] = [];

  for (const group of groups) {
    if (!group.length) continue;
    const mentions = group.filter((token) => token.role === "FIELD");

    if (mentions.length > 1) {
      throw new InputError("ambiguous", "separate sort keys and null placement clauses");
    }

    const mention = mentions[0];

    if (mention && mention.lexeme.field === undefined) {
      throw new InputError("unknown-field", "unknown field mention");
    }

    const directions = new Set(
      group
        .filter((token) => token.role === "ASC" || token.role === "DESC")
        .map((token) => token.role),
    );

    if (directions.size > 1) throw new InputError("conflict", "contradictory directions");

    const placements = new Set(
      group
        .filter((token) => token.role === "FIRST" || token.role === "LAST")
        .map((token) => token.role),
    );

    if (placements.size > 1) throw new InputError("conflict", "contradictory null placements");
    const hasNulls = group.some((token) => token.role === "NULLS");

    if (hasNulls !== (placements.size === 1)) {
      throw new InputError("missing-null-placement", "specify first or last for null values");
    }

    const direction: Direction | undefined = directions.size
      ? directions.has("ASC")
        ? "asc"
        : "desc"
      : undefined;

    const nulls: NullPlacement | undefined = hasNulls
      ? placements.has("FIRST")
        ? "first"
        : "last"
      : undefined;

    const field = mention?.lexeme.field;

    if (direction) {
      if (!field) throw new InputError("unknown-field", "provide a known field for the direction");

      if (pending.some((key) => key.field === field)) {
        throw new InputError("conflict", "duplicate field ordering");
      }

      pending.push({ field, direction, ...(nulls ? { nulls } : {}) });
    } else if (nulls) {
      const key = field
        ? pending.find((key) => key.field === field)
        : pending.length === 1
          ? pending[0]
          : undefined;

      if (!key) {
        throw new InputError("ambiguous", "identify an earlier sort key for null placement");
      }

      if (key.nulls !== undefined && key.nulls !== nulls) {
        throw new InputError("conflict", "contradictory null placements");
      }

      key.nulls = nulls;
    } else {
      throw new InputError(
        "missing-direction",
        "provide an explicit direction or caller ordering alias",
      );
    }
  }

  const keys: SortKey[] = pending.map((key) => {
    const nulls = key.nulls ?? context.defaultNulls;

    if (nulls === undefined) {
      throw new InputError(
        "missing-null-placement",
        `specify null placement for ${key.field} or a caller default`,
      );
    }

    return {
      ...key,
      nulls,
      nullsSource: key.nulls === undefined ? "caller-default" : "explicit",
    };
  });

  const intent: SortIntent = {
    keys,
    ...(context.tieBreaker === undefined ? {} : { tieBreaker: context.tieBreaker }),
  };

  validate(intent, context);

  return { intent, confidence };
}
