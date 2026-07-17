/** Clean catalog traversal: no carrier Cartesian product; optional value sampling. */
import { fingerprintExample, sha256Hex } from "../../fingerprint";
import { type KindDef, KINDS } from "../../kinds";
import { normalizeAnnotatedText, Sentence } from "../../sentence";
import type { CorpusRecord } from "../merge";

export const CATALOG_SOURCE_ID = "systematic-catalog-v1";

export interface CatalogCandidate {
  kind: KindDef["id"];
  intentId: string;
  constructionId: string;
  /** No production split has been assigned. Related constructions need grouping first. */
  record: Omit<CorpusRecord, "split">;
}

export interface CatalogTraversal {
  kind: KindDef["id"];
  intentId: string;
  constructionId: string;
  rendered: boolean;
}

/**
 * Speech wrappers and corrections are supplied by the reviewed whole-utterance
 * source. This source preserves every finite schedule value and clean template
 * from the existing catalog, including templates formerly marked heldout.
 * Old generator splits are not valid ownership keys for a newly grouped corpus.
 *
 * `include` selects (template, intent) pairs before rendering. Intents are
 * full value grids (every minute of the day × every weekday set), so a
 * per-template sample keeps every template while skipping most of the grid.
 */
export function* catalogCandidates(
  kinds: readonly KindDef[] = KINDS,
  visit?: (attempt: CatalogTraversal) => void,
  include?: (constructionId: string, intentId: string) => boolean,
): Generator<CatalogCandidate> {
  for (const kind of kinds) {
    for (const intent of kind.intents) {
      if (intent === null) continue; // Legacy discourse/all-O examples are not kernels.
      const intentId = sha256Hex(JSON.stringify(intent));

      for (const template of kind.templates) {
        const constructionId = `${kind.id}/${template.id}`;

        if (include !== undefined && !include(constructionId, intentId)) continue;
        const sentence = new Sentence();

        sentence.clause();
        const rendered = kind.render(sentence, intent, template.id);

        visit?.({ kind: kind.id, intentId, constructionId, rendered });
        if (!rendered) continue;
        const annotated = normalizeAnnotatedText(sentence.text, sentence.spans);
        const example = { ...annotated, speechAct: "request" as const };

        yield {
          kind: kind.id,
          intentId,
          constructionId,
          record: {
            id: `${CATALOG_SOURCE_ID}/${constructionId}/${intentId}`,
            family: kind.family,
            ...example,
            intent: structuredClone(intent),
            source: "rule",
            template: constructionId,
            fingerprint: fingerprintExample(example),
            provenance: {
              authorship: "programmatic",
              sourceId: CATALOG_SOURCE_ID,
              // Provisional template key, NOT yet a collision-unioned split group.
              groupId: constructionId,
              constructionId,
              sourceRecordId: intentId,
            },
          },
        };
      }
    }
  }
}
