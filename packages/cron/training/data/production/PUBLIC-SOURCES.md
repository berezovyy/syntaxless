# Public-source investigation

## Tatoeba English detailed export, 2026-09-19 snapshot

The production expansion uses the detailed public export so source sentence IDs
and the listed owner can be retained, rather than keeping only anonymous text.
This is ordinary-English source material, **not consented real-user scheduling
requests**, and public-source provenance does not establish human authorship of
every individual sentence.

- Export: <https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences_detailed.tsv.bz2>
- Official export index: <https://downloads.tatoeba.org/exports/per_language/eng/>
- Official file-format and licensing reference:
  <https://tatoeba.org/ca/downloads> (English page returned HTTP 429 during this
  check; the alternate locale exposes the same official download documentation).
- Export license stated by Tatoeba: **CC BY 2.0 FR**. Some sentences have a separate
  CC0 release, but the general detailed export is not being represented as CC0.
- Expected attribution: Tatoeba contributors, <https://tatoeba.org>, with a link
  to each retained sentence and the listed owner when available. A listed owner
  is not asserted to be the original author. Normalization changes must be noted.
- Retrieved on 2026-09-20; bzip2 integrity check passed.
- Archive SHA-256:
  `028ba8e2b8ae3a5e1694e95d901a3f50b9eb8fcdf6ebe033ce6ab265ec183468`.
- Preserved archive (ignored generated-data area):
  `data/synth/production-sources/tatoeba-20260919/eng_sentences_detailed.tsv.bz2`.

## Published expanded source pool

The old 50,000-sentence pin is unchanged. The complete import inspected
**2,036,986 records** and published **1,632,327 distinct eligible sentences**
in `data/synth/production-sources/tatoeba-english-v2/`. The earlier exploratory
count of 1,632,355 used the older raw-text filter: 28 of those candidates were
excluded after full validation exposed normalized schedule words or unstable
normalization. `easy-no-v2` checks both raw and runtime-normalized wording.

The pool contains 1,225,140 training, 202,819 validation, and 204,368 held-out
rows, assigned by **1,525,639 related-source components**. It retains source IDs,
original text, listed owners, dates, licensing and normalization disclosure.
All 1,632,327 records passed the integrity audit and a separate complete-stream
JSONL/SQLite readback; artifact sizes and SHA-256 digests matched before and after
verification. This is a **source pool**, not the final training mixture.

### Related-source grouping

The importer also pins the official English-English sentence-link export:

- URL: <https://downloads.tatoeba.org/exports/per_language/eng/eng-eng_links.tsv.bz2>
- SHA-256: `cefa98434db0bac2d2ade367d072b407fe3e9d7d382b08994db13e61cd73b787`.
- 246,084 directed link rows are merged transitively, including filtered-out
  sentences that connect eligible variants.
- Normalized duplicate identities and configured numeric/common-name variants
  also share one component. Components are formed before split assignment;
  protected matches exclude the related component.
- The largest retained component has 16 rows. Unlinked paraphrases may remain
  undiscovered, so this is not a claim of independent acceptance-data isolation.

### Concentration and historical selection evidence

Distinct wording is not automatically diverse language. The largest listed
owner supplies 607,994 rows (about 37.25% of the eligible pool); 36,724 have no
listed owner. Owner names are not proof of original authorship.

The retired quota-selection implementation previously demonstrated that 208,301
rows could be selected twice with the same ordered-ID hash under 10% owner and
opening-word caps and a four-row related-group cap. This remains historical
diversity evidence, not a production target. The complete-corpus architecture
keeps every eligible unique public row; training-time sampling controls R/H/E
exposure without removing source records.

Reports: `results/production-tatoeba-v2-readback.json` and
`results/production-public-selection-feasibility.json`. The published pool's
`manifest.json` records exact source hashes, exclusions, counts and attribution.

### Reproduction evidence

After repairing the decompression reader, a second complete build reproduced
all 1,632,327 rows with byte-identical `corpus.jsonl`, `source.sqlite`, and
`validation.json`. Both versions' artifact lengths and hashes were independently
checked. The manifests retain their respective implementation hashes rather
than pretending no code changed. See `results/production-tatoeba-v2-reproduction.json`.
This does not substitute for final mixed-corpus and shard reproducibility.
