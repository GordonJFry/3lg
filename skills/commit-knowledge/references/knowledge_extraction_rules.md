# Knowledge Extraction Rules

## Candidate extraction

- Parse normalized conversation text line-by-line.
- Ignore empty/non-text lines.
- Extract candidate statements from meaningful lines.
- Use Unicode-safe text detection/tokenization (do not require Latin-only lines).

## Claim typing

- `definition`: statement contains definition markers (`means`, `refers to`, `defined as`).
- `constraint`: statement contains policy/modal markers (`must`, `never`, `do not`, `always`, `only`, `should`).
- `assumption`: statement contains uncertainty markers (`might`, `may`, `could`, `likely`, `assume`, `uncertain`).
- `fact`: default for declarative statements that are neither definition nor constraint nor assumption.
- Include multilingual marker support (including RU variants of definition/constraint/assumption markers).

## Confidence and filtering

- Apply confidence by type and textual signals.
- Filter non-assumptions by `--min-confidence`.
- Keep assumptions only when `--include-assumptions true`.

## Hashing rules

`source_request_hash_raw`:
1. trim leading/trailing whitespace
2. normalize newlines (`CRLF`/`CR` -> `LF`)
3. preserve case, punctuation, and interior spacing
4. SHA-256 over UTF-8

`source_request_hash_norm`:
1. trim
2. lowercase
3. replace non-alphanumeric with spaces
4. collapse repeated whitespace
5. SHA-256 over UTF-8

`source_digest` for provenance uses `source_request_hash_raw`.

## Scope and keys

- Base scope: `project::<normalized_project_name>`.
- If a KG entity is resolvable, append `::entity::<entity_id>`.

`knowledge_key`:
- `definition`: `def::<scope>::<normalized_title>`
- `constraint`: `constraint::<scope>::<normalized_title>`
- `fact`: `fact::<scope>::<normalized_title>`
- `assumption`: `assump::<scope>::<normalized_title>`

`normalized_title` uses lowercase, Unicode-safe non-word-to-space normalization, and collapsed spaces.

## Source upsert

- `KG-SOURCE` upsert key is `source_key`.
- same `source_key` + same `source_digest` => no source write
- same `source_key` + changed `source_digest` => append same source `id` with `rev+1`
- missing `source_key` => create new source node

## Knowledge upsert

- Upsert key is `knowledge_key`.
- unchanged semantic content => skip
- changed semantic content => append same `id` with `rev+1`
- missing key => create new KG node with `rev=1`

## Evidence requirements

Each candidate evidence entry includes:
- `source_id`
- `confidence`
- `source_digest`
- `span.start_line`, `span.end_line`, `span.start_char`, `span.end_char`
- `excerpt` (max 200 chars)

## Conflict detection

Conflict candidate when all are true:
- same `topic_key`
- different `knowledge_key`
- opposite polarity OR modal incompatibility (`always` vs `never`, `must` vs `must_not`, `should` vs `should_not`)

Default behavior is suggestion-only warnings.

- `--apply-conflicts true`: write conflict records
- `--apply-conflict-task true`: additionally create TG `conflict_resolution` task(s)
