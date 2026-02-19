# Retrieval Rules

## Intent

- `auto` classifier picks one of: `plan`, `execute`, `explain`, `debug`, `research`.
- Explicit `--intent` overrides classifier.
- Auto classifier uses multilingual keyword sets (EN + RU markers).

## Entity extraction

- Match direct IDs: `KG-*`, `RG-*`, `TG-*`.
- Match lexical overlap against node `title`, `tags`, `type`, and selected attribute text.

## Latest-node resolver

For records with same `id`:

1. If all candidates have integer `rev`, pick highest `rev`.
2. If any candidate is missing `rev`, fallback to newest `updated_at`.
3. If still tied, pick latest line order.

## Critical info missing

`critical_info_missing` is true when any condition holds:

- `entities[]` is empty and intent in `{plan, execute, debug, research}`.
- TG candidate set is empty and intent in `{execute, debug}`.
- KG candidate set is empty and intent is not `explain`.

## Deterministic scoring

Additive score components:

- `+5` direct entity match
- `+4` referenced by selected TG
- `+3` referenced by accepted RG
- `+2` tag in `{constraint, risk}`
- recency buckets:
  - updated within 1 day: `+2`
  - updated within 7 days: `+1`
  - older: `+0`

Deterministic tie-break:

1. score descending
2. `updated_at` descending
3. `id` ascending

## Size caps

- TG <= `max_tasks` (default 15, with dependencies + blockers closure)
- KG <= `max_kg` (default 30)
- RG <= `max_rg` (default 10)
- Recent deltas <= `max_deltas` (default 20)

## Delta anchor priority

1. provided `--commit-id`
2. latest commit in `post_response_commits.jsonl`
3. prior `request_seq`

## Clarify behavior

- Clarify suggestion is emitted only when critical info is missing and intent is not `explain`.
- TG mutation occurs only with `--apply-clarify-task true`.
- When clarify task is applied, post-response sync is required:
  1. write `context_packs/context_pack.<timestamp>.<commit_id>.json`
  2. validate written JSON
  3. atomically replace `context_pack.latest.json`
  4. append `post_response_commits.jsonl` row last
