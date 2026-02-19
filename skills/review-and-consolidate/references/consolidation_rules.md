# Consolidation Rules

## Global invariants

- Append-only JSONL only; never edit prior lines.
- Updates append same `id` with incremented `rev`.
- Latest resolver order:
  - if all candidates have integer `rev`: `rev` desc, then `updated_at` desc, then line order desc
  - if any candidate is missing `rev`: `updated_at` desc, then line order desc
- Suggest mode never mutates graph state.
- Apply no-op returns success without commit/context/history writes.

## Archive task rules

A TG task is archive-eligible only when all are true:

- latest status is in terminal set (`done|completed|cancelled|canceled|superseded` by default; `canceled` normalizes to `cancelled`)
- age threshold passes (`updated_at <= now - archive_age_days`)
- task is not referenced by active TG dependency/block links
- task is not referenced by unresolved conflict records
- task is not referenced in latest resolved view of any RG decision `refs.tg_refs`

Archive apply result:

- append TG row with same `id`, `rev+1`, `status=archived`
- append archive log event and metadata (`origin=task_archive`)

## Supersede decision rules

- Group only by exact `decision_key`.
- Skip decisions missing `decision_key` and emit warning.
- Canonical selection:
  - latest accepted in group
  - else latest non-superseded
  - else latest overall
- Non-canonical rows are superseded by appending same `id` with `rev+1` and:
  - `outcome.status = superseded`
  - `outcome.supersedes` includes canonical `id`
  - deterministic supersede note

## Resolve conflict rules

Conflict is resolvable only when all are true:

- conflict status is unresolved (`status` not in `resolved|closed`)
- linked `resolution_task_ids` exist
- linked tasks are tagged `conflict_resolution`
- linked tasks are `done|completed`
- conflict evidence is non-empty

Missing resolution task behavior:

- default: warn + skip
- opt-in: create TG task when `--create-missing-conflict-task true`
- strict mode: fail apply when unresolved prerequisites exist

## Sync ordering

For apply runs with changes:

1. append TG/RG/conflict rows
2. rebuild and verify indexes
3. write context history
4. validate context history JSON
5. atomically replace `context_pack.latest.json`
6. append `post_response_commits.jsonl` row last
7. write consolidation history artifact

No rollback is attempted.
