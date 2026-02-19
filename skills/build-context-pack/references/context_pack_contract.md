# Request Context Pack Contract

## Purpose

Request-time Context Packs are derived runtime views. KG/RG/TG node logs remain the source of truth.

## Required fields

- `request_id`
- `request_seq`
- `generated_at`
- `intent`
- `entities`
- `goal`
- `active_tasks`
- `task_snapshot`
- `knowledge_snapshot`
- `decision_snapshot`
- `recent_deltas`

## Optional fields

- `commit_id`
- `clarify_suggestion`
- `clarify_applied_task_id`

## Persistence

History file path pattern:

- `graphs/request_context_packs/request_context.<timestamp>.<request_id>.json`

Latest pointer file:

- `graphs/request_context.latest.json`

Write order:

1. write history file
2. validate written JSON
3. atomically replace latest pointer

## Request sequence allocation

- `request_seq = max(existing request_seq) + 1`
- allocate after reading history and before write
- if conflict is detected after write, retry allocation once
- if conflict persists, fail with explicit error

## Clarify task mutation contract

When `--apply-clarify-task true`, append TG task with:

- `rev = 1`
- `created_by = "build-context-pack"`
- `origin = "clarify_suggestion"`
- tag `clarify`

Then rebuild and verify indexes.
