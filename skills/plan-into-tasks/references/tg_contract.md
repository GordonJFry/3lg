# TG Planning Contract

## Suggestion output (top-level)

Required fields:

- `plan_id`
- `plan_seq`
- `generated_at`
- `intent`
- `tasks`
- `warnings`
- `link_suggestions`

Optional fields:

- `request_id`
- `source_request_hash`

## Task object contract

Required task fields:

- `temp_id`
- `type` (always `task`)
- `title`
- `description`
- `phase`
- `status`
- `priority`
- `owner_role`
- `dependencies`
- `blocks`
- `acceptance_criteria`
- `inputs`
- `outputs`
- `tags`

## Apply mode contract

Applied TG nodes must include:

- `rev = 1`
- `created_by = "plan-into-tasks"`
- `origin = "request_task_plan"`
- `origin_plan_id = <plan_id>`
- `plan_seq`
- optional `request_id`
- optional `source_request_hash`

## Idempotency

- Before apply, scan latest TG nodes for `origin_plan_id`.
- If already applied and `--force false`, fail.
- If `--force true`, allow and emit warning.

## Dependencies in apply mode

- `TEMP-*` dependencies must map to newly created `TG-TASK-*` IDs.
- Applied dependencies must contain only TG IDs.
- Unresolved dependency behavior:
  - strict mode: fail
  - non-strict mode: drop unresolved dep and emit warning

## Commit logging

On apply, append post-response commit entry with:

- `mode = "plan_into_tasks_apply"`
- `plan_id`
- `plan_seq`
- optional `request_id`
- optional `source_request_hash`
- `changed_node_ids`
