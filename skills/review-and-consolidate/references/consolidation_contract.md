# Consolidation Contract

## CLI

Required:

- `--project-root`

Core:

- `--graphs-dir`
- `--apply`
- `--dry-run`
- `--stdout-json`
- `--archive-tasks`
- `--supersede-decisions`
- `--resolve-conflicts`
- `--archive-age-days`
- `--context-pack-history-limit`
- `--consolidation-history-limit`

Optional:

- `--terminal-task-statuses`
- `--active-task-statuses`
- `--create-missing-conflict-task`
- `--strict-conflicts`
- `--request-id`
- `--source-title`

## Suggest output

Required fields:

- `consolidation_run_id`
- `generated_at`
- `scope`
- `candidate_actions`
- `warnings`
- `stats`

Optional fields:

- `request_id`
- `consolidation_seq`
- `commit_id`
- `context_pack_path`
- `consolidation_history_path`

## Apply commit row

For apply with actual mutations, append one row to `post_response_commits.jsonl` with:

- `mode = review_consolidate_apply`
- `event = post_response_commit`
- `commit_id`
- `consolidation_run_id`
- `consolidation_seq`
- `changed_node_ids`
- `actions_summary`
- `context_pack_path`
- `created_at`
- optional `request_id`

## History artifact

Write apply-only history artifact when changes exist:

`graphs/consolidation_runs/consolidation_run.<timestamp>.<consolidation_run_id>.json`

Contains at least:

- identity (`consolidation_run_id`, `consolidation_seq`, `generated_at`)
- action plan/applied summary
- warnings/stats
- changed node ids
- commit/context linkage
