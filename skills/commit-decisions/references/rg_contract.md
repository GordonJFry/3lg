# RG Decision Contract

## Suggestion output

Required top-level fields:
- `decision_commit_id`
- `decision_seq`
- `generated_at`
- `source`
- `decision_candidate`
- `warnings`
- `stats`

Optional top-level fields:
- `request_id`
- `source_request_hash_raw`
- `source_request_hash_norm`
- `link_suggestions`
- `followup_task_suggestion`

## decision_candidate fields

- `decision_key`
- `decision_payload_hash`
- `title`
- `context`
- `decision`
- `alternatives` (`option_id`, `title`, `pros`, `cons`)
- `selected_option` (`option_id` or `undecided`)
- `rationale`
- `assumptions`
- `risks`
- `validation`
- `refs` (`kg_refs`, `tg_refs`, `artifacts`)
- `links`
- `tags`
- `outcome` (`status`, `supersedes`, `notes`, optional `supersedes_rev`)

## Apply writes

RG row requirements:
- append-only, never edit prior lines
- `rev` starts at 1 and increments by 1 for updates
- include required RG schema fields
- include deterministic metadata in attributes:
  - `decision_key`
  - `decision_payload_hash`
  - `selected_option`
  - source hashes

Optional apply side effect:
- when undecided and `--apply-followup-task true`, append TG task tagged `decision_followup`

## Commit row requirements

For apply with changes, post_response_commits row must include:
- `mode`, `commit_id`
- `decision_commit_id`, `decision_seq`
- `decision_id`, `decision_key`
- `outcome_status`
- `changed_node_ids`, `context_pack_path`
- optional request/hash fields

## Strictness behavior

- Suggest mode: missing refs produce warnings only.
- Apply mode: enforce `--require-kg-tg` if true.
