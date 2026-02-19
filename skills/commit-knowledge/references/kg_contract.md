# KG Commit Contract

## Suggestion output

Required top-level fields:
- `knowledge_commit_id`
- `knowledge_seq`
- `generated_at`
- `source`
- `candidates`
- `warnings`
- `stats`

Optional top-level fields:
- `request_id`
- `source_request_hash_raw`
- `source_request_hash_norm`
- `conflict_candidates`
- `suggested_conflict_tasks`

## Source object

Required fields:
- `source_key`
- `source_digest`
- `source_kind`
- `source_title`

Optional:
- `source_node_id`

## Candidate object

Required fields:
- `temp_id`
- `type` (`fact|definition|constraint|assumption`)
- `title`
- `summary`
- `status`
- `confidence`
- `claim_hash`
- `knowledge_key`
- `tags`
- `links`
- `evidence`

Evidence item required fields:
- `source_id`
- `confidence`
- `source_digest`
- `span.start_line`
- `span.end_line`
- `span.start_char`
- `span.end_char`
- `excerpt`

## Apply mode writes

KG rows are append-only with:
- `rev` increment semantics
- `created_by="commit-knowledge"`
- `origin="conversation_knowledge_commit"`
- `origin_knowledge_commit_id`
- `knowledge_seq`

Optional:
- `request_id`
- `source_request_hash_raw`
- `source_request_hash_norm`

## Conflict write flags

- `--apply-conflicts true`: append conflict records.
- `--apply-conflict-task true`: append TG resolution tasks (requires `--apply true` and `--apply-conflicts true`).

Conflict task creation is never implicit.

## Post-response sync order

When apply produces changed KG/TG rows:
1. build post-response context pack
2. write context history file
3. validate written history JSON
4. atomically replace `context_pack.latest.json`
5. append `post_response_commits.jsonl` row last

No rollback is attempted for commit-log append failures.
