# 3-Layer Graph Store

This directory contains the append-only graph memory store:

- `kg_nodes.jsonl`, `rg_nodes.jsonl`, `tg_nodes.jsonl` are authoritative node logs.
- `kg_index.jsonl`, `rg_index.jsonl`, `tg_index.jsonl` are derived artifacts.
- `meta.json` stores store/version configuration.
- `post_response_commits.jsonl` stores commit records.
- `context_pack.latest.json` mirrors the newest commit context.
- `context_packs/` stores historical context pack snapshots.
- `conflicts.jsonl` stores conflict objects only.

## Invariants

1. Never edit old JSONL lines; append updates with same `id` and higher `rev`.
2. Rebuild indexes from node logs whenever drift is detected.
3. Keep context pack `commit_id` aligned with latest post-response commit.
4. Resolve conflicts via TG tasks tagged `conflict_resolution`.
