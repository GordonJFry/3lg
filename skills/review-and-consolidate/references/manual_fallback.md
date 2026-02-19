# Manual Fallback

Use this only when `scripts/review_consolidate.py` cannot run.

## 1) Build latest views

- load `kg_nodes.jsonl`, `rg_nodes.jsonl`, `tg_nodes.jsonl`, `conflicts.jsonl`
- resolve latest per `id` by `rev`, then `updated_at`, then line order

## 2) Compute candidate actions

- archive eligible TG tasks using terminal+age plus replay-safe exclusions
- group RG decisions by exact `decision_key`, choose canonical, identify supersede updates
- detect resolvable conflicts from linked done/completed `conflict_resolution` tasks and evidence

## 3) Apply updates append-only

- append TG archive rows and optional conflict-resolution seed tasks
- append RG supersede rows
- append conflict rows for link/update/resolve operations

## 4) Rebuild and verify indexes

- regenerate KG/RG/TG indexes from authoritative node logs
- ensure derived indexes match expected content hashes

## 5) Sync context and commit

- write context history file
- validate written JSON
- replace `context_pack.latest.json`
- append `post_response_commits.jsonl` last

## 6) Write consolidation run artifact

- only for apply runs with real changes
- rotate history to configured limit
