# Manual Fallback Procedure

Use this only when scripts cannot run.

## 1) Create folder layout

Create:

- `graphs/`
- `graphs/context_packs/`
- `graphs/schemas/`

## 2) Create required files

- `graphs/meta.json`
- `graphs/kg_nodes.jsonl`
- `graphs/rg_nodes.jsonl`
- `graphs/tg_nodes.jsonl`
- `graphs/kg_index.jsonl`
- `graphs/rg_index.jsonl`
- `graphs/tg_index.jsonl`
- `graphs/conflicts.jsonl`
- `graphs/post_response_commits.jsonl`
- `graphs/context_pack.latest.json`
- `graphs/README.md`
- schema files in `graphs/schemas/`

## 3) Write `meta.json`

Include:

- `store_version`
- `skill_version`
- `initializer_version`
- `project_root`
- `project_name`
- `clock`
- `created_at`
- `id_strategy`
- `id_prefixes`
- `schema_versions`

## 4) Seed starter nodes

Append universal starters to node logs with `rev: 1` and stable `starter_key` values.
Ensure starter RG decision rows include `decision_key` (and mirrored `attributes.decision_key`).

Optionally append project starters if relevant files exist.
For legacy stores: append `rev+1` repair row for any RG decision missing `decision_key`.

## 5) Rebuild indexes from node logs

Generate index rows from latest revision per `id` and write to index files.

## 6) Create first commit-linked context pack

- append first commit record to `post_response_commits.jsonl`
- write matching context pack to `context_packs/`
- copy same payload to `context_pack.latest.json`
- ensure both use same `commit_id`

## 7) Validate

Check:

- required fields exist
- timestamp format is ISO UTC with `Z`
- links are valid
- conflict entries reference TG tasks tagged `conflict_resolution`
- indexes match derived state
- latest context pack `commit_id` matches latest commit
