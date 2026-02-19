# Manual Fallback

Use this only if `scripts/plan_into_tasks.py` cannot run.

## 1) Normalize request and compute hash

- trim request text
- convert CRLF/CR to LF
- collapse repeated spaces/tabs per line
- preserve case
- compute sha256 hash from normalized text

## 2) Draft 3-10 execution-ready tasks

Each task must include:

- phase (`setup|implement|verify|docs|deploy`)
- explicit dependencies
- 2-4 measurable acceptance criteria

## 3) Validate quality

- no vague acceptance criteria
- no cyclic dependencies
- no self-dependencies

## 4) Suggestion mode output

Write `task_plan.<timestamp>.<plan_id>.json` with:

- `plan_id`, `plan_seq`, `generated_at`, `intent`
- `tasks`, `warnings`, `link_suggestions`
- optional `request_id`, `source_request_hash`

## 5) Apply mode (if explicitly requested)

- map `TEMP-*` dependencies to real TG IDs
- append TG nodes with origin metadata
- rebuild and verify indexes
- append post-response commit row
