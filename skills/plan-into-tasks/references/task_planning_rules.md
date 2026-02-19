# Task Planning Rules

## Intent and decomposition

- Resolve intent from explicit `--intent` or classifier (`auto`).
- Auto classifier uses multilingual keyword sets (EN + RU markers).
- Produce execution-ready plans by default (3-10 tasks).
- Each task must include phase, dependencies, and measurable DoD.

## Phase model

Allowed phases:

- `setup`
- `implement`
- `verify`
- `docs`
- `deploy`

Deterministic ordering:

1. phase order
2. priority order
3. title
4. temp_id

## Request normalization and hashing

Normalize request text before hashing:

1. trim leading/trailing whitespace
2. convert CRLF (`\r\n`) and CR (`\r`) to LF (`\n`)
3. collapse repeated spaces/tabs within each line to one space
4. preserve original letter case

Hash with `sha256` over normalized UTF-8 text as `source_request_hash`.

## DoD quality rules

- Each task requires 2-4 acceptance criteria.
- Reject vague criteria unless measurable qualifiers exist.
- Vague tokens include: `ensure`, `properly`, `etc`, `as needed`, `should work`, `handle edge cases`.

## Dependency rules

- Suggestion mode uses `TEMP-*` dependencies.
- Dependency graph must be acyclic.
- No self-dependencies.

## Apply sync contract

For `--apply true` with actual TG mutations:

1. append TG rows
2. rebuild + verify indexes
3. write `context_packs/context_pack.<timestamp>.<commit_id>.json`
4. validate written context history
5. atomically replace `context_pack.latest.json`
6. append `post_response_commits.jsonl` row last

## Sequence allocation

- `plan_seq = max(existing plan_seq) + 1` from `graphs/task_plans`.
- Allocate after reading history and before write.
- If conflict occurs, retry once.
- If still conflicting, fail explicitly.
