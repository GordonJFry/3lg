# Manual Fallback

Use this fallback only if `scripts/build_context_pack.py` cannot run.

## 1) Read latest nodes

Load and resolve latest records from:

- `graphs/kg_nodes.jsonl`
- `graphs/rg_nodes.jsonl`
- `graphs/tg_nodes.jsonl`

Apply resolver rules:

- prefer highest `rev`
- if `rev` missing for any candidate, fallback to newest `updated_at`
- tie-break by latest line order

## 2) Detect intent and entities

- choose intent from `plan|execute|explain|debug|research`
- extract entity matches by IDs and lexical overlap

## 3) Build snapshots with caps

- TG snapshot with dependencies + blockers closure (<=15)
- KG snapshot (<=30)
- RG decision snapshot (<=10)
- recent deltas (<=20)

## 4) Handle missing critical info

- create `clarify_suggestion` only when critical rules match
- do not mutate TG unless explicitly requested

## 5) Persist safely

- write history file under `graphs/request_context_packs/`
- validate JSON
- atomically replace `graphs/request_context.latest.json`

## 6) Request sequence

- allocate `request_seq = max(existing) + 1`
- if duplicate detected, retry once
