# Manual Fallback

Use this when `scripts/commit_decisions.py` cannot run.

## 1) Parse and normalize

- Read decision text.
- Extract alternatives and keep original order.
- Assign stable `option_id` values `OPT-1..N`.

## 2) Resolve selected option

- Match explicit `option_id` first.
- Else exact normalized title match.
- Else mark `undecided`.

## 3) Build decision candidate

Include:
- title, context, decision statement
- alternatives
- selected option
- rationale/assumptions/risks/validation
- refs to KG and TG
- outcome status
- `decision_key`
- `decision_payload_hash`

## 4) Upsert to RG

- find existing by `decision_key`
- same semantic hash => skip
- changed => append same RG id with `rev+1`
- missing => append new RG decision row rev=1

## 5) Optional follow-up task

If undecided and explicitly requested, append TG task tagged `decision_followup`.

## 6) Rebuild indexes and sync

- rebuild and verify indexes
- if changes exist:
  - write + validate context history
  - replace context latest
  - append commit row last
