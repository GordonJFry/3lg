# Decision Rules

## Decision intent

- Use suggest-first by default.
- Apply writes only when `--apply true`.

## Alternatives and option IDs

- Parse alternatives from text by preserving original parse order.
- If `--alternatives-file` is provided, use it as source of alternatives.
- If alternatives are provided via unordered JSON object, materialize and sort by `(normalized_title asc, original_title asc)`.
- Assign stable `option_id` as `OPT-1..N` from final ordered list.
- Keep original title text in stored alternatives.
- Use normalized title only for matching and hashing.
- Support multilingual alternative markers (including RU forms like `Вариант`, `Выбрано`).
- In apply mode, at least one parsed alternative is required (fail-fast when empty).

## Selected option

Resolution order:
1. exact `option_id`
2. exact normalized-title match
3. fail if ambiguous/unresolved when explicit input was provided

If unresolved and no explicit selected option found, store `selected_option="undecided"`.

## Hashing and idempotency

- `decision_key = decision::<scope>::<normalized_title>`.
- `decision_payload_hash` is computed from canonical semantic subset only:
  - context
  - decision
  - selected_option
  - normalized alternatives (`option_id`, normalized title/pros/cons)
  - normalized rationale
  - sorted/deduped refs
  - outcome.status
- Exclude volatile fields: ids, rev, timestamps, commit ids, logs.

Upsert behavior:
- same key + same hash => skip
- same key + changed hash => same RG id with `rev+1`
- missing key => new RG id `rev=1`

## Link strictness boundary

- Suggest mode: allow missing KG/TG refs and emit warnings + link suggestions.
- Apply mode with `--require-kg-tg true`: require both KG and TG refs and all resolvable.

## Lifecycle semantics

- Default outcome status: `proposed`.
- `accepted` requires concrete selected option.
- If selected option changes across revs:
  - `outcome.notes = "supersedes prior choice in rev <N>"`
  - `outcome.supersedes_rev = <N>`

## Sync ordering

For apply with changed rows:
1. write context history file
2. validate history JSON
3. replace `context_pack.latest.json`
4. append post_response_commits row last

No rollback is attempted.
