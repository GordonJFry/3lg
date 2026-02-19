# Schema and Hashing Rules

## Store Contract

- `kg_nodes.jsonl`, `rg_nodes.jsonl`, `tg_nodes.jsonl` are authoritative append-only logs.
- `kg_index.jsonl`, `rg_index.jsonl`, `tg_index.jsonl` are derived and rebuildable.
- Node IDs use prefix + ULID:
  - `KG-<TYPE>-<ULID>`
  - `RG-DEC-<ULID>`
  - `TG-TASK-<ULID>`

## Required Core Fields

Every KG/RG/TG node requires:

- `id`, `type`, `title`
- `created_at`, `updated_at` (ISO UTC `Z`)
- `links`, `tags`
- `rev` (integer >= 1)

Schemas are versioned and permissive in v1 (`additionalProperties: true`).

## Revision Model

- Never edit old JSONL lines.
- Update by appending a new line with same `id` and higher `rev`.
- Rebuild logic selects latest record by highest `rev`, then `updated_at` tie-break.

## Canonical Hashing

Use canonical JSON for hashing:

1. UTF-8 encoding.
2. Object keys sorted recursively.
3. No insignificant whitespace.

Two hashes are maintained in index rows:

- `full_hash`: hash of full node payload.
- `content_hash`: stable hash excluding volatile fields:
  - remove `updated_at` for all layers
  - remove TG `log`

## Index Row Fields

- `id`, `title`, `tags`, `updated_at`
- `type`, `layer`, `status`
- `content_hash`, `full_hash`

## Link Model

Supported references:

- Internal IDs: `KG-...`, `RG-...`, `TG-...`
- Artifact URIs: `artifact://<path>`
- Optional context pack URIs: `context_pack://<filename>`

Validation severity:

- ERROR: missing/malformed internal references
- WARN: missing optional artifact files (strict mode)
- INFO: rotated context pack history references
