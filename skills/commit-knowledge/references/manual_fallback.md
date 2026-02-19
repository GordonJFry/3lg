# Manual Fallback

Use this only when `scripts/commit_knowledge.py` cannot run.

## 1) Normalize and hash conversation text

Compute:
- `source_request_hash_raw`: newline-normalized + trimmed, preserve case/punctuation
- `source_request_hash_norm`: lowercase + punctuation-stripped + collapsed spaces

## 2) Extract candidate claims

From conversation lines, extract:
- facts
- definitions
- constraints
- assumptions

Each candidate must include:
- stable `knowledge_key`
- confidence
- evidence with span and excerpt

## 3) Build/reuse source node

Upsert `KG-SOURCE` by `source_key`:
- unchanged digest => no source write
- changed digest => same id with `rev+1`
- missing => new source node

## 4) Apply knowledge rows idempotently

Upsert by `knowledge_key`:
- unchanged => skip
- changed => append same id `rev+1`
- missing => append new node `rev=1`

## 5) Optional conflicts

- default: warnings only
- with explicit flags, append conflicts and optional TG conflict-resolution tasks

## 6) Rebuild indexes and sync context

- rebuild/verify indexes
- if changes exist: update post-response context pack then append commit log row
