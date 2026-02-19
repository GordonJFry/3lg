---
name: commit-knowledge
description: Extract facts, definitions, constraints, and assumptions from conversation text into KG candidates with source provenance, deterministic keys, and idempotent upsert apply mode. Use when Codex needs to commit durable conversation knowledge into graphs/kg_nodes.jsonl with evidence spans, optional conflict recording, and post-response context/commit synchronization.
---

# Commit Knowledge

## Overview

Convert conversation text into structured KG candidates and optionally apply them into the append-only KG store with revisioned upserts.

## Workflow

1. Run `scripts/commit_knowledge.py` in suggestion mode (default `--apply false`).
2. Inspect candidates, conflicts, and hashes.
3. Re-run with `--apply true` only when candidates are correct.
4. Use `--apply-conflicts true` to write conflict records, and `--apply-conflict-task true` only when TG task creation is explicitly intended.

## Quick Start

```bash
scripts/commit_knowledge.py \
  --project-root /absolute/project/path \
  --conversation-text "Facts and constraints from this conversation"
```

Apply mode:

```bash
scripts/commit_knowledge.py \
  --project-root /absolute/project/path \
  --conversation-file /absolute/path/conversation.txt \
  --apply true
```

Apply with conflict records and explicit conflict tasks:

```bash
scripts/commit_knowledge.py \
  --project-root /absolute/project/path \
  --conversation-file /absolute/path/conversation.txt \
  --apply true \
  --apply-conflicts true \
  --apply-conflict-task true
```

## Defaults

- Suggest-only by default (`--apply false`).
- Idempotent upsert by `knowledge_key` and source upsert by `source_key`.
- Dual request hashes are emitted (`source_request_hash_raw`, `source_request_hash_norm`).

## References

- Extraction and key rules: `references/knowledge_extraction_rules.md`
- Output/apply data contracts: `references/kg_contract.md`
- Manual fallback procedure: `references/manual_fallback.md`
