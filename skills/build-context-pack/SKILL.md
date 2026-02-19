---
name: build-context-pack
description: Build deterministic, request-time Context Packs from KG/RG/TG graph stores using intent and entity retrieval rules, scoring, and strict size caps. Use when Codex must answer a new user request from graph memory (not chat history), produce minimal task/knowledge/decision snapshots, suggest Clarify tasks for missing critical information, or optionally append a TG clarify task.
---

# Build Context Pack

## Overview

Assemble a minimal request-time Context Pack from graph layers with deterministic ranking and bounded snapshots.

## Workflow

1. Run `scripts/build_context_pack.py` with `--project-root` and request text/file.
2. Let the script infer intent (`auto`) or force it with `--intent`.
3. Review concise summary output and inspect saved JSON in `graphs/request_context_packs/`.
4. Use `--apply-clarify-task true` only when you explicitly want TG mutation.

## Quick Start

```bash
scripts/build_context_pack.py \
  --project-root /absolute/project/path \
  --request-text "Debug failing context pack ranking for execute intent"
```

## Safety Defaults

- Derived-view only by default: no TG mutation.
- Clarify is suggestion-only unless apply flag is enabled.
- Persist history first, then atomically replace latest pointer.

## References

- Retrieval logic and scoring: `references/retrieval_rules.md`
- Pack schema and field contract: `references/context_pack_contract.md`
- Manual fallback workflow: `references/manual_fallback.md`
