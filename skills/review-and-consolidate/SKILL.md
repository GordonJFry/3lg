---
name: review-and-consolidate
description: Periodically review graph state and consolidate lifecycle drift across TG/RG/conflicts. Use when Codex must archive stale terminal tasks, supersede duplicate or outdated decisions by exact decision_key, and resolve conflicts with task-gated safety while preserving append-only history, replay-safe traceability, and context/commit synchronization.
---

# Review And Consolidate

## Overview

Consolidate graph lifecycle state safely without rewriting history.

## Workflow

1. Run suggestion mode first (`--apply false`).
2. Inspect `candidate_actions` and warnings.
3. Run apply mode (`--apply true`) only after review.
4. Use `--strict-conflicts true` if unresolved conflict prerequisites must block apply.
5. Use `--create-missing-conflict-task true` only when explicit conflict-resolution task seeding is desired.

## Quick Start

Suggestion only:

```bash
scripts/review_consolidate.py \
  --project-root /absolute/project/path
```

Apply with defaults:

```bash
scripts/review_consolidate.py \
  --project-root /absolute/project/path \
  --apply true
```

Apply with stricter conflict gating:

```bash
scripts/review_consolidate.py \
  --project-root /absolute/project/path \
  --apply true \
  --strict-conflicts true
```

## Defaults

- Suggest-first (`--apply false`).
- Single-script interface for archive/supersede/resolve scope.
- Replay-safe archive exclusions include active TG refs, unresolved conflicts, and latest RG `refs.tg_refs` regardless of outcome.
- Context sync is context-first, commit-row-last.

## References

- Consolidation rules and invariants: `references/consolidation_rules.md`
- Output and apply contracts: `references/consolidation_contract.md`
- Manual fallback process: `references/manual_fallback.md`
