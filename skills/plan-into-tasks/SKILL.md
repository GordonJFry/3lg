---
name: plan-into-tasks
description: Convert a user request into execution-ready Task Graph (TG) tasks with dependencies and measurable Definition of Done (DoD), deterministic ordering, auto-linked KG/RG references, and optional idempotent apply mode. Use when Codex needs to decompose work into actionable tasks (Skill C), validate dependency integrity and DoD quality, and optionally append planned tasks into TG with commit logging.
---

# Plan Into Tasks

## Overview

Generate a deterministic task plan from a user request and optionally apply it to TG with idempotency and index verification.

## Workflow

1. Run `scripts/plan_into_tasks.py` with request text or file.
2. Review suggestion output (`plan_id`, `plan_seq`, tasks, warnings).
3. Apply only when ready using `--apply true`.
4. Optionally refresh request context using `--refresh-request-context true`.

## Quick Start

```bash
scripts/plan_into_tasks.py \
  --project-root /absolute/project/path \
  --request-text "Plan implementation work for request-context freshness checks"
```

Apply mode:

```bash
scripts/plan_into_tasks.py \
  --project-root /absolute/project/path \
  --request-text "Plan implementation work for request-context freshness checks" \
  --apply true
```

## Defaults

- Suggest-only by default (`--apply false`).
- Execution-ready decomposition with deterministic ordering.
- Auto-link KG/RG refs from request entities.

## References

- Planning algorithm, sequencing, DoD quality checks: `references/task_planning_rules.md`
- Output and apply contracts: `references/tg_contract.md`
- Manual fallback workflow: `references/manual_fallback.md`
