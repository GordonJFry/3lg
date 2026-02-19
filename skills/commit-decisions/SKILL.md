---
name: commit-decisions
description: Convert alternative-analysis input into deterministic RG decision records linked to KG and TG with suggest-first review, revision-safe upsert by decision_key, stable option IDs, and post-response context/commit synchronization. Use when Codex must choose among options and persist reasoning outcomes in rg_nodes.jsonl.
---

# Commit Decisions

## Overview

Extract alternatives, selected option, and rationale from decision text and optionally apply a deterministic RG decision update.

## Workflow

1. Run suggestion mode (`--apply false`) to inspect candidate decision, links, and warnings.
2. Verify alternatives and selected option resolution (`option_id` based).
3. Re-run with `--apply true` to write RG revision safely.
4. Optionally use `--apply-followup-task true` for undecided decisions.

## Quick Start

```bash
scripts/commit_decisions.py \
  --project-root /absolute/project/path \
  --decision-text "Option A vs Option B. Recommended: Option A"
```

Apply mode with strict linking:

```bash
scripts/commit_decisions.py \
  --project-root /absolute/project/path \
  --decision-file /absolute/path/decision.txt \
  --kg-refs KG-ENTITY-... \
  --tg-refs TG-TASK-... \
  --apply true
```

## Defaults

- Suggest-only by default.
- Apply-mode strict linking can require both KG and TG refs.
- Decision history is append-only with rev-aware upsert by `decision_key`.

## References

- Rules and determinism: `references/decision_rules.md`
- RG output/apply contract: `references/rg_contract.md`
- Manual fallback steps: `references/manual_fallback.md`
