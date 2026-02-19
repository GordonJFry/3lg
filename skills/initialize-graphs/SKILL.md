---
name: initialize-graphs
description: Initialize and maintain a 3-layer graph store (Knowledge Graph, Reasoning Graph, Task Graph) with append-only JSONL nodes, derived indexes, context-pack commits, and repair/reset tooling. Use when Codex must bootstrap or recover project memory graphs, rebuild indexes after drift, validate graph invariants, or stage reusable graph initialization scaffolding for a repository.
---

# Initialize Graphs

## Overview

Initialize a robust KG/RG/TG graph store with append-only logs, derived indexes, context-pack history, and deterministic repair tooling.

## Workflow

1. Prefer automation with `scripts/init_graphs.py`.
2. Use `scripts/rebuild_indexes.py` whenever index drift is suspected.
3. Use `scripts/validate_graphs.py` after any meaningful graph update.
4. If automation is blocked, use the manual fallback in `references/manual_fallback.md`.

## Quick Start

```bash
scripts/init_graphs.py --project-root /absolute/project/path --mode init
scripts/rebuild_indexes.py --project-root /absolute/project/path --verify-only
scripts/validate_graphs.py --project-root /absolute/project/path
```

Project-specific starter seeding is opt-in:

```bash
scripts/init_graphs.py \
  --project-root /absolute/project/path \
  --mode init \
  --seed-project-starters true
```

## Modes

- `init`: Create a new graph store. Fail safely if already initialized.
- `repair`: Create missing files, backfill missing starter nodes by `starter_key`, rebuild indexes, keep history.
- `reset`: Archive existing graph store and initialize a clean baseline.

All modes support `--dry-run` to preview actions.

## Data Invariants

- Keep node logs append-only; never edit old JSONL lines.
- Append node updates as new lines with same `id` and incremented `rev`.
- Treat index files as derived artifacts from node logs.
- Keep `context_pack.latest.json` commit-aligned with the newest `post_response_commits.jsonl` entry.
- Record conflicts in `conflicts.jsonl`; resolve via TG tasks tagged `conflict_resolution`.

## References

- Hashing, schemas, ID strategy: `references/schemas.md`
- Starter nodes and `starter_key` rules: `references/bootstrap_nodes.md`
- Manual fallback procedure: `references/manual_fallback.md`
