# 3LG Skills + Example Graph + Scientific Draft

Repository package for the 3LG (Three-Layer Graph) approach.

## Contents

- `skills/`
  - `initialize-graphs/`
  - `build-context-pack/`
  - `plan-into-tasks/`
  - `commit-knowledge/`
  - `commit-decisions/`
  - `review-and-consolidate/`
- `example_graph/`
  - Filled example graph store (KG/RG/TG + indexes + context packs + commit logs)
- `docs/`
  - `3-layer_graph.md` (operational guide)
  - `3lg_scientific_paper_draft.md` (scientific draft)

## Validation

All six skills in `skills/` were validated with:

```bash
python3 /Users/anatolylobkov/.codex/skills/.system/skill-creator/scripts/quick_validate.py <skill_dir>
```

## Quick start

1. Review the architecture in `docs/3lg_scientific_paper_draft.md`.
2. Inspect data contracts in each skill's `references/`.
3. Use `skills/initialize-graphs` scripts to bootstrap or repair graph storage.
4. Use the remaining skills as runtime graph operations.

## Notes

- Graph data is append-only JSONL with revision (`rev`) semantics.
- `example_graph/` is a working sample dataset for demos and reproducibility.
