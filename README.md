# 3LG: Skills + Example Graph + Research Draft

Public repository package for the 3LG (Three-Layer Graph) approach.

## Document policy (single source of truth)

- Normative SSOT: `docs/3-layer_graph.md`
- Non-normative research draft: `docs/3lg_scientific_paper_draft.md`
- Navigation and document roles: `docs/README.md`

`README.md` is intentionally concise and should not duplicate normative rules from SSOT.

## Repository structure

- `skills/`
  - `initialize-graphs/`
  - `build-context-pack/`
  - `plan-into-tasks/`
  - `commit-knowledge/`
  - `commit-decisions/`
  - `review-and-consolidate/`
- `example_graph/` (filled sample graph store: KG/RG/TG, indexes, packs, commit logs)
- `docs/` (specification + research materials)

## Validation

Each skill can be validated with:

```bash
python3 "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" <skill_dir>
```

Example:

```bash
python3 "$CODEX_HOME/skills/.system/skill-creator/scripts/quick_validate.py" ./skills/initialize-graphs
```

## Quick start

1. Read `docs/README.md`.
2. Use `docs/3-layer_graph.md` as the operational spec.
3. Use `skills/initialize-graphs` to bootstrap/repair graph storage.
4. Use runtime skills (`build-context-pack`, `plan-into-tasks`, `commit-knowledge`, `commit-decisions`, `review-and-consolidate`) for day-to-day workflow.
