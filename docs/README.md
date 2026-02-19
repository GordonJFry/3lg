# Documentation Map

## Source-of-truth policy

- `3-layer_graph.md` is the **normative SSOT** for 3LG operating behavior.
- `3lg_scientific_paper_draft.md` is a **research manuscript draft** and is non-normative.
- `3-layer_graph.md` already contains an extended duplicated technical section with diagrams, so it remains self-contained even if the scientific draft is removed.

If there is any mismatch, follow `3-layer_graph.md`.

## File roles

- `3-layer_graph.md`
  - Operational specification.
  - Layer contracts (KG/RG/TG), invariants, and workflow rules.
- `3lg_scientific_paper_draft.md`
  - Extended scientific framing, formalization, diagrams, and evaluation design.
  - Intended for publication preparation.

## Editing rules for contributors

- Put enforceable behavior changes only into `3-layer_graph.md`.
- Keep `3lg_scientific_paper_draft.md` aligned with SSOT after spec changes.
- Keep root `README.md` short and index-like (no duplicated deep spec text).
