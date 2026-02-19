# Starter Node Strategy

## Universal Starters (default)

Universal starters are seeded when `--seed-universal-starters true` (default):

- KG constraint: graph layers are authoritative memory
- KG definition: 3-layer model roles (KG/RG/TG)
- RG decision: append-only logs + derived indexes
- TG tasks:
  - maintain context packs
  - record post-response commits
  - resolve conflicts through tagged TG tasks

## Project Starters (opt-in)

Project starters are seeded only when `--seed-project-starters true`.

Examples:

- KG source nodes for known project files if they exist
- KG entity nodes for repository and pipeline modules
- KG fact or assumption for observed project shape

If expected files are missing, create an assumption node rather than failing.

## Idempotency Rule

Starter records include `starter_key`.

- `repair` mode checks `starter_key` presence before adding starters.
- Never detect duplicates by title alone.
- Keep `starter_key` stable over time.

Example `starter_key` values:

- `universal.constraint.source_of_truth`
- `universal.task.post_response_commit`
- `project.source.3_layer_graph_manual`
