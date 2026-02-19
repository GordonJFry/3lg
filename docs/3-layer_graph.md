# Universal 3-Layer Graph Operating Manual for LLM

*(Knowledge Graph + Reasoning Graph + Task Graph)*

> Document role: **Normative SSOT** for 3LG operational behavior.
> Companion (non-normative): `docs/3lg_scientific_paper_draft.md`.

## 0) Purpose

You are an LLM that must operate on long-running projects without relying on a growing linear chat log. You will maintain three external, structured layers:

1. **Knowledge Graph (KG)** — “What is true / known (or assumed)”
2. **Reasoning Graph (RG)** — “Why we decided / how we reasoned”
3. **Task Graph (TG)** — “What needs to be done / execution state”

These layers solve:

* context overflow and cost from rereading everything,
* loss of consistency across time,
* inability to retrieve relevant prior decisions,
* confusion between facts, hypotheses, and plans.

**Core principle:** Never treat the chat transcript as the source of truth. The graphs are the source of truth.

---

## 1) Definitions and responsibilities

### 1.1 Knowledge Graph (KG)

**Stores:** facts, definitions, constraints, specs, data sources, entities, relationships.
**Does not store:** long deliberations, step-by-step thinking, or transient TODOs.

KG answers: *“What do we know?”*

Key properties:

* Every claim should have a **source** and **confidence**.
* Knowledge is **structured**: entities + attributes + relations.
* Knowledge is **stable** unless explicitly updated.

### 1.2 Reasoning Graph (RG)

**Stores:** decisions, alternatives considered, rationale, tradeoffs, risks, assumptions, validation steps, and decision outcomes.

RG answers: *“Why did we choose this?”*

Key properties:

* Keep it compact: **decision records**, not long free-form reasoning.
* Must link to KG nodes used in justification.
* Must reference TG tasks that created/validated the decision.

### 1.3 Task Graph (TG)

**Stores:** tasks, dependencies, status, owners (roles), acceptance criteria, checkpoints, logs of completion, and links to artifacts.

TG answers: *“What are we doing next, what’s done, and what blocks what?”*

Key properties:

* Tasks have clear **Definition of Done** (DoD).
* Tasks can be decomposed and have dependencies.
* Tasks link to the decisions (RG) and knowledge (KG) they use/update.

---

## 2) Required data formats

All three layers must be stored in machine-friendly formats. Use **JSON Lines (JSONL)** or **YAML** per node. Prefer JSONL for append-only logs.

### 2.1 Global IDs and linking

Every node must have:

* `id`: unique stable identifier (UUID or readable ID)
* `type`: node type (e.g., `fact`, `entity`, `decision`, `task`)
* `title`: short human-readable label
* `created_at`, `updated_at`: ISO timestamp
* `links`: list of references to other nodes (by `id`)
* `tags`: list of strings

IDs are the glue between layers. **Never link by name only**.

---

## 3) Knowledge Graph schema (KG)

### 3.1 Node types (minimum set)

* `entity` — object in the domain (system, module, person, feature, dataset)
* `fact` — verifiable statement
* `definition` — glossary entry / canonical meaning
* `constraint` — rule that must be satisfied
* `source` — document, URL, dataset, message excerpt
* `assumption` — unverified claim used temporarily
* `risk` — potential issue with probability/impact

### 3.2 KG node template (JSON)

```json
{
  "id": "KG-ENTITY-...",
  "type": "entity",
  "title": "Entity name",
  "summary": "1-3 sentences",
  "attributes": {
    "key": "value"
  },
  "relationships": [
    {"rel": "depends_on", "target_id": "KG-ENTITY-..."},
    {"rel": "owned_by", "target_id": "KG-ENTITY-..."}
  ],
  "evidence": [
    {"source_id": "KG-SOURCE-...", "quote": "optional short excerpt", "confidence": 0.8}
  ],
  "status": "active",
  "tags": ["..."],
  "created_at": "2026-02-17T00:00:00Z",
  "updated_at": "2026-02-17T00:00:00Z"
}
```

### 3.3 Knowledge quality rules

* Facts must include `evidence` or be marked as `assumption` with low confidence.
* When contradictions appear, create a `conflict` note (or mark both claims with a conflict tag) and create a TG task to resolve.
* Keep summaries short. Store detail in `source` nodes.

---

## 4) Reasoning Graph schema (RG)

### 4.1 RG is a “Decision Record system”

Main unit: `decision`. Each decision should be small and atomic.

### 4.2 Decision node template (JSON)

```json
{
  "id": "RG-DEC-...",
  "type": "decision",
  "title": "Decision title",
  "context": "What problem we were solving (short).",
  "decision": "What we chose (one sentence).",
  "rationale": [
    "Reason 1 (linkable to facts/constraints).",
    "Reason 2"
  ],
  "alternatives": [
    {"option": "Alternative A", "pros": ["..."], "cons": ["..."]},
    {"option": "Alternative B", "pros": ["..."], "cons": ["..."]}
  ],
  "assumptions": ["KG-ASSUMP-..."],
  "risks": ["KG-RISK-..."],
  "validation": [
    {"method": "test/analysis/review", "task_id": "TG-TASK-...", "expected": "What would confirm this"}
  ],
  "outcome": {
    "status": "proposed|accepted|rejected|superseded",
    "supersedes": ["RG-DEC-..."],
    "notes": "Short update after validation."
  },
  "links": {
    "kg_refs": ["KG-..."],
    "tg_refs": ["TG-..."],
    "artifacts": ["ART-..."]
  },
  "tags": ["architecture", "tradeoff"],
  "created_at": "2026-02-17T00:00:00Z",
  "updated_at": "2026-02-17T00:00:00Z"
}
```

### 4.3 RG maintenance rules

* RG is not a diary. Capture **decision + why + how validated**.
* If new evidence invalidates a decision: mark as `superseded` and create a new decision node referencing the old one.
* Always link back to KG facts/constraints that the decision depends on.

---

## 5) Task Graph schema (TG)

### 5.1 Task node template (JSON)

```json
{
  "id": "TG-TASK-...",
  "type": "task",
  "title": "Task title",
  "description": "What to do (short but explicit).",
  "status": "backlog|ready|in_progress|blocked|review|done|canceled",
  "priority": "low|medium|high|urgent",
  "owner_role": "lead|research|build|qa|ops|user",
  "dependencies": ["TG-TASK-..."],
  "blocks": ["TG-TASK-..."],
  "acceptance_criteria": [
    "Concrete check 1",
    "Concrete check 2"
  ],
  "inputs": {
    "kg_refs": ["KG-..."],
    "rg_refs": ["RG-..."],
    "sources": ["KG-SOURCE-..."]
  },
  "outputs": {
    "artifacts": ["ART-..."],
    "kg_updates": ["KG-..."],
    "rg_updates": ["RG-..."]
  },
  "log": [
    {"at": "2026-02-17T00:00:00Z", "event": "created", "by": "system"},
    {"at": "2026-02-17T00:00:00Z", "event": "status_changed", "from": "backlog", "to": "in_progress"}
  ],
  "created_at": "2026-02-17T00:00:00Z",
  "updated_at": "2026-02-17T00:00:00Z"
}
```

### 5.2 TG rules

* No task without acceptance criteria.
* No task marked `done` without:

  * acceptance criteria satisfied (explicitly confirmed),
  * outputs captured (artifact links, KG/RG updates),
  * short completion log entry.

---

## 6) Context retrieval workflow (how to “read” the layers)

You must not load entire graphs each time. Instead build a **Context Pack** per user request.

### 6.1 Context Pack contents (default)

1. **Active goal:** 1 sentence
2. **Relevant tasks:** current task + its dependencies + blockers (≤ 15 nodes)
3. **Relevant knowledge:** the minimal KG subgraph needed (≤ 30 nodes)
4. **Relevant decisions:** last accepted decisions + any directly related decisions (≤ 10 nodes)
5. **Recent deltas:** changes since last interaction (≤ 20 lines)

### 6.2 Retrieval strategy (universal)

When you receive a new user request:

1. Identify **intent**: plan / execute / explain / debug / research.
2. Identify **entities** mentioned or implied.
3. Pull:

   * KG nodes for those entities + constraints/definitions
   * RG decisions that reference those KG nodes
   * TG tasks in progress that reference those KG/RG nodes
4. If missing critical info: create a TG task “Clarify X” and ask the user only what is necessary.

### 6.3 Relevance ranking heuristics

Rank nodes higher if:

* referenced by current TG tasks
* referenced by accepted RG decisions
* updated recently
* tagged as `constraint` or `risk`
* directly match entities in the request

---

## 7) Update workflow (how to keep layers consistent)

### 7.1 After every meaningful response

You must perform a **Post-Response Commit**:

1. **TG update**:

   * create tasks for new work
   * move status for tasks progressed
2. **RG update**:

   * if you made/changed a decision, write a decision record
3. **KG update**:

   * add new facts/constraints/definitions from user inputs or verified sources
   * add new assumptions explicitly if uncertain

If nothing changed, write a single minimal “no-op” commit note (optional).

### 7.2 Atomicity rule

A single user request may cause multiple updates, but:

* avoid mixing unrelated topics in one node,
* prefer several small nodes over one giant node.

### 7.3 Conflict handling

If the user request conflicts with KG or prior decisions:

* do not silently overwrite.
* create:

  * a `conflict` tag on the affected KG nodes,
  * a TG task “Resolve conflict: …”
  * an RG decision if a resolution is chosen.

---

## 8) Efficiency rules (to prevent bloat)

### 8.1 Summarize, don’t hoard

* KG: store *facts + sources*, not long text.
* RG: store *decision records*, not chains of reasoning.
* TG: store *actionable tasks*, not discussions.

### 8.2 Use tiers

Maintain three levels of detail:

* **Index** (titles + IDs + 1 line each)
* **Summary** (what you usually retrieve)
* **Source/Artifact** (deep detail, rarely retrieved)

### 8.3 Decay / archival policy

Periodically (e.g., every 50 tasks or weekly):

* archive TG tasks marked `done` older than N days (keep index)
* compress RG decisions into “decision summaries” if too many
* mark KG nodes as `deprecated` if superseded

Never delete without an archive reference.

---

## 9) Safety and correctness

### 9.1 Separation of fact vs hypothesis

* Facts: must cite `source_id` or be “user-provided”.
* Hypotheses: must be `assumption` with confidence < 0.6 until validated.

### 9.2 Validation-first actions

Any action that changes external systems must be linked to:

* a TG task with acceptance criteria,
* an RG decision (if it’s a choice),
* and relevant KG constraints.

---

## 10) Minimal “skills” you must implement (as the model)

You must implement these internal skills (workflows). The content below tells you what to build; you will implement your own prompts/scripts around them.

### Skill A — Initialize Graphs

Create empty KG/RG/TG indexes, define node schemas, set naming/ID policy.

### Skill B — Build Context Pack

Given a user request, assemble minimal context pack using retrieval rules.

### Skill C — Plan into Tasks

Convert request into TG tasks with dependencies and DoD.

### Skill D — Commit Knowledge

Extract facts/definitions/constraints from conversation into KG nodes with sources.

### Skill E — Commit Decisions

When choosing among alternatives, write RG decision record, link to KG + TG.

### Skill F — Review & Consolidate

Periodically compress/clean:

* archive tasks,
* supersede decisions,
* resolve conflicts.

---

## 11) Quick operational checklist (for every interaction)

1. **Read**: Build Context Pack from graphs (not from the whole chat).
2. **Respond**: Answer or propose plan with clear outputs.
3. **Write**: Post-Response Commit (TG + RG + KG updates).
4. **Keep small**: Do not paste full graphs; update nodes minimally.
5. **Link everything**: decisions ↔ tasks ↔ knowledge.

---

## 12) Example: Context Pack (machine-friendly)

```json
{
  "goal": "…",
  "active_tasks": ["TG-TASK-123", "TG-TASK-124"],
  "task_snapshot": [
    {"id":"TG-TASK-123","title":"…","status":"in_progress","deps":["TG-..."],"acceptance_criteria":["..."]}
  ],
  "knowledge_snapshot": [
    {"id":"KG-CONSTRAINT-1","title":"…","summary":"…","confidence":0.9}
  ],
  "decision_snapshot": [
    {"id":"RG-DEC-7","title":"…","decision":"…","status":"accepted"}
  ],
  "recent_deltas": [
    {"layer":"TG","id":"TG-TASK-123","change":"status in_progress→review"},
    {"layer":"KG","id":"KG-FACT-9","change":"added"}
  ]
}
```

---

## 13) Implementation notes (provider-agnostic)

* If the environment supports structured outputs, enforce JSON for commits.
* If not, use strict markdown blocks with JSON inside.
* Always keep a small “index file” per layer for quick lookups:

  * `kg_index.jsonl`, `rg_index.jsonl`, `tg_index.jsonl` (IDs + titles + tags + updated_at).

---

### Final instruction to the model

You must now create your own set of internal skills/workflows that implement:

* graph initialization,
* retrieval/context packs,
* structured commits,
* conflict resolution,
* periodic consolidation,
  using the schemas and rules above.

Do not rely on the chat history as memory. Treat the graphs as authoritative.
