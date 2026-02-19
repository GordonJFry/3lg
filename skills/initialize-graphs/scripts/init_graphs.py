#!/usr/bin/env python3
"""Initialize and maintain a 3-layer graph store (KG/RG/TG)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

STORE_VERSION = "1.0.0"
SKILL_VERSION = "0.1.0"
INITIALIZER_VERSION = "0.1.0"

LAYER_TO_NODE_FILE = {
    "KG": "kg_nodes.jsonl",
    "RG": "rg_nodes.jsonl",
    "TG": "tg_nodes.jsonl",
}

LAYER_TO_INDEX_FILE = {
    "KG": "kg_index.jsonl",
    "RG": "rg_index.jsonl",
    "TG": "tg_index.jsonl",
}

SCHEMA_FILE_NAMES = {
    "kg": "kg_node.schema.json",
    "rg": "rg_decision.schema.json",
    "tg": "tg_task.schema.json",
    "conflict": "conflict.schema.json",
    "context_pack": "context_pack.schema.json",
}

META_FILE = "meta.json"
CONTEXT_LATEST_FILE = "context_pack.latest.json"
POST_COMMITS_FILE = "post_response_commits.jsonl"
CONFLICTS_FILE = "conflicts.jsonl"
README_FILE = "README.md"

ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
INTERNAL_ID_PATTERN = re.compile(r"^(KG|RG|TG)-[A-Z0-9-]+$")
NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
SPACE_RE = re.compile(r"\s+")

SCHEMA_VERSIONS = {
    "kg": "1.0.0",
    "rg": "1.0.0",
    "tg": "1.0.0",
    "conflict": "1.0.0",
    "context_pack": "1.0.0",
}

GRAPH_README = """# 3-Layer Graph Store

This directory contains the append-only graph memory store:

- `kg_nodes.jsonl`, `rg_nodes.jsonl`, `tg_nodes.jsonl` are authoritative node logs.
- `kg_index.jsonl`, `rg_index.jsonl`, `tg_index.jsonl` are derived artifacts.
- `meta.json` stores store/version configuration.
- `post_response_commits.jsonl` stores commit records.
- `context_pack.latest.json` mirrors the newest commit context.
- `context_packs/` stores historical context pack snapshots.
- `conflicts.jsonl` stores conflict objects only.

## Invariants

1. Never edit old JSONL lines; append updates with same `id` and higher `rev`.
2. Rebuild indexes from node logs whenever drift is detected.
3. Keep context pack `commit_id` aligned with latest post-response commit.
4. Resolve conflicts via TG tasks tagged `conflict_resolution`.
"""


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_bool(raw: str) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw}")


def encode_crockford(value: int, length: int) -> str:
    chars: List[str] = []
    for _ in range(length):
        chars.append(ULID_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def new_ulid() -> str:
    ts_ms = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    value = (ts_ms << 80) | rand
    return encode_crockford(value, 26)


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def normalize_key_text(text: str) -> str:
    value = str(text).strip().lower().replace("_", " ")
    value = NON_WORD_RE.sub(" ", value)
    value = SPACE_RE.sub(" ", value).strip()
    return value


def derive_decision_key(title: str) -> str:
    normalized_title = normalize_key_text(title)
    if not normalized_title:
        normalized_title = f"item_{hashlib.sha256(str(title).encode('utf-8')).hexdigest()[:12]}"
    return f"decision::bootstrap::{normalized_title}"


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path} line {idx}: {exc}") from exc
        records.append(payload)
    return records


def write_text_atomic(path: Path, content: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY-RUN] write {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{new_ulid()}")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: Any, dry_run: bool) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", dry_run)


def write_jsonl_atomic(path: Path, records: Iterable[Dict[str, Any]], dry_run: bool) -> None:
    lines = [json.dumps(record, sort_keys=True, ensure_ascii=False) for record in records]
    content = ("\n".join(lines) + "\n") if lines else ""
    write_text_atomic(path, content, dry_run)


def append_jsonl_atomic(path: Path, records: Iterable[Dict[str, Any]], dry_run: bool) -> None:
    new_records = list(records)
    if not new_records:
        return
    existing = read_jsonl(path)
    existing.extend(new_records)
    write_jsonl_atomic(path, existing, dry_run)


def latest_by_id(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for record in records:
        node_id = record.get("id")
        if not isinstance(node_id, str) or not node_id:
            continue
        candidate_rev = int(record.get("rev", 1))
        candidate_ts = str(record.get("updated_at", ""))
        current = latest.get(node_id)
        if current is None:
            latest[node_id] = record
            continue
        current_rev = int(current.get("rev", 1))
        current_ts = str(current.get("updated_at", ""))
        if candidate_rev > current_rev or (candidate_rev == current_rev and candidate_ts > current_ts):
            latest[node_id] = record
    return latest


@dataclass
class GraphPaths:
    project_root: Path
    graphs_dir: Path

    @property
    def schemas_dir(self) -> Path:
        return self.graphs_dir / "schemas"

    @property
    def context_packs_dir(self) -> Path:
        return self.graphs_dir / "context_packs"

    @property
    def meta_file(self) -> Path:
        return self.graphs_dir / META_FILE

    @property
    def context_latest_file(self) -> Path:
        return self.graphs_dir / CONTEXT_LATEST_FILE

    @property
    def post_commits_file(self) -> Path:
        return self.graphs_dir / POST_COMMITS_FILE

    @property
    def conflicts_file(self) -> Path:
        return self.graphs_dir / CONFLICTS_FILE

    @property
    def readme_file(self) -> Path:
        return self.graphs_dir / README_FILE

    def node_file(self, layer: str) -> Path:
        return self.graphs_dir / LAYER_TO_NODE_FILE[layer]

    def index_file(self, layer: str) -> Path:
        return self.graphs_dir / LAYER_TO_INDEX_FILE[layer]


def id_prefix_for_layer(layer: str, node_type: str) -> str:
    if layer == "KG":
        return f"KG-{str(node_type).upper()}"
    if layer == "RG":
        return "RG-DEC"
    if layer == "TG":
        return "TG-TASK"
    raise ValueError(f"Unknown layer: {layer}")


def make_node_id(layer: str, node_type: str) -> str:
    return f"{id_prefix_for_layer(layer, node_type)}-{new_ulid()}"


def ensure_store_layout(paths: GraphPaths, dry_run: bool) -> None:
    targets = [
        paths.graphs_dir,
        paths.schemas_dir,
        paths.context_packs_dir,
    ]
    for target in targets:
        if dry_run:
            print(f"[DRY-RUN] mkdir -p {target}")
        else:
            target.mkdir(parents=True, exist_ok=True)


def ensure_file_exists(path: Path, dry_run: bool) -> None:
    if path.exists():
        return
    write_text_atomic(path, "", dry_run)


def build_meta(paths: GraphPaths, project_name: str, clock: str) -> Dict[str, Any]:
    return {
        "created_at": now_iso_utc(),
        "clock": clock,
        "id_prefixes": {
            "kg": "KG-<TYPE>-<ULID>",
            "rg": "RG-DEC-<ULID>",
            "tg": "TG-TASK-<ULID>",
        },
        "id_strategy": "prefix+ulid",
        "initializer_version": INITIALIZER_VERSION,
        "project_name": project_name,
        "project_root": str(paths.project_root),
        "schema_versions": SCHEMA_VERSIONS,
        "skill_version": SKILL_VERSION,
        "store_version": STORE_VERSION,
    }


def build_schema_documents() -> Dict[str, Dict[str, Any]]:
    base_props = {
        "id": {"type": "string"},
        "type": {"type": "string"},
        "title": {"type": "string"},
        "created_at": {"type": "string"},
        "updated_at": {"type": "string"},
        "links": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
        "rev": {"type": "integer", "minimum": 1},
        "starter_key": {"type": "string"},
        "human_ref": {"type": "string"},
    }

    kg_props = dict(base_props)
    kg_props.update(
        {
            "summary": {"type": "string"},
            "attributes": {"type": "object"},
            "relationships": {"type": "array", "items": {"type": "object"}},
            "evidence": {"type": "array", "items": {"type": "object"}},
            "status": {"type": "string"},
        }
    )

    rg_props = dict(base_props)
    rg_props.update(
        {
            "context": {"type": "string"},
            "decision": {"type": "string"},
            "rationale": {"type": "array", "items": {"type": "string"}},
            "alternatives": {"type": "array", "items": {"type": "object"}},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "risks": {"type": "array", "items": {"type": "string"}},
            "validation": {"type": "array", "items": {"type": "object"}},
            "outcome": {"type": "object"},
        }
    )

    tg_props = dict(base_props)
    tg_props.update(
        {
            "description": {"type": "string"},
            "status": {"type": "string"},
            "priority": {"type": "string"},
            "owner_role": {"type": "string"},
            "dependencies": {"type": "array", "items": {"type": "string"}},
            "blocks": {"type": "array", "items": {"type": "string"}},
            "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
            "inputs": {"type": "object"},
            "outputs": {"type": "object"},
            "log": {"type": "array", "items": {"type": "object"}},
        }
    )

    conflict_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": [
            "id",
            "title",
            "status",
            "created_at",
            "updated_at",
            "node_ids",
            "evidence",
            "resolution_task_ids",
        ],
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "status": {"type": "string"},
            "created_at": {"type": "string"},
            "updated_at": {"type": "string"},
            "node_ids": {"type": "array", "items": {"type": "string"}},
            "evidence": {"type": "array", "items": {"type": "object"}},
            "resolution_task_ids": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": True,
    }

    context_pack_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": [
            "goal",
            "generated_at",
            "commit_id",
            "active_tasks",
            "task_snapshot",
            "knowledge_snapshot",
            "decision_snapshot",
            "recent_deltas",
        ],
        "properties": {
            "goal": {"type": "string"},
            "generated_at": {"type": "string"},
            "commit_id": {"type": "string"},
            "active_tasks": {"type": "array", "items": {"type": "string"}},
            "task_snapshot": {"type": "array", "items": {"type": "object"}},
            "knowledge_snapshot": {"type": "array", "items": {"type": "object"}},
            "decision_snapshot": {"type": "array", "items": {"type": "object"}},
            "recent_deltas": {"type": "array", "items": {"type": "object"}},
        },
        "additionalProperties": True,
    }

    schemas = {
        "kg": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": [
                "id",
                "type",
                "title",
                "created_at",
                "updated_at",
                "links",
                "tags",
                "rev",
                "summary",
                "attributes",
                "relationships",
                "evidence",
                "status",
            ],
            "properties": kg_props,
            "additionalProperties": True,
        },
        "rg": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": [
                "id",
                "type",
                "title",
                "created_at",
                "updated_at",
                "links",
                "tags",
                "rev",
                "context",
                "decision",
                "rationale",
                "alternatives",
                "assumptions",
                "risks",
                "validation",
                "outcome",
            ],
            "properties": rg_props,
            "additionalProperties": True,
        },
        "tg": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": [
                "id",
                "type",
                "title",
                "created_at",
                "updated_at",
                "links",
                "tags",
                "rev",
                "description",
                "status",
                "priority",
                "owner_role",
                "dependencies",
                "blocks",
                "acceptance_criteria",
                "inputs",
                "outputs",
                "log",
            ],
            "properties": tg_props,
            "additionalProperties": True,
        },
        "conflict": conflict_schema,
        "context_pack": context_pack_schema,
    }
    return schemas


def ensure_schema_files(paths: GraphPaths, dry_run: bool, overwrite: bool) -> None:
    schemas = build_schema_documents()
    for key, schema_doc in schemas.items():
        schema_file = paths.schemas_dir / SCHEMA_FILE_NAMES[key]
        if schema_file.exists() and not overwrite:
            continue
        write_json_atomic(schema_file, schema_doc, dry_run)


def index_row_for_node(layer: str, node: Dict[str, Any]) -> Dict[str, Any]:
    content_view = copy.deepcopy(node)
    content_view.pop("updated_at", None)
    if layer == "TG":
        content_view.pop("log", None)

    row = {
        "content_hash": sha256_hex(content_view),
        "full_hash": sha256_hex(node),
        "id": node.get("id"),
        "layer": layer,
        "status": node.get("status") if layer == "TG" else None,
        "tags": node.get("tags", []),
        "title": node.get("title", ""),
        "type": node.get("type", ""),
        "updated_at": node.get("updated_at", ""),
    }
    return row


def rebuild_indexes(paths: GraphPaths, verify_only: bool, dry_run: bool) -> Tuple[bool, Dict[str, int]]:
    drift_found = False
    counts: Dict[str, int] = {}

    for layer in ("KG", "RG", "TG"):
        node_file = paths.node_file(layer)
        index_file = paths.index_file(layer)

        all_records = read_jsonl(node_file)
        latest_nodes = latest_by_id(all_records)
        rows = [index_row_for_node(layer, node) for _, node in sorted(latest_nodes.items())]
        counts[layer] = len(rows)

        if verify_only:
            current_rows = read_jsonl(index_file)
            expected = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows]
            actual = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in current_rows]
            if expected != actual:
                drift_found = True
                print(f"[DRIFT] {layer} index differs from derived view")
        else:
            write_jsonl_atomic(index_file, rows, dry_run)

    return drift_found, counts


def make_starter_kg_node(
    node_type: str,
    title: str,
    summary: str,
    tags: List[str],
    starter_key: str,
    attributes: Optional[Dict[str, Any]] = None,
    relationships: Optional[List[Dict[str, Any]]] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    links: Optional[List[str]] = None,
    status: str = "active",
) -> Dict[str, Any]:
    now = now_iso_utc()
    return {
        "id": make_node_id("KG", node_type),
        "type": node_type,
        "title": title,
        "summary": summary,
        "attributes": attributes or {},
        "relationships": relationships or [],
        "evidence": evidence or [],
        "status": status,
        "tags": tags,
        "links": links or [],
        "starter_key": starter_key,
        "rev": 1,
        "created_at": now,
        "updated_at": now,
    }


def make_starter_rg_decision(
    title: str,
    context: str,
    decision: str,
    rationale: List[str],
    kg_refs: List[str],
    tg_refs: List[str],
    starter_key: str,
) -> Dict[str, Any]:
    now = now_iso_utc()
    decision_key = derive_decision_key(title)
    return {
        "id": make_node_id("RG", "decision"),
        "type": "decision",
        "title": title,
        "context": context,
        "decision": decision,
        "rationale": rationale,
        "alternatives": [
            {"option": "YAML-only graph files", "pros": ["Human-readable"], "cons": ["Weak append-only lineage"]},
            {
                "option": "Unstructured markdown notes",
                "pros": ["Easy to write"],
                "cons": ["No stable linking across KG/RG/TG"],
            },
        ],
        "assumptions": [],
        "risks": [],
        "validation": [],
        "outcome": {
            "status": "accepted",
            "supersedes": [],
            "notes": "Bootstrap decision for graph store initialization.",
        },
        "links": kg_refs + tg_refs,
        "refs": {
            "kg_refs": kg_refs,
            "tg_refs": tg_refs,
            "artifacts": [],
        },
        "decision_key": decision_key,
        "attributes": {
            "decision_key": decision_key,
        },
        "tags": ["bootstrap", "architecture"],
        "starter_key": starter_key,
        "rev": 1,
        "created_at": now,
        "updated_at": now,
    }


def make_starter_tg_task(
    title: str,
    description: str,
    acceptance_criteria: List[str],
    starter_key: str,
    kg_refs: Optional[List[str]] = None,
    rg_refs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    now = now_iso_utc()
    return {
        "id": make_node_id("TG", "task"),
        "type": "task",
        "title": title,
        "description": description,
        "status": "ready",
        "priority": "high",
        "owner_role": "lead",
        "dependencies": [],
        "blocks": [],
        "acceptance_criteria": acceptance_criteria,
        "inputs": {
            "kg_refs": kg_refs or [],
            "rg_refs": rg_refs or [],
            "sources": [],
        },
        "outputs": {
            "artifacts": [],
            "kg_updates": [],
            "rg_updates": [],
        },
        "log": [
            {
                "at": now,
                "event": "created",
                "by": "initializer",
            }
        ],
        "links": [],
        "tags": ["bootstrap"],
        "starter_key": starter_key,
        "rev": 1,
        "created_at": now,
        "updated_at": now,
    }


def build_universal_starters() -> Dict[str, List[Dict[str, Any]]]:
    kg_constraint = make_starter_kg_node(
        node_type="constraint",
        title="Graphs are authoritative memory",
        summary="Treat KG/RG/TG as the source of truth; do not rely on chat transcript for long-term memory.",
        tags=["constraint", "memory", "universal"],
        starter_key="universal.constraint.source_of_truth",
    )

    kg_definition = make_starter_kg_node(
        node_type="definition",
        title="Three-layer graph model",
        summary="Knowledge Graph stores facts, Reasoning Graph stores decisions, Task Graph stores execution state.",
        tags=["definition", "universal"],
        starter_key="universal.definition.layers",
    )

    tg_context_task = make_starter_tg_task(
        title="Maintain Context Pack for each meaningful interaction",
        description="Generate or refresh context packs after meaningful updates and link them to commit records.",
        acceptance_criteria=[
            "context_pack.latest.json points to newest commit_id",
            "A timestamped context pack history file is written",
        ],
        starter_key="universal.task.context_pack",
        kg_refs=[kg_constraint["id"], kg_definition["id"]],
    )

    tg_commit_task = make_starter_tg_task(
        title="Record post-response commits",
        description="Append structured post-response commit events including changed nodes and context pack reference.",
        acceptance_criteria=[
            "post_response_commits.jsonl has commit_id per meaningful response",
            "Each commit entry includes changed node ids",
        ],
        starter_key="universal.task.post_response_commit",
        kg_refs=[kg_constraint["id"]],
    )

    tg_conflict_task = make_starter_tg_task(
        title="Resolve graph conflicts via tagged tasks",
        description="Any conflict record must be linked to TG tasks tagged conflict_resolution.",
        acceptance_criteria=[
            "Every conflict record has at least one TG resolution task",
            "Resolution tasks include tag conflict_resolution",
        ],
        starter_key="universal.task.conflict_resolution",
        kg_refs=[kg_constraint["id"]],
    )
    tg_conflict_task["tags"].append("conflict_resolution")

    rg_decision = make_starter_rg_decision(
        title="Use append-only JSONL with derived indexes",
        context="Initialize durable project memory with auditable updates.",
        decision="Use append-only JSONL node logs as authority and rebuildable indexes as derived artifacts.",
        rationale=[
            "Append-only logs preserve change lineage.",
            "Derived indexes can be repaired when drift occurs.",
            "ULID identifiers support sortable unique IDs across layers.",
        ],
        kg_refs=[kg_constraint["id"], kg_definition["id"]],
        tg_refs=[tg_context_task["id"], tg_commit_task["id"], tg_conflict_task["id"]],
        starter_key="universal.decision.append_only_and_derived_indexes",
    )

    tg_context_task["inputs"]["rg_refs"].append(rg_decision["id"])
    tg_commit_task["inputs"]["rg_refs"].append(rg_decision["id"])
    tg_conflict_task["inputs"]["rg_refs"].append(rg_decision["id"])

    return {
        "KG": [kg_constraint, kg_definition],
        "RG": [rg_decision],
        "TG": [tg_context_task, tg_commit_task, tg_conflict_task],
    }


def build_project_starters(project_root: Path, project_name: str) -> Dict[str, List[Dict[str, Any]]]:
    records: Dict[str, List[Dict[str, Any]]] = {"KG": [], "RG": [], "TG": []}

    three_layer_doc = project_root / "3-layer_graph.md"
    strategy_file = project_root / "fvg3_long_15m.py"

    source_nodes: List[Dict[str, Any]] = []

    if three_layer_doc.exists():
        source_nodes.append(
            make_starter_kg_node(
                node_type="source",
                title="3-layer graph operating manual",
                summary="Project-local operating rules for KG/RG/TG workflows and schemas.",
                tags=["source", "project"],
                starter_key="project.source.3_layer_graph_manual",
                attributes={
                    "kind": "file",
                    "path": str(three_layer_doc),
                },
            )
        )

    if strategy_file.exists():
        source_nodes.append(
            make_starter_kg_node(
                node_type="source",
                title="Primary strategy script",
                summary="Main trading strategy pipeline implementation file in the repository.",
                tags=["source", "project"],
                starter_key="project.source.primary_strategy_script",
                attributes={
                    "kind": "file",
                    "path": str(strategy_file),
                },
            )
        )

    records["KG"].extend(source_nodes)

    project_entity = make_starter_kg_node(
        node_type="entity",
        title=f"Project {project_name}",
        summary="Repository-level entity representing this graph-enabled project workspace.",
        tags=["entity", "project"],
        starter_key="project.entity.repo",
        attributes={
            "project_name": project_name,
            "project_root": str(project_root),
        },
        links=[node["id"] for node in source_nodes],
    )

    records["KG"].append(project_entity)

    if strategy_file.exists():
        strategy_entity = make_starter_kg_node(
            node_type="entity",
            title="Strategy Pipeline fvg3_long_15m",
            summary="Strategy pipeline entity for data loading, feature generation, simulation, and search orchestration.",
            tags=["entity", "pipeline", "project"],
            starter_key="project.entity.strategy_pipeline",
            attributes={
                "entry_file": str(strategy_file),
                "language": "python",
            },
            links=[project_entity["id"]],
            relationships=[
                {
                    "rel": "part_of",
                    "target_id": project_entity["id"],
                }
            ],
        )
        records["KG"].append(strategy_entity)

    repo_fact_evidence = [{"source_id": node["id"], "confidence": 0.8} for node in source_nodes]
    repo_fact = make_starter_kg_node(
        node_type="fact",
        title="Initial repository shape observed",
        summary="At initialization time, repository includes graph manual and at least one strategy implementation script.",
        tags=["fact", "project"],
        starter_key="project.fact.repo_shape",
        attributes={
            "observed_files": [
                str(three_layer_doc) if three_layer_doc.exists() else None,
                str(strategy_file) if strategy_file.exists() else None,
            ],
        },
        evidence=repo_fact_evidence,
        links=[project_entity["id"]],
    )
    repo_fact["attributes"]["observed_files"] = [v for v in repo_fact["attributes"]["observed_files"] if v]
    records["KG"].append(repo_fact)

    if not source_nodes:
        assumption = make_starter_kg_node(
            node_type="assumption",
            title="Project sources not found during bootstrap",
            summary="Expected project starter source files were missing; verify project-specific starter files later.",
            tags=["assumption", "project"],
            starter_key="project.assumption.missing_sources",
            status="active",
            attributes={"project_root": str(project_root)},
        )
        records["KG"].append(assumption)

    return records


def flatten_records(layer_records: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    result: List[Tuple[str, Dict[str, Any]]] = []
    for layer in ("KG", "RG", "TG"):
        for record in layer_records.get(layer, []):
            result.append((layer, record))
    return result


def collect_existing_starter_keys(paths: GraphPaths) -> set[str]:
    keys: set[str] = set()
    for layer in ("KG", "RG", "TG"):
        for record in read_jsonl(paths.node_file(layer)):
            starter_key = record.get("starter_key")
            if isinstance(starter_key, str) and starter_key:
                keys.add(starter_key)
    return keys


def append_starter_records(paths: GraphPaths, starters: Dict[str, List[Dict[str, Any]]], dry_run: bool) -> List[str]:
    existing_keys = collect_existing_starter_keys(paths)
    changed_ids: List[str] = []
    for layer in ("KG", "RG", "TG"):
        candidates = starters.get(layer, [])
        to_append = []
        for record in candidates:
            starter_key = record.get("starter_key")
            if isinstance(starter_key, str) and starter_key in existing_keys:
                continue
            to_append.append(record)
            if isinstance(starter_key, str) and starter_key:
                existing_keys.add(starter_key)
            changed_ids.append(record["id"])
        append_jsonl_atomic(paths.node_file(layer), to_append, dry_run)
    return changed_ids


def backfill_missing_decision_keys(paths: GraphPaths, dry_run: bool) -> List[str]:
    rg_latest = latest_by_id(read_jsonl(paths.node_file("RG")))
    rows_to_append: List[Dict[str, Any]] = []
    changed_ids: List[str] = []
    now = now_iso_utc()

    for node_id, node in sorted(rg_latest.items()):
        if str(node.get("type", "")) != "decision":
            continue

        decision_key = node.get("decision_key")
        attrs = node.get("attributes") if isinstance(node.get("attributes"), dict) else {}
        nested_key = attrs.get("decision_key") if isinstance(attrs, dict) else None
        has_key = isinstance(decision_key, str) and decision_key.strip()
        has_nested = isinstance(nested_key, str) and nested_key.strip()
        if has_key and has_nested:
            continue

        new_row = copy.deepcopy(node)
        current_rev = int(new_row.get("rev", 1)) if isinstance(new_row.get("rev"), int) else 1
        resolved_key = derive_decision_key(str(new_row.get("title", node_id)))
        attrs_copy = copy.deepcopy(attrs) if isinstance(attrs, dict) else {}
        attrs_copy["decision_key"] = resolved_key

        new_row["decision_key"] = resolved_key
        new_row["attributes"] = attrs_copy
        new_row["rev"] = current_rev + 1
        new_row["updated_at"] = now
        new_row["created_by"] = "initialize-graphs"
        new_row["origin"] = "repair_backfill_decision_key"

        rows_to_append.append(new_row)
        changed_ids.append(node_id)

    append_jsonl_atomic(paths.node_file("RG"), rows_to_append, dry_run)
    return changed_ids


def build_context_pack(
    paths: GraphPaths,
    commit_id: str,
    changed_node_ids: List[str],
    goal_hint: str,
) -> Dict[str, Any]:
    kg_latest = latest_by_id(read_jsonl(paths.node_file("KG")))
    rg_latest = latest_by_id(read_jsonl(paths.node_file("RG")))
    tg_latest = latest_by_id(read_jsonl(paths.node_file("TG")))

    tg_nodes = list(tg_latest.values())
    tg_nodes.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)

    active_tg_nodes = [
        node for node in tg_nodes if str(node.get("status")) in {"in_progress", "ready", "blocked", "review"}
    ]

    if active_tg_nodes:
        goal = str(active_tg_nodes[0].get("title", goal_hint))
    else:
        goal = goal_hint

    task_snapshot = [
        {
            "id": node.get("id"),
            "title": node.get("title"),
            "status": node.get("status"),
            "deps": node.get("dependencies", []),
            "acceptance_criteria": node.get("acceptance_criteria", []),
        }
        for node in active_tg_nodes[:15]
    ]

    kg_nodes = list(kg_latest.values())
    kg_nodes.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)

    knowledge_snapshot = [
        {
            "id": node.get("id"),
            "title": node.get("title"),
            "summary": node.get("summary", ""),
            "confidence": (
                max((float(entry.get("confidence", 0.0)) for entry in node.get("evidence", []) if isinstance(entry, dict)), default=0.0)
                if isinstance(node.get("evidence"), list)
                else 0.0
            ),
        }
        for node in kg_nodes[:30]
    ]

    rg_nodes = [
        node
        for node in rg_latest.values()
        if str(node.get("outcome", {}).get("status", "")) in {"accepted", "proposed", "superseded"}
    ]
    rg_nodes.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)

    decision_snapshot = [
        {
            "id": node.get("id"),
            "title": node.get("title"),
            "decision": node.get("decision", ""),
            "status": node.get("outcome", {}).get("status", ""),
        }
        for node in rg_nodes[:10]
    ]

    recent_deltas = []
    for node_id in changed_node_ids[:20]:
        layer = "UNKNOWN"
        if isinstance(node_id, str) and "-" in node_id:
            layer = node_id.split("-", 1)[0]
        recent_deltas.append({"layer": layer, "id": node_id, "change": "added_or_updated"})

    return {
        "goal": goal,
        "generated_at": now_iso_utc(),
        "commit_id": commit_id,
        "active_tasks": [node.get("id") for node in active_tg_nodes[:15]],
        "task_snapshot": task_snapshot,
        "knowledge_snapshot": knowledge_snapshot,
        "decision_snapshot": decision_snapshot,
        "recent_deltas": recent_deltas,
    }


def write_context_pack(
    paths: GraphPaths,
    context_pack: Dict[str, Any],
    history_limit: int,
    dry_run: bool,
) -> str:
    stamp = timestamp_slug()
    history_name = f"context_pack.{stamp}.{context_pack['commit_id']}.json"
    history_path = paths.context_packs_dir / history_name
    write_json_atomic(history_path, context_pack, dry_run)
    write_json_atomic(paths.context_latest_file, context_pack, dry_run)

    history_files = sorted(paths.context_packs_dir.glob("context_pack.*.json"))
    if len(history_files) > history_limit:
        removable = history_files[: len(history_files) - history_limit]
        for old_path in removable:
            if dry_run:
                print(f"[DRY-RUN] remove {old_path}")
            else:
                old_path.unlink(missing_ok=True)

    return str(history_path)


def append_commit_record(
    paths: GraphPaths,
    commit_id: str,
    mode: str,
    changed_node_ids: List[str],
    context_pack_path: str,
    dry_run: bool,
) -> None:
    record = {
        "commit_id": commit_id,
        "mode": mode,
        "changed_node_ids": changed_node_ids,
        "context_pack_path": context_pack_path,
        "created_at": now_iso_utc(),
        "event": "post_response_commit",
    }
    append_jsonl_atomic(paths.post_commits_file, [record], dry_run)


def ensure_base_files(paths: GraphPaths, dry_run: bool) -> None:
    for layer in ("KG", "RG", "TG"):
        ensure_file_exists(paths.node_file(layer), dry_run)
        ensure_file_exists(paths.index_file(layer), dry_run)
    ensure_file_exists(paths.conflicts_file, dry_run)
    ensure_file_exists(paths.post_commits_file, dry_run)
    ensure_file_exists(paths.readme_file, dry_run)
    if not paths.readme_file.exists() or paths.readme_file.stat().st_size == 0:
        write_text_atomic(paths.readme_file, GRAPH_README, dry_run)


def init_mode(
    paths: GraphPaths,
    project_name: str,
    clock: str,
    seed_universal_starters: bool,
    seed_project_starters: bool,
    history_limit: int,
    dry_run: bool,
    allow_existing_meta: bool = False,
) -> int:
    if paths.meta_file.exists() and not allow_existing_meta:
        raise RuntimeError(f"Graph store is already initialized at {paths.meta_file}. Use --mode repair or --mode reset.")

    ensure_store_layout(paths, dry_run)
    ensure_base_files(paths, dry_run)
    ensure_schema_files(paths, dry_run, overwrite=True)
    write_json_atomic(paths.meta_file, build_meta(paths, project_name, clock), dry_run)

    changed_ids: List[str] = []

    if seed_universal_starters:
        changed_ids.extend(append_starter_records(paths, build_universal_starters(), dry_run))

    if seed_project_starters:
        changed_ids.extend(append_starter_records(paths, build_project_starters(paths.project_root, project_name), dry_run))

    changed_ids.extend(backfill_missing_decision_keys(paths, dry_run))

    rebuild_indexes(paths, verify_only=False, dry_run=dry_run)

    commit_id = f"COMMIT-{new_ulid()}"
    context_pack = build_context_pack(paths, commit_id, changed_ids, goal_hint="Initialize graph-driven project memory")
    context_pack_path = write_context_pack(paths, context_pack, history_limit, dry_run)
    append_commit_record(paths, commit_id, "init", changed_ids, context_pack_path, dry_run)

    print(f"[OK] init complete: {paths.graphs_dir}")
    print(f"[OK] commit_id: {commit_id}")
    return 0


def repair_mode(
    paths: GraphPaths,
    project_name: str,
    clock: str,
    seed_universal_starters: bool,
    seed_project_starters: bool,
    history_limit: int,
    dry_run: bool,
) -> int:
    ensure_store_layout(paths, dry_run)
    ensure_base_files(paths, dry_run)
    ensure_schema_files(paths, dry_run, overwrite=False)

    if not paths.meta_file.exists():
        write_json_atomic(paths.meta_file, build_meta(paths, project_name, clock), dry_run)

    changed_ids: List[str] = []

    if seed_universal_starters:
        changed_ids.extend(append_starter_records(paths, build_universal_starters(), dry_run))

    if seed_project_starters:
        changed_ids.extend(append_starter_records(paths, build_project_starters(paths.project_root, project_name), dry_run))

    changed_ids.extend(backfill_missing_decision_keys(paths, dry_run))

    rebuild_indexes(paths, verify_only=False, dry_run=dry_run)

    commit_id = f"COMMIT-{new_ulid()}"
    context_pack = build_context_pack(paths, commit_id, changed_ids, goal_hint="Repair graph store consistency")
    context_pack_path = write_context_pack(paths, context_pack, history_limit, dry_run)
    append_commit_record(paths, commit_id, "repair", changed_ids, context_pack_path, dry_run)

    print(f"[OK] repair complete: {paths.graphs_dir}")
    print(f"[OK] commit_id: {commit_id}")
    return 0


def reset_mode(
    paths: GraphPaths,
    project_name: str,
    clock: str,
    seed_universal_starters: bool,
    seed_project_starters: bool,
    history_limit: int,
    archive_dir: Path,
    dry_run: bool,
) -> int:
    if paths.graphs_dir.exists() and any(paths.graphs_dir.iterdir()):
        archive_dir = archive_dir.resolve()
        archive_target = archive_dir / f"graphs-{timestamp_slug()}"
        suffix = 1
        while archive_target.exists():
            archive_target = archive_dir / f"graphs-{timestamp_slug()}-{suffix}"
            suffix += 1

        if dry_run:
            print(f"[DRY-RUN] mkdir -p {archive_dir}")
            print(f"[DRY-RUN] move {paths.graphs_dir} -> {archive_target}")
        else:
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(paths.graphs_dir), str(archive_target))
            print(f"[OK] archived previous graph store to {archive_target}")

    return init_mode(
        paths=paths,
        project_name=project_name,
        clock=clock,
        seed_universal_starters=seed_universal_starters,
        seed_project_starters=seed_project_starters,
        history_limit=history_limit,
        dry_run=dry_run,
        allow_existing_meta=dry_run,
    )


def extract_links(node: Dict[str, Any], layer: str) -> List[str]:
    links: List[str] = []

    base_links = node.get("links", [])
    if isinstance(base_links, list):
        links.extend(str(item) for item in base_links if isinstance(item, str))

    if layer == "KG":
        relationships = node.get("relationships", [])
        if isinstance(relationships, list):
            for rel in relationships:
                if isinstance(rel, dict):
                    target_id = rel.get("target_id")
                    if isinstance(target_id, str):
                        links.append(target_id)
        evidence = node.get("evidence", [])
        if isinstance(evidence, list):
            for ev in evidence:
                if isinstance(ev, dict):
                    source_id = ev.get("source_id")
                    if isinstance(source_id, str):
                        links.append(source_id)

    if layer == "RG":
        for key in ("assumptions", "risks"):
            value = node.get(key, [])
            if isinstance(value, list):
                links.extend(str(item) for item in value if isinstance(item, str))
        links_map = node.get("refs", {})
        if isinstance(links_map, dict):
            for key in ("kg_refs", "tg_refs", "artifacts"):
                values = links_map.get(key, [])
                if isinstance(values, list):
                    links.extend(str(item) for item in values if isinstance(item, str))

    if layer == "TG":
        for key in ("dependencies", "blocks"):
            value = node.get(key, [])
            if isinstance(value, list):
                links.extend(str(item) for item in value if isinstance(item, str))

        inputs = node.get("inputs", {})
        if isinstance(inputs, dict):
            for key in ("kg_refs", "rg_refs", "sources"):
                values = inputs.get(key, [])
                if isinstance(values, list):
                    links.extend(str(item) for item in values if isinstance(item, str))

        outputs = node.get("outputs", {})
        if isinstance(outputs, dict):
            for key in ("artifacts", "kg_updates", "rg_updates"):
                values = outputs.get(key, [])
                if isinstance(values, list):
                    links.extend(str(item) for item in values if isinstance(item, str))

    return links


def validate_graph_store(paths: GraphPaths, strict_artifacts: bool = False) -> Tuple[List[str], List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    infos: List[str] = []

    required_fields = {
        "KG": ["id", "type", "title", "created_at", "updated_at", "links", "tags", "rev", "summary", "attributes", "relationships", "evidence", "status"],
        "RG": ["id", "type", "title", "created_at", "updated_at", "links", "tags", "rev", "context", "decision", "rationale", "alternatives", "assumptions", "risks", "validation", "outcome"],
        "TG": ["id", "type", "title", "created_at", "updated_at", "links", "tags", "rev", "description", "status", "priority", "owner_role", "dependencies", "blocks", "acceptance_criteria", "inputs", "outputs", "log"],
    }

    all_records_by_layer: Dict[str, List[Dict[str, Any]]] = {"KG": [], "RG": [], "TG": []}

    for layer in ("KG", "RG", "TG"):
        path = paths.node_file(layer)
        try:
            records = read_jsonl(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        all_records_by_layer[layer] = records

        seen_id_rev: set[Tuple[str, int]] = set()
        for idx, record in enumerate(records, start=1):
            for field in required_fields[layer]:
                if field not in record:
                    errors.append(f"{path}: line {idx} missing required field '{field}'")

            node_id = record.get("id")
            rev = record.get("rev")
            if not isinstance(node_id, str) or not node_id:
                errors.append(f"{path}: line {idx} invalid id")
            if not isinstance(rev, int) or rev < 1:
                errors.append(f"{path}: line {idx} invalid rev")
            if isinstance(node_id, str) and isinstance(rev, int):
                marker = (node_id, rev)
                if marker in seen_id_rev:
                    errors.append(f"{path}: duplicate id+rev pair {node_id} rev={rev}")
                seen_id_rev.add(marker)

            for time_field in ("created_at", "updated_at"):
                value = record.get(time_field)
                if not isinstance(value, str) or ISO_PATTERN.match(value) is None:
                    errors.append(f"{path}: line {idx} invalid {time_field} (expected ISO UTC Z format)")

    all_internal_ids: set[str] = set()
    latest_tg = latest_by_id(all_records_by_layer["TG"])

    for layer in ("KG", "RG", "TG"):
        for record in all_records_by_layer[layer]:
            node_id = record.get("id")
            if isinstance(node_id, str):
                all_internal_ids.add(node_id)

    for layer in ("KG", "RG", "TG"):
        for record in all_records_by_layer[layer]:
            node_id = record.get("id", "<missing-id>")
            links = extract_links(record, layer)
            for ref in links:
                if ref.startswith("artifact://"):
                    artifact_path = ref[len("artifact://") :]
                    if not artifact_path:
                        errors.append(f"{node_id}: invalid empty artifact URI")
                    elif strict_artifacts:
                        full = (paths.project_root / artifact_path).resolve() if not os.path.isabs(artifact_path) else Path(artifact_path)
                        if not full.exists():
                            warnings.append(f"{node_id}: artifact missing on disk: {ref}")
                    continue

                if ref.startswith("context_pack://"):
                    pack_rel = ref[len("context_pack://") :]
                    pack_path = paths.context_packs_dir / pack_rel
                    if pack_path.exists():
                        continue
                    infos.append(f"{node_id}: context pack ref rotated or unavailable: {ref}")
                    continue

                if INTERNAL_ID_PATTERN.match(ref):
                    if ref not in all_internal_ids:
                        errors.append(f"{node_id}: missing required internal reference: {ref}")
                    continue

                if ref.startswith("ART-"):
                    warnings.append(f"{node_id}: non-URI artifact reference kept as optional: {ref}")
                    continue

                if ref.startswith("http://") or ref.startswith("https://"):
                    continue

                if ref.startswith("KG-") or ref.startswith("RG-") or ref.startswith("TG-"):
                    errors.append(f"{node_id}: malformed internal id reference: {ref}")

    conflicts: List[Dict[str, Any]] = []
    try:
        conflicts = read_jsonl(paths.conflicts_file)
    except ValueError as exc:
        errors.append(str(exc))

    for idx, conflict in enumerate(conflicts, start=1):
        cid = conflict.get("id", f"conflict-line-{idx}")
        resolution_ids = conflict.get("resolution_task_ids")
        if not isinstance(resolution_ids, list) or not resolution_ids:
            errors.append(f"{cid}: conflict entry requires at least one resolution_task_id")
            continue
        for task_id in resolution_ids:
            task = latest_tg.get(task_id)
            if task is None:
                errors.append(f"{cid}: resolution task missing: {task_id}")
                continue
            tags = task.get("tags", [])
            if not isinstance(tags, list) or "conflict_resolution" not in tags:
                errors.append(f"{cid}: resolution task must include tag 'conflict_resolution': {task_id}")

    try:
        commits = read_jsonl(paths.post_commits_file)
    except ValueError as exc:
        commits = []
        errors.append(str(exc))

    latest_commit_id = None
    if commits:
        latest_commit = commits[-1]
        latest_commit_id = latest_commit.get("commit_id")
        if not isinstance(latest_commit_id, str) or not latest_commit_id:
            errors.append("latest commit missing valid commit_id")

    latest_context = read_json(paths.context_latest_file)
    if latest_context is None:
        errors.append(f"missing {paths.context_latest_file}")
    else:
        context_commit_id = latest_context.get("commit_id")
        if latest_commit_id and context_commit_id != latest_commit_id:
            errors.append("context_pack.latest.json is stale: commit_id does not match latest post_response_commits")

    drift, _counts = rebuild_indexes(paths, verify_only=True, dry_run=False)
    if drift:
        errors.append("Index drift detected: run rebuild_indexes.py")

    return errors, warnings, infos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize and maintain 3-layer graph stores.")
    parser.add_argument("--project-root", required=True, help="Absolute or relative path to project root")
    parser.add_argument("--graphs-dir", default="graphs", help="Graph directory under project root")
    parser.add_argument("--project-name", default=None, help="Project name override")
    parser.add_argument("--mode", choices=["init", "repair", "reset"], default="init")
    parser.add_argument("--seed-universal-starters", type=parse_bool, default=True)
    parser.add_argument("--seed-project-starters", type=parse_bool, default=False)
    parser.add_argument("--clock", choices=["utc", "local"], default="utc")
    parser.add_argument("--archive-dir", default=None, help="Archive directory for reset mode")
    parser.add_argument("--context-pack-history-limit", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without writing files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.exists() or not project_root.is_dir():
        print(f"[ERROR] project root not found or not a directory: {project_root}")
        return 2

    if args.context_pack_history_limit < 1:
        print("[ERROR] --context-pack-history-limit must be >= 1")
        return 2

    graphs_dir = (project_root / args.graphs_dir).resolve()
    project_name = args.project_name or project_root.name

    archive_dir = Path(args.archive_dir).expanduser().resolve() if args.archive_dir else (project_root / "graphs_archive").resolve()

    paths = GraphPaths(project_root=project_root, graphs_dir=graphs_dir)

    try:
        if args.mode == "init":
            return init_mode(
                paths=paths,
                project_name=project_name,
                clock=args.clock,
                seed_universal_starters=args.seed_universal_starters,
                seed_project_starters=args.seed_project_starters,
                history_limit=args.context_pack_history_limit,
                dry_run=args.dry_run,
            )

        if args.mode == "repair":
            return repair_mode(
                paths=paths,
                project_name=project_name,
                clock=args.clock,
                seed_universal_starters=args.seed_universal_starters,
                seed_project_starters=args.seed_project_starters,
                history_limit=args.context_pack_history_limit,
                dry_run=args.dry_run,
            )

        return reset_mode(
            paths=paths,
            project_name=project_name,
            clock=args.clock,
            seed_universal_starters=args.seed_universal_starters,
            seed_project_starters=args.seed_project_starters,
            history_limit=args.context_pack_history_limit,
            archive_dir=archive_dir,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
