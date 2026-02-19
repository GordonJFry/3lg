#!/usr/bin/env python3
"""Review and consolidate TG/RG/conflicts with deterministic append-only semantics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

ACTION_ORDER = {
    "archive_task": 0,
    "supersede_decision": 1,
    "create_conflict_task": 2,
    "resolve_conflict": 3,
}


class ConsolidateError(RuntimeError):
    """Raised for deterministic consolidation errors."""


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


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


def parse_bool(raw: str) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw}")


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def unique_keep_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def parse_statuses(raw: str) -> List[str]:
    values = [normalize_status(part) for part in str(raw).split(",") if str(part).strip()]
    if not values:
        raise ConsolidateError("Status list cannot be empty")
    return unique_keep_order(values)


def normalize_status(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower()
    if normalized == "canceled":
        return "cancelled"
    return normalized


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ConsolidateError(f"Invalid JSONL in {path} line {idx}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConsolidateError(f"Invalid JSONL object in {path} line {idx}")
        rows.append(payload)
    return rows


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


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]], dry_run: bool) -> None:
    serialized = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
    content = ("\n".join(serialized) + "\n") if serialized else ""
    write_text_atomic(path, content, dry_run)


def append_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]], dry_run: bool) -> None:
    append_rows = list(rows)
    if not append_rows:
        return
    existing = read_jsonl(path)
    existing.extend(append_rows)
    write_jsonl_atomic(path, existing, dry_run)


def strip_internal(node: Dict[str, Any]) -> Dict[str, Any]:
    clean = copy.deepcopy(node)
    clean.pop("_line_no", None)
    return clean


class GraphPaths:
    def __init__(self, project_root: Path, graphs_dir_name: str) -> None:
        self.project_root = project_root
        self.graphs_dir = (project_root / graphs_dir_name).resolve()

    @property
    def kg_nodes(self) -> Path:
        return self.graphs_dir / "kg_nodes.jsonl"

    @property
    def rg_nodes(self) -> Path:
        return self.graphs_dir / "rg_nodes.jsonl"

    @property
    def tg_nodes(self) -> Path:
        return self.graphs_dir / "tg_nodes.jsonl"

    @property
    def conflicts(self) -> Path:
        return self.graphs_dir / "conflicts.jsonl"

    @property
    def kg_index(self) -> Path:
        return self.graphs_dir / "kg_index.jsonl"

    @property
    def rg_index(self) -> Path:
        return self.graphs_dir / "rg_index.jsonl"

    @property
    def tg_index(self) -> Path:
        return self.graphs_dir / "tg_index.jsonl"

    @property
    def post_commits(self) -> Path:
        return self.graphs_dir / "post_response_commits.jsonl"

    @property
    def context_latest(self) -> Path:
        return self.graphs_dir / "context_pack.latest.json"

    @property
    def context_packs_dir(self) -> Path:
        return self.graphs_dir / "context_packs"

    @property
    def consolidation_runs_dir(self) -> Path:
        return self.graphs_dir / "consolidation_runs"


def read_raw_layers(paths: GraphPaths) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "KG": read_jsonl(paths.kg_nodes),
        "RG": read_jsonl(paths.rg_nodes),
        "TG": read_jsonl(paths.tg_nodes),
        "CONFLICT": read_jsonl(paths.conflicts),
    }


def _resolve_latest_for_id(records: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    rev_values: List[Optional[int]] = []
    for _, rec in records:
        rev_raw = rec.get("rev")
        rev_values.append(rev_raw if isinstance(rev_raw, int) and rev_raw >= 1 else None)
    all_have_rev = all(value is not None for value in rev_values)

    best_key: Optional[Tuple[Any, ...]] = None
    best_line = -1
    best_record: Optional[Dict[str, Any]] = None

    for line_no, rec in records:
        updated = parse_iso(rec.get("updated_at"))
        updated_epoch = updated.timestamp() if updated else float("-inf")
        rev_raw = rec.get("rev")
        rev = int(rev_raw) if isinstance(rev_raw, int) and rev_raw >= 1 else 0
        key = (rev, updated_epoch, line_no) if all_have_rev else (updated_epoch, line_no)

        if best_key is None or key > best_key:
            best_key = key
            best_line = line_no
            best_record = rec

    if best_record is None:
        raise ConsolidateError("Unable to resolve latest record")

    resolved = copy.deepcopy(best_record)
    resolved["_line_no"] = best_line
    return resolved


def resolve_latest_view(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, rec in enumerate(records, start=1):
        node_id = rec.get("id")
        if not isinstance(node_id, str) or not node_id:
            continue
        grouped.setdefault(node_id, []).append((idx, rec))

    latest: Dict[str, Dict[str, Any]] = {}
    for node_id, rows in grouped.items():
        latest[node_id] = _resolve_latest_for_id(rows)
    return latest


def index_row_for_node(layer: str, node: Dict[str, Any]) -> Dict[str, Any]:
    full_node = strip_internal(node)
    content_node = copy.deepcopy(full_node)
    content_node.pop("updated_at", None)
    if layer == "TG":
        content_node.pop("log", None)

    return {
        "id": full_node.get("id"),
        "title": full_node.get("title", ""),
        "tags": full_node.get("tags", []),
        "updated_at": full_node.get("updated_at", ""),
        "type": full_node.get("type", ""),
        "layer": layer,
        "status": full_node.get("status") if layer == "TG" else None,
        "content_hash": sha256_hex_obj(content_node),
        "full_hash": sha256_hex_obj(full_node),
    }


def build_index_rows(layer: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest = resolve_latest_view(rows)
    return [index_row_for_node(layer, node) for _, node in sorted(latest.items())]


def rebuild_indexes(paths: GraphPaths, dry_run: bool) -> None:
    raw = read_raw_layers(paths)
    write_jsonl_atomic(paths.kg_index, build_index_rows("KG", raw["KG"]), dry_run)
    write_jsonl_atomic(paths.rg_index, build_index_rows("RG", raw["RG"]), dry_run)
    write_jsonl_atomic(paths.tg_index, build_index_rows("TG", raw["TG"]), dry_run)


def verify_indexes(paths: GraphPaths) -> None:
    raw = read_raw_layers(paths)
    expected = {
        "KG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in build_index_rows("KG", raw["KG"])],
        "RG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in build_index_rows("RG", raw["RG"])],
        "TG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in build_index_rows("TG", raw["TG"])],
    }
    actual = {
        "KG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in read_jsonl(paths.kg_index)],
        "RG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in read_jsonl(paths.rg_index)],
        "TG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in read_jsonl(paths.tg_index)],
    }

    for layer in ("KG", "RG", "TG"):
        if expected[layer] != actual[layer]:
            raise ConsolidateError(f"Index verification failed for {layer}")


def task_status(node: Dict[str, Any]) -> str:
    return normalize_status(node.get("status"))


def conflict_status(node: Dict[str, Any]) -> str:
    return normalize_status(node.get("status"))


def sort_key_latest(node: Dict[str, Any]) -> Tuple[int, float, int]:
    rev = int(node.get("rev", 0)) if isinstance(node.get("rev"), int) else 0
    updated = parse_iso(node.get("updated_at"))
    updated_epoch = updated.timestamp() if updated else float("-inf")
    line_no = int(node.get("_line_no", 0)) if isinstance(node.get("_line_no"), int) else 0
    return rev, updated_epoch, line_no


def get_next_rev(existing: Dict[str, Any]) -> int:
    rev_raw = existing.get("rev")
    if isinstance(rev_raw, int) and rev_raw >= 1:
        return rev_raw + 1
    return 2


def is_older_than(node: Dict[str, Any], days: int, now_dt: datetime) -> bool:
    updated = parse_iso(node.get("updated_at"))
    if updated is None:
        return False
    return updated <= now_dt - timedelta(days=days)


def collect_active_task_references(tg_latest: Dict[str, Dict[str, Any]], active_statuses: Set[str]) -> Set[str]:
    referenced: Set[str] = set()
    for node in tg_latest.values():
        if task_status(node) not in active_statuses:
            continue
        deps = node.get("dependencies") if isinstance(node.get("dependencies"), list) else []
        blocks = node.get("blocks") if isinstance(node.get("blocks"), list) else []
        for task_id in deps + blocks:
            if isinstance(task_id, str) and task_id.startswith("TG-"):
                referenced.add(task_id)
    return referenced


def collect_rg_task_refs(rg_latest: Dict[str, Dict[str, Any]]) -> Set[str]:
    refs: Set[str] = set()
    for node in rg_latest.values():
        rg_refs = node.get("refs") if isinstance(node.get("refs"), dict) else {}
        tg_refs = rg_refs.get("tg_refs") if isinstance(rg_refs.get("tg_refs"), list) else []
        for task_id in tg_refs:
            if isinstance(task_id, str) and task_id.startswith("TG-"):
                refs.add(task_id)
    return refs


def unresolved_conflict_rows(conflict_latest: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for conflict_id, row in conflict_latest.items():
        status = conflict_status(row)
        if status not in {"resolved", "closed"}:
            rows[conflict_id] = row
    return rows


def collect_conflict_task_refs(conflict_row: Dict[str, Any]) -> Set[str]:
    refs: Set[str] = set()

    for key in ("node_ids", "resolution_task_ids", "links"):
        values = conflict_row.get(key)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and value.startswith("TG-"):
                    refs.add(value)

    refs_obj = conflict_row.get("refs")
    if isinstance(refs_obj, dict):
        tg_refs = refs_obj.get("tg_refs")
        if isinstance(tg_refs, list):
            for value in tg_refs:
                if isinstance(value, str) and value.startswith("TG-"):
                    refs.add(value)

    return refs


def collect_unresolved_conflict_task_refs(conflict_latest: Dict[str, Dict[str, Any]]) -> Set[str]:
    refs: Set[str] = set()
    for row in unresolved_conflict_rows(conflict_latest).values():
        refs.update(collect_conflict_task_refs(row))
    return refs


def decision_key(node: Dict[str, Any]) -> Optional[str]:
    direct = node.get("decision_key")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    attrs = node.get("attributes")
    if isinstance(attrs, dict):
        nested = attrs.get("decision_key")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()

    return None


def decision_outcome_status(node: Dict[str, Any]) -> str:
    outcome = node.get("outcome")
    if isinstance(outcome, dict):
        value = outcome.get("status")
        if isinstance(value, str):
            return value.strip().lower()
    return ""


def parse_ids(values: Sequence[str]) -> Tuple[List[str], List[str]]:
    valid: List[str] = []
    invalid: List[str] = []
    for value in values:
        if isinstance(value, str) and value.startswith("TG-"):
            valid.append(value)
        elif isinstance(value, str):
            invalid.append(value)
    return valid, invalid


def choose_canonical_decision(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    accepted = [row for row in group if decision_outcome_status(row) == "accepted"]
    if accepted:
        accepted.sort(key=sort_key_latest, reverse=True)
        return accepted[0]

    non_superseded = [row for row in group if decision_outcome_status(row) != "superseded"]
    if non_superseded:
        non_superseded.sort(key=sort_key_latest, reverse=True)
        return non_superseded[0]

    ordered = sorted(group, key=sort_key_latest, reverse=True)
    return ordered[0]


def can_resolve_conflict(
    conflict_row: Dict[str, Any],
    tg_latest: Dict[str, Dict[str, Any]],
) -> Tuple[bool, List[str], List[str], List[str]]:
    reasons: List[str] = []
    linked_tasks_raw = conflict_row.get("resolution_task_ids") if isinstance(conflict_row.get("resolution_task_ids"), list) else []
    linked_tasks, invalid = parse_ids([str(v) for v in linked_tasks_raw if isinstance(v, str)])

    if invalid:
        reasons.append("resolution_task_ids contains non-TG ids")

    if not linked_tasks:
        reasons.append("no linked resolution_task_ids")

    missing: List[str] = []
    not_done: List[str] = []
    for task_id in linked_tasks:
        node = tg_latest.get(task_id)
        if node is None:
            missing.append(task_id)
            continue
        tags = node.get("tags") if isinstance(node.get("tags"), list) else []
        status = task_status(node)
        if "conflict_resolution" not in [str(t) for t in tags]:
            reasons.append(f"linked task missing conflict_resolution tag: {task_id}")
        if status not in {"done", "completed"}:
            not_done.append(task_id)

    if missing:
        reasons.append(f"missing tasks: {', '.join(sorted(missing))}")
    if not_done:
        reasons.append(f"tasks not done/completed: {', '.join(sorted(not_done))}")

    evidence = conflict_row.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        reasons.append("conflict has no evidence")

    return (len(reasons) == 0), reasons, linked_tasks, missing + not_done


def create_conflict_resolution_task(conflict_row: Dict[str, Any], now: str, run_id: str) -> Dict[str, Any]:
    conflict_id = str(conflict_row.get("id", "CONFLICT-UNKNOWN"))
    node_ids = conflict_row.get("node_ids") if isinstance(conflict_row.get("node_ids"), list) else []
    kg_refs = [str(v) for v in node_ids if isinstance(v, str) and v.startswith("KG-")]
    rg_refs = [str(v) for v in node_ids if isinstance(v, str) and v.startswith("RG-")]

    task_id = f"TG-TASK-{new_ulid()}"
    return {
        "id": task_id,
        "type": "task",
        "title": f"Resolve conflict {conflict_id}",
        "description": "Review conflict evidence, choose authoritative claim/decision, and mark conflict resolved.",
        "status": "ready",
        "priority": "high",
        "owner_role": "lead",
        "dependencies": [],
        "blocks": [],
        "acceptance_criteria": [
            "Conflict evidence is reviewed and a final disposition is documented",
            "Conflict record status is updated to resolved or closed with notes",
        ],
        "inputs": {
            "kg_refs": kg_refs,
            "rg_refs": rg_refs,
            "sources": [],
        },
        "outputs": {
            "artifacts": [],
            "kg_updates": [],
            "rg_updates": rg_refs,
        },
        "log": [{"at": now, "event": "created", "by": "review-and-consolidate"}],
        "links": unique_keep_order([conflict_id] + kg_refs + rg_refs),
        "tags": ["conflict_resolution", "maintenance"],
        "rev": 1,
        "created_at": now,
        "updated_at": now,
        "created_by": "review-and-consolidate",
        "origin": "conflict_resolution_seed",
        "origin_consolidation_run_id": run_id,
    }


def make_archive_row(node: Dict[str, Any], now: str, run_id: str, reason: str) -> Dict[str, Any]:
    row = strip_internal(node)
    row["rev"] = get_next_rev(node)
    row["status"] = "archived"
    row["updated_at"] = now
    row["created_by"] = "review-and-consolidate"
    row["origin"] = "task_archive"
    row["origin_consolidation_run_id"] = run_id
    tags = row.get("tags") if isinstance(row.get("tags"), list) else []
    row["tags"] = unique_keep_order([str(t) for t in tags] + ["archived"])
    log = row.get("log") if isinstance(row.get("log"), list) else []
    log.append({"at": now, "event": "archived", "by": "review-and-consolidate", "reason": reason})
    row["log"] = log
    row["archive_reason"] = reason
    row["archived_at"] = now
    return row


def make_superseded_row(node: Dict[str, Any], canonical_id: str, now: str, run_id: str) -> Dict[str, Any]:
    row = strip_internal(node)
    row["rev"] = get_next_rev(node)
    row["updated_at"] = now
    row["created_by"] = "review-and-consolidate"
    row["origin"] = "decision_consolidation"
    row["origin_consolidation_run_id"] = run_id

    outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    supersedes = outcome.get("supersedes") if isinstance(outcome.get("supersedes"), list) else []
    outcome["status"] = "superseded"
    outcome["supersedes"] = unique_keep_order([str(v) for v in supersedes if isinstance(v, str)] + [canonical_id])
    outcome["notes"] = f"superseded by {canonical_id} in consolidation run {run_id}"
    row["outcome"] = outcome

    tags = row.get("tags") if isinstance(row.get("tags"), list) else []
    filtered = [str(tag) for tag in tags if not str(tag).startswith("outcome:")]
    row["tags"] = unique_keep_order(filtered + ["outcome:superseded"])
    return row


def make_conflict_link_update_row(conflict_row: Dict[str, Any], task_id: str, now: str, run_id: str) -> Dict[str, Any]:
    row = strip_internal(conflict_row)
    row["rev"] = get_next_rev(conflict_row)
    row["updated_at"] = now
    tasks = row.get("resolution_task_ids") if isinstance(row.get("resolution_task_ids"), list) else []
    row["resolution_task_ids"] = unique_keep_order([str(v) for v in tasks if isinstance(v, str)] + [task_id])
    notes = row.get("notes") if isinstance(row.get("notes"), str) else ""
    extra = f"seeded conflict_resolution task {task_id} via run {run_id}"
    row["notes"] = f"{notes}; {extra}" if notes else extra
    return row


def make_resolved_conflict_row(conflict_row: Dict[str, Any], now: str, run_id: str, resolution_tasks: List[str]) -> Dict[str, Any]:
    row = strip_internal(conflict_row)
    row["rev"] = get_next_rev(conflict_row)
    row["status"] = "resolved"
    row["updated_at"] = now
    tasks = row.get("resolution_task_ids") if isinstance(row.get("resolution_task_ids"), list) else []
    row["resolution_task_ids"] = unique_keep_order([str(v) for v in tasks if isinstance(v, str)] + resolution_tasks)
    notes = row.get("notes") if isinstance(row.get("notes"), str) else ""
    extra = f"resolved by review-and-consolidate run {run_id}"
    row["notes"] = f"{notes}; {extra}" if notes else extra
    return row


def build_post_response_context_pack(
    paths: GraphPaths,
    commit_id: str,
    changed_node_ids: List[str],
    goal_hint: str,
) -> Dict[str, Any]:
    raw_layers = read_raw_layers(paths)
    kg_latest = resolve_latest_view(raw_layers["KG"])
    rg_latest = resolve_latest_view(raw_layers["RG"])
    tg_latest = resolve_latest_view(raw_layers["TG"])

    tg_nodes = [strip_internal(node) for node in tg_latest.values()]
    tg_nodes.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)

    active_tasks = [
        node for node in tg_nodes if task_status(node) in {"ready", "in_progress", "blocked", "review"}
    ]
    goal = str(active_tasks[0].get("title", goal_hint)) if active_tasks else goal_hint

    task_snapshot = [
        {
            "id": node.get("id"),
            "title": node.get("title"),
            "status": node.get("status"),
            "deps": node.get("dependencies", []),
            "acceptance_criteria": node.get("acceptance_criteria", []),
        }
        for node in active_tasks[:15]
    ]

    kg_nodes = [strip_internal(node) for node in kg_latest.values()]
    kg_nodes.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)

    def max_confidence(node: Dict[str, Any]) -> float:
        evidence = node.get("evidence") if isinstance(node.get("evidence"), list) else []
        best = 0.0
        for row in evidence:
            if not isinstance(row, dict):
                continue
            try:
                best = max(best, float(row.get("confidence", 0.0)))
            except (TypeError, ValueError):
                continue
        return round(best, 4)

    knowledge_snapshot = [
        {
            "id": node.get("id"),
            "title": node.get("title"),
            "summary": node.get("summary", ""),
            "confidence": max_confidence(node),
        }
        for node in kg_nodes[:30]
    ]

    rg_nodes = [strip_internal(node) for node in rg_latest.values()]
    rg_nodes.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)

    changed_rg_ids = [node_id for node_id in changed_node_ids if node_id.startswith("RG-")]
    prioritized: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for node_id in changed_rg_ids:
        node = rg_latest.get(node_id)
        if node is None:
            continue
        if node_id in seen:
            continue
        seen.add(node_id)
        prioritized.append(node)
    for node in rg_nodes:
        node_id = str(node.get("id", ""))
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        prioritized.append(node)

    decision_snapshot = [
        {
            "id": node.get("id"),
            "title": node.get("title"),
            "decision": node.get("decision", ""),
            "status": node.get("outcome", {}).get("status", ""),
        }
        for node in prioritized[:10]
    ]

    recent_deltas = []
    for node_id in changed_node_ids[:20]:
        layer = node_id.split("-", 1)[0] if "-" in node_id else "UNKNOWN"
        recent_deltas.append({"layer": layer, "id": node_id, "change": "added_or_updated"})

    return {
        "goal": goal,
        "generated_at": now_iso_utc(),
        "commit_id": commit_id,
        "active_tasks": [node.get("id") for node in active_tasks[:15]],
        "task_snapshot": task_snapshot,
        "knowledge_snapshot": knowledge_snapshot,
        "decision_snapshot": decision_snapshot,
        "recent_deltas": recent_deltas,
    }


def write_context_pack(paths: GraphPaths, context_pack: Dict[str, Any], history_limit: int, dry_run: bool) -> str:
    stamp = timestamp_slug()
    history_name = f"context_pack.{stamp}.{context_pack['commit_id']}.json"
    history_path = paths.context_packs_dir / history_name

    write_json_atomic(history_path, context_pack, dry_run)

    if not dry_run:
        written = read_json(history_path)
        if written is None or written.get("commit_id") != context_pack.get("commit_id"):
            raise ConsolidateError("Context history write validation failed")

    write_json_atomic(paths.context_latest, context_pack, dry_run)

    history_files = sorted(paths.context_packs_dir.glob("context_pack.*.json"))
    if len(history_files) > history_limit:
        for old in history_files[: len(history_files) - history_limit]:
            if dry_run:
                print(f"[DRY-RUN] remove {old}")
            else:
                old.unlink(missing_ok=True)

    return str(history_path)


def append_commit_row(
    paths: GraphPaths,
    commit_id: str,
    consolidation_run_id: str,
    consolidation_seq: int,
    changed_node_ids: List[str],
    actions_summary: Dict[str, int],
    context_pack_path: str,
    request_id: Optional[str],
    dry_run: bool,
) -> None:
    row: Dict[str, Any] = {
        "mode": "review_consolidate_apply",
        "event": "post_response_commit",
        "commit_id": commit_id,
        "consolidation_run_id": consolidation_run_id,
        "consolidation_seq": consolidation_seq,
        "changed_node_ids": changed_node_ids,
        "actions_summary": actions_summary,
        "context_pack_path": context_pack_path,
        "created_at": now_iso_utc(),
    }
    if request_id:
        row["request_id"] = request_id

    append_jsonl_atomic(paths.post_commits, [row], dry_run)


def consolidation_run_files(paths: GraphPaths) -> List[Path]:
    if not paths.consolidation_runs_dir.exists():
        return []
    return sorted(paths.consolidation_runs_dir.glob("consolidation_run.*.json"))


def load_consolidation_seq_values(paths: GraphPaths) -> List[int]:
    values: List[int] = []
    for file_path in consolidation_run_files(paths):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        seq = payload.get("consolidation_seq")
        if isinstance(seq, int) and seq >= 1:
            values.append(seq)
    return values


def next_consolidation_seq(paths: GraphPaths) -> int:
    seq_values = load_consolidation_seq_values(paths)
    return max(seq_values) + 1 if seq_values else 1


def rotate_consolidation_history(paths: GraphPaths, history_limit: int, dry_run: bool) -> None:
    files = consolidation_run_files(paths)
    if len(files) <= history_limit:
        return
    for old in files[: len(files) - history_limit]:
        if dry_run:
            print(f"[DRY-RUN] remove {old}")
        else:
            old.unlink(missing_ok=True)


def write_consolidation_history(
    paths: GraphPaths,
    payload: Dict[str, Any],
    history_limit: int,
    dry_run: bool,
) -> str:
    run_id = str(payload["consolidation_run_id"])
    file_name = f"consolidation_run.{timestamp_slug()}.{run_id}.json"
    file_path = paths.consolidation_runs_dir / file_name
    write_json_atomic(file_path, payload, dry_run)
    if not dry_run:
        written = read_json(file_path)
        if written is None or written.get("consolidation_run_id") != run_id:
            raise ConsolidateError("Consolidation history validation failed")
    rotate_consolidation_history(paths, history_limit=history_limit, dry_run=dry_run)
    return str(file_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review and consolidate TG/RG/conflict graph state")
    parser.add_argument("--project-root", required=True, help="Project root path")
    parser.add_argument("--graphs-dir", default="graphs", help="Graph directory under project root")

    parser.add_argument("--apply", type=parse_bool, default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stdout-json", type=parse_bool, default=False)

    parser.add_argument("--archive-tasks", type=parse_bool, default=True)
    parser.add_argument("--supersede-decisions", type=parse_bool, default=True)
    parser.add_argument("--resolve-conflicts", type=parse_bool, default=True)
    parser.add_argument("--archive-age-days", type=int, default=14)
    parser.add_argument("--context-pack-history-limit", type=int, default=200)
    parser.add_argument("--consolidation-history-limit", type=int, default=200)

    parser.add_argument("--terminal-task-statuses", default="done,completed,cancelled,canceled,superseded")
    parser.add_argument("--active-task-statuses", default="ready,in_progress,blocked,review")
    parser.add_argument("--create-missing-conflict-task", type=parse_bool, default=False)
    parser.add_argument("--strict-conflicts", type=parse_bool, default=False)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--source-title", default="Periodic review and consolidate")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.archive_age_days < 0:
        print("[ERROR] --archive-age-days must be >= 0")
        return 2
    if args.context_pack_history_limit < 1 or args.consolidation_history_limit < 1:
        print("[ERROR] history limits must be >= 1")
        return 2

    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.exists() or not project_root.is_dir():
        print(f"[ERROR] invalid project root: {project_root}")
        return 2

    paths = GraphPaths(project_root=project_root, graphs_dir_name=args.graphs_dir)
    if not paths.graphs_dir.exists() or not paths.graphs_dir.is_dir():
        print(f"[ERROR] graphs directory not found: {paths.graphs_dir}")
        return 2

    try:
        terminal_statuses = set(parse_statuses(args.terminal_task_statuses))
        active_statuses = set(parse_statuses(args.active_task_statuses))

        now = now_iso_utc()
        now_dt = parse_iso(now)
        if now_dt is None:
            raise ConsolidateError("Failed to compute current UTC timestamp")

        raw_layers = read_raw_layers(paths)
        latest = {
            "KG": resolve_latest_view(raw_layers["KG"]),
            "RG": resolve_latest_view(raw_layers["RG"]),
            "TG": resolve_latest_view(raw_layers["TG"]),
            "CONFLICT": resolve_latest_view(raw_layers["CONFLICT"]),
        }

        consolidation_run_id = f"CONRUN-{new_ulid()}"

        actions: List[Dict[str, Any]] = []
        warnings: List[str] = []

        changed_tg_rows: List[Dict[str, Any]] = []
        changed_rg_rows: List[Dict[str, Any]] = []
        changed_conflict_rows: List[Dict[str, Any]] = []
        changed_node_ids: List[str] = []

        active_inbound_refs = collect_active_task_references(latest["TG"], active_statuses)
        rg_task_refs = collect_rg_task_refs(latest["RG"])
        unresolved_conflicts = unresolved_conflict_rows(latest["CONFLICT"])
        unresolved_conflict_task_refs = collect_unresolved_conflict_task_refs(latest["CONFLICT"])

        if args.archive_tasks:
            for task_id, node in sorted(latest["TG"].items()):
                status = task_status(node)
                if status not in terminal_statuses:
                    continue
                if status == "archived":
                    continue

                if not is_older_than(node, args.archive_age_days, now_dt):
                    continue

                reasons_blocked: List[str] = []
                if task_id in active_inbound_refs:
                    reasons_blocked.append("referenced by active TG dependency/block links")
                if task_id in unresolved_conflict_task_refs:
                    reasons_blocked.append("referenced by unresolved conflict")
                if task_id in rg_task_refs:
                    reasons_blocked.append("referenced by latest RG decision refs.tg_refs")

                if reasons_blocked:
                    warnings.append(f"Skip archive {task_id}: {', '.join(reasons_blocked)}")
                    continue

                reason = f"terminal status '{status}' older than {args.archive_age_days} days"
                actions.append(
                    {
                        "action": "archive_task",
                        "target_id": task_id,
                        "reason": reason,
                    }
                )

                if args.apply:
                    row = make_archive_row(node, now=now, run_id=consolidation_run_id, reason=reason)
                    changed_tg_rows.append(row)
                    changed_node_ids.append(task_id)

        if args.supersede_decisions:
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for node in latest["RG"].values():
                if str(node.get("type", "")) != "decision":
                    continue
                key = decision_key(node)
                if not key:
                    node_id = str(node.get("id", "<missing-id>"))
                    warnings.append(f"RG decision {node_id} missing decision_key; skipped from supersede grouping")
                    continue
                grouped.setdefault(key, []).append(node)

            for key in sorted(grouped.keys()):
                group = grouped[key]
                if len(group) <= 1:
                    continue

                canonical = choose_canonical_decision(group)
                canonical_id = str(canonical.get("id"))

                for node in sorted(group, key=lambda row: str(row.get("id", ""))):
                    node_id = str(node.get("id", ""))
                    if node_id == canonical_id:
                        continue

                    outcome = node.get("outcome") if isinstance(node.get("outcome"), dict) else {}
                    status = str(outcome.get("status", "")).lower()
                    supersedes = outcome.get("supersedes") if isinstance(outcome.get("supersedes"), list) else []
                    already_points = canonical_id in [str(v) for v in supersedes if isinstance(v, str)]
                    if status == "superseded" and already_points:
                        continue

                    actions.append(
                        {
                            "action": "supersede_decision",
                            "target_id": node_id,
                            "decision_key": key,
                            "canonical_id": canonical_id,
                            "reason": "non-canonical decision in decision_key group",
                        }
                    )

                    if args.apply:
                        row = make_superseded_row(node, canonical_id=canonical_id, now=now, run_id=consolidation_run_id)
                        changed_rg_rows.append(row)
                        changed_node_ids.append(node_id)

        if args.resolve_conflicts:
            for conflict_id, conflict_row in sorted(unresolved_conflicts.items()):
                can_resolve, reasons, linked_tasks, _ = can_resolve_conflict(conflict_row, latest["TG"])

                if can_resolve:
                    actions.append(
                        {
                            "action": "resolve_conflict",
                            "target_id": conflict_id,
                            "resolution_task_ids": linked_tasks,
                            "reason": "resolution tasks done/completed with evidence",
                        }
                    )
                    if args.apply:
                        row = make_resolved_conflict_row(
                            conflict_row,
                            now=now,
                            run_id=consolidation_run_id,
                            resolution_tasks=linked_tasks,
                        )
                        changed_conflict_rows.append(row)
                        changed_node_ids.append(conflict_id)
                    continue

                missing_link = any("no linked resolution_task_ids" in reason for reason in reasons)
                if missing_link and args.create_missing_conflict_task:
                    actions.append(
                        {
                            "action": "create_conflict_task",
                            "target_id": conflict_id,
                            "reason": "missing linked conflict_resolution task",
                        }
                    )
                    if args.apply:
                        task_row = create_conflict_resolution_task(conflict_row, now=now, run_id=consolidation_run_id)
                        changed_tg_rows.append(task_row)
                        changed_node_ids.append(str(task_row.get("id")))

                        conflict_link_row = make_conflict_link_update_row(
                            conflict_row,
                            task_id=str(task_row.get("id")),
                            now=now,
                            run_id=consolidation_run_id,
                        )
                        changed_conflict_rows.append(conflict_link_row)
                        changed_node_ids.append(conflict_id)

                warning = f"Conflict {conflict_id} not resolvable: {', '.join(reasons)}"
                warnings.append(warning)

        actions.sort(key=lambda row: (ACTION_ORDER.get(str(row.get("action", "")), 999), str(row.get("target_id", ""))))

        if args.strict_conflicts and args.resolve_conflicts:
            blocking = [w for w in warnings if w.startswith("Conflict ") and "not resolvable" in w]
            if blocking:
                raise ConsolidateError(
                    "--strict-conflicts true and unresolved conflict prerequisites found: " + "; ".join(blocking[:5])
                )

        stats = {
            "candidate_actions": len(actions),
            "archive_candidates": sum(1 for row in actions if row.get("action") == "archive_task"),
            "supersede_candidates": sum(1 for row in actions if row.get("action") == "supersede_decision"),
            "create_conflict_task_candidates": sum(1 for row in actions if row.get("action") == "create_conflict_task"),
            "resolve_conflict_candidates": sum(1 for row in actions if row.get("action") == "resolve_conflict"),
            "apply_requested": bool(args.apply),
        }

        output: Dict[str, Any] = {
            "consolidation_run_id": consolidation_run_id,
            "generated_at": now,
            "scope": {
                "archive_tasks": bool(args.archive_tasks),
                "supersede_decisions": bool(args.supersede_decisions),
                "resolve_conflicts": bool(args.resolve_conflicts),
            },
            "candidate_actions": actions,
            "warnings": warnings,
            "stats": stats,
        }
        if args.request_id:
            output["request_id"] = args.request_id

        if args.apply:
            action_counts = {
                "archive_task": sum(1 for row in actions if row.get("action") == "archive_task"),
                "supersede_decision": sum(1 for row in actions if row.get("action") == "supersede_decision"),
                "create_conflict_task": sum(1 for row in actions if row.get("action") == "create_conflict_task"),
                "resolve_conflict": sum(1 for row in actions if row.get("action") == "resolve_conflict"),
            }

            output["stats"]["tg_rows_written"] = len(changed_tg_rows)
            output["stats"]["rg_rows_written"] = len(changed_rg_rows)
            output["stats"]["conflict_rows_written"] = len(changed_conflict_rows)

            changed_node_ids = unique_keep_order(changed_node_ids)
            has_changes = bool(changed_node_ids)

            if has_changes:
                for row in changed_tg_rows:
                    for field in ("created_at", "updated_at"):
                        val = row.get(field)
                        if not isinstance(val, str) or ISO_PATTERN.match(val) is None:
                            raise ConsolidateError(f"TG row invalid {field}")
                for row in changed_rg_rows:
                    for field in ("created_at", "updated_at"):
                        val = row.get(field)
                        if not isinstance(val, str) or ISO_PATTERN.match(val) is None:
                            raise ConsolidateError(f"RG row invalid {field}")
                for row in changed_conflict_rows:
                    for field in ("created_at", "updated_at"):
                        val = row.get(field)
                        if not isinstance(val, str) or ISO_PATTERN.match(val) is None:
                            raise ConsolidateError(f"Conflict row invalid {field}")

                append_jsonl_atomic(paths.tg_nodes, changed_tg_rows, args.dry_run)
                append_jsonl_atomic(paths.rg_nodes, changed_rg_rows, args.dry_run)
                append_jsonl_atomic(paths.conflicts, changed_conflict_rows, args.dry_run)

                rebuild_indexes(paths, args.dry_run)
                if not args.dry_run:
                    verify_indexes(paths)

            if has_changes:
                consolidation_seq = next_consolidation_seq(paths)
                output["consolidation_seq"] = consolidation_seq

                commit_id = f"COMMIT-{new_ulid()}"
                context_pack = build_post_response_context_pack(
                    paths=paths,
                    commit_id=commit_id,
                    changed_node_ids=changed_node_ids,
                    goal_hint=str(args.source_title),
                )

                context_pack_path = write_context_pack(
                    paths=paths,
                    context_pack=context_pack,
                    history_limit=args.context_pack_history_limit,
                    dry_run=args.dry_run,
                )

                try:
                    append_commit_row(
                        paths=paths,
                        commit_id=commit_id,
                        consolidation_run_id=consolidation_run_id,
                        consolidation_seq=consolidation_seq,
                        changed_node_ids=changed_node_ids,
                        actions_summary=action_counts,
                        context_pack_path=context_pack_path,
                        request_id=args.request_id,
                        dry_run=args.dry_run,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    raise ConsolidateError(
                        "Context pack updated but commit row append failed. "
                        "Re-run review-consolidate with --apply true to backfill post_response_commits. "
                        f"Original error: {exc}"
                    ) from exc

                run_payload = {
                    "consolidation_run_id": consolidation_run_id,
                    "consolidation_seq": consolidation_seq,
                    "generated_at": now,
                    "source_title": str(args.source_title),
                    "request_id": args.request_id,
                    "candidate_actions": actions,
                    "applied_actions": [
                        {
                            "action": row.get("action"),
                            "target_id": row.get("target_id"),
                        }
                        for row in actions
                    ],
                    "warnings": warnings,
                    "stats": output["stats"],
                    "changed_node_ids": changed_node_ids,
                    "commit_id": commit_id,
                    "context_pack_path": context_pack_path,
                }

                run_history_path = write_consolidation_history(
                    paths=paths,
                    payload=run_payload,
                    history_limit=args.consolidation_history_limit,
                    dry_run=args.dry_run,
                )

                output["commit_id"] = commit_id
                output["context_pack_path"] = context_pack_path
                output["consolidation_history_path"] = run_history_path
            else:
                warnings.append("Apply mode produced no graph changes; commit/context/history skipped")

        if args.stdout_json:
            print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(f"[OK] consolidation_run_id: {consolidation_run_id}")
            print(f"[OK] candidate_actions: {len(actions)}")
            if output.get("consolidation_seq") is not None:
                print(f"[OK] consolidation_seq: {output['consolidation_seq']}")
            if args.apply:
                print(f"[OK] tg_rows_written: {output.get('stats', {}).get('tg_rows_written', 0)}")
                print(f"[OK] rg_rows_written: {output.get('stats', {}).get('rg_rows_written', 0)}")
                print(f"[OK] conflict_rows_written: {output.get('stats', {}).get('conflict_rows_written', 0)}")
                if output.get("commit_id"):
                    print(f"[OK] commit_id: {output['commit_id']}")
                if output.get("context_pack_path"):
                    print(f"[OK] context_pack_path: {output['context_pack_path']}")
                if output.get("consolidation_history_path"):
                    print(f"[OK] consolidation_history_path: {output['consolidation_history_path']}")
            for warning in warnings:
                print(f"[WARN] {warning}")

        return 0

    except ConsolidateError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] unexpected failure: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
