#!/usr/bin/env python3
"""Convert user requests into deterministic TG task plans and optionally apply them."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
INTERNAL_ID_PATTERN = re.compile(r"\b(?:KG|RG|TG)-[A-Z0-9-]{8,}\b")

INTENTS = {"auto", "plan", "execute", "explain", "debug", "research"}
VALID_PHASES = ["setup", "implement", "verify", "docs", "deploy"]
PHASE_ORDER = {phase: idx for idx, phase in enumerate(VALID_PHASES)}
PRIORITY_ORDER = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
OWNER_ROLES = {"lead", "research", "build", "qa", "ops", "user"}
CONTEXT_PACK_HISTORY_LIMIT_DEFAULT = 200
INTENT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "debug": (
        "bug",
        "error",
        "debug",
        "fail",
        "exception",
        "ошиб",
        "баг",
        "исключен",
        "трейс",
        "сломал",
        "не работает",
    ),
    "execute": (
        "implement",
        "build",
        "execute",
        "run",
        "fix",
        "create",
        "apply",
        "сделай",
        "выполни",
        "запусти",
        "исправ",
        "создай",
        "реализ",
        "примени",
    ),
    "plan": (
        "plan",
        "roadmap",
        "design",
        "task",
        "architecture",
        "план",
        "дорожн",
        "декомпоз",
        "архитектур",
        "задач",
    ),
    "research": (
        "research",
        "investigate",
        "analyze",
        "explore",
        "compare",
        "исслед",
        "проанализ",
        "разбер",
        "сравни",
        "изучи",
    ),
}

VAGUE_TOKENS = {
    "ensure",
    "properly",
    "etc",
    "as needed",
    "should work",
    "handle edge cases",
}


class PlanIntoTasksError(RuntimeError):
    """Raised when deterministic planning fails."""


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def parse_bool(raw: str) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw}")


def parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def tokenize(text: str) -> List[str]:
    lowered = text.lower()
    lowered = lowered.replace("_", " ")
    lowered = NON_WORD_RE.sub(" ", lowered)
    lowered = WHITESPACE_RE.sub(" ", lowered).strip()
    return [tok for tok in WORD_PATTERN.findall(lowered) if len(tok) > 1]


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
            raise PlanIntoTasksError(f"Invalid JSON in {path} line {idx}: {exc}") from exc
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


def write_json_atomic(path: Path, payload: Dict[str, Any], dry_run: bool) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", dry_run)


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]], dry_run: bool) -> None:
    content_rows = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
    content = ("\n".join(content_rows) + "\n") if content_rows else ""
    write_text_atomic(path, content, dry_run)


def append_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]], dry_run: bool) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    existing = read_jsonl(path)
    existing.extend(rows_list)
    write_jsonl_atomic(path, existing, dry_run)


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
    def kg_index(self) -> Path:
        return self.graphs_dir / "kg_index.jsonl"

    @property
    def rg_index(self) -> Path:
        return self.graphs_dir / "rg_index.jsonl"

    @property
    def tg_index(self) -> Path:
        return self.graphs_dir / "tg_index.jsonl"

    @property
    def post_response_commits(self) -> Path:
        return self.graphs_dir / "post_response_commits.jsonl"

    @property
    def plan_history_dir(self) -> Path:
        return self.graphs_dir / "task_plans"

    @property
    def context_latest(self) -> Path:
        return self.graphs_dir / "context_pack.latest.json"

    @property
    def context_packs_dir(self) -> Path:
        return self.graphs_dir / "context_packs"


def read_raw_layers(paths: GraphPaths) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "KG": read_jsonl(paths.kg_nodes),
        "RG": read_jsonl(paths.rg_nodes),
        "TG": read_jsonl(paths.tg_nodes),
    }


def _resolve_latest_for_id(records: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    rev_values: List[Optional[int]] = []
    for _, rec in records:
        rev = rec.get("rev")
        rev_values.append(rev if isinstance(rev, int) and rev >= 1 else None)

    all_have_rev = all(v is not None for v in rev_values)

    best_key: Optional[Tuple[Any, ...]] = None
    best_record: Optional[Dict[str, Any]] = None
    best_line = -1

    for line_no, rec in records:
        updated = parse_iso(rec.get("updated_at"))
        updated_epoch = updated.timestamp() if updated else float("-inf")
        rev = rec.get("rev") if isinstance(rec.get("rev"), int) else None

        if all_have_rev:
            key = (int(rev), updated_epoch, line_no)
        else:
            key = (updated_epoch, line_no)

        if best_key is None or key > best_key:
            best_key = key
            best_record = rec
            best_line = line_no

    if best_record is None:
        raise PlanIntoTasksError("Unable to resolve latest record for grouped id")

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
    for node_id, items in grouped.items():
        latest[node_id] = _resolve_latest_for_id(items)
    return latest


def normalize_request_text(raw: str) -> str:
    # Fixed normalization: trim, CRLF->LF, collapse repeated spaces/tabs within lines, preserve case.
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(normalized_lines).strip()
    return normalized


def classify_intent(request_text: str, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    text = request_text.lower()
    if any(word in text for word in INTENT_KEYWORDS["debug"]):
        return "debug"
    if any(word in text for word in INTENT_KEYWORDS["plan"]):
        return "plan"
    if any(word in text for word in INTENT_KEYWORDS["research"]):
        return "research"
    if any(word in text for word in INTENT_KEYWORDS["execute"]):
        return "execute"
    return "explain"


def _node_search_text(node: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("id", "type", "title", "summary", "decision", "description"):
        value = node.get(key)
        if isinstance(value, str):
            parts.append(value)
    tags = node.get("tags")
    if isinstance(tags, list):
        parts.extend(str(tag) for tag in tags)
    attrs = node.get("attributes")
    if isinstance(attrs, dict):
        for val in attrs.values():
            if isinstance(val, (str, int, float, bool)):
                parts.append(str(val))
    return " ".join(parts)


def extract_entities(
    request_text: str,
    layers_latest: Dict[str, Dict[str, Dict[str, Any]]],
    max_entities: int = 40,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    request_tokens = set(tokenize(request_text))
    explicit_ids = set(INTERNAL_ID_PATTERN.findall(request_text))
    explicit_full_ids = set(re.findall(r"\b(?:KG|RG|TG)-[A-Z0-9-]{8,}\b", request_text))

    entity_rows: Dict[str, Dict[str, Any]] = {}
    scores: Dict[str, float] = {}

    for layer, nodes in layers_latest.items():
        for node_id, node in nodes.items():
            score = 0.0
            reason = ""

            if node_id in explicit_full_ids:
                score = 1.0
                reason = "id_match"
            else:
                node_tokens = set(tokenize(_node_search_text(node)))
                if node_tokens:
                    overlap = request_tokens.intersection(node_tokens)
                    if overlap:
                        coverage = len(overlap) / max(1, len(request_tokens))
                        density = len(overlap) / max(1, len(node_tokens))
                        score = min(0.99, 0.8 * coverage + 0.2 * density)
                        reason = "lexical_overlap"

            if score <= 0.0:
                continue

            scores[node_id] = max(scores.get(node_id, 0.0), score)
            existing = entity_rows.get(node_id)
            row = {
                "node_id": node_id,
                "layer": layer,
                "title": node.get("title", ""),
                "confidence": round(score, 4),
                "reason": reason,
            }
            if existing is None or row["confidence"] > existing["confidence"]:
                entity_rows[node_id] = row

    entities = sorted(entity_rows.values(), key=lambda item: (-item["confidence"], item["layer"], item["node_id"]))
    return entities[:max_entities], scores


def collect_auto_links(
    entities: List[Dict[str, Any]],
    layers_latest: Dict[str, Dict[str, Dict[str, Any]]],
    top_k: int = 12,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    kg_refs: List[str] = []
    rg_refs: List[str] = []
    link_suggestions: List[Dict[str, Any]] = []

    for row in entities[:top_k]:
        node_id = row["node_id"]
        layer = row["layer"]
        confidence = row["confidence"]
        if layer == "KG":
            kg_refs.append(node_id)
        elif layer == "RG":
            rg_refs.append(node_id)
        elif layer == "TG":
            tg_node = layers_latest["TG"].get(node_id, {})
            inputs = tg_node.get("inputs") if isinstance(tg_node.get("inputs"), dict) else {}
            for kg in inputs.get("kg_refs", []):
                if isinstance(kg, str):
                    kg_refs.append(kg)
            for rg in inputs.get("rg_refs", []):
                if isinstance(rg, str):
                    rg_refs.append(rg)

        link_suggestions.append(
            {
                "node_id": node_id,
                "layer": layer,
                "title": row.get("title", ""),
                "confidence": confidence,
            }
        )

    # Deduplicate preserving order.
    kg_refs = list(dict.fromkeys(kg_refs))
    rg_refs = list(dict.fromkeys(rg_refs))
    return kg_refs, rg_refs, link_suggestions


def measurable_criterion(text: str) -> bool:
    lowered = text.lower()

    for token in VAGUE_TOKENS:
        if token in lowered:
            # Allow if measurable qualifiers exist.
            if re.search(r"\b(file|path|json|yaml|exit code|status|count|<=|>=|==|must contain|exactly|regex|id\b|task\b|command\b)\b", lowered):
                return True
            return False

    if re.search(r"\b(file|path|json|yaml|exit code|status|count|<=|>=|==|must contain|exactly|regex|id\b|task\b|command\b)\b", lowered):
        return True

    if re.search(r"\d", lowered):
        return True

    return len(text.strip()) >= 12


def validate_dod(acceptance: List[str]) -> None:
    if not (2 <= len(acceptance) <= 4):
        raise PlanIntoTasksError("Each task must have 2-4 acceptance criteria")
    for criterion in acceptance:
        if not isinstance(criterion, str) or not criterion.strip():
            raise PlanIntoTasksError("Acceptance criteria must be non-empty strings")
        if not measurable_criterion(criterion):
            raise PlanIntoTasksError(f"Acceptance criterion is too vague: {criterion}")


def default_task_templates(intent: str, request_text: str) -> List[Dict[str, Any]]:
    short = " ".join(request_text.split())
    if len(short) > 90:
        short = short[:87].rstrip() + "..."

    templates = [
        {
            "phase": "setup",
            "title": "Clarify scope and constraints",
            "description": f"Extract concrete target, constraints, and success criteria from request: {short}",
            "priority": "high",
            "owner_role": "lead",
            "acceptance_criteria": [
                "A scoped objective statement is recorded in task description",
                "At least one explicit non-goal or constraint is documented in task inputs",
            ],
        },
        {
            "phase": "implement",
            "title": "Prepare execution plan and work breakdown",
            "description": "Translate scoped objective into actionable steps, owners, and dependency order.",
            "priority": "high",
            "owner_role": "build",
            "acceptance_criteria": [
                "Task decomposition contains 3-10 actionable items",
                "Dependency edges are acyclic and every dependency references a valid task id",
            ],
        },
        {
            "phase": "verify",
            "title": "Validate planned tasks and Definition of Done",
            "description": "Check all tasks for measurable DoD and graph-contract compliance before execution.",
            "priority": "medium",
            "owner_role": "qa",
            "acceptance_criteria": [
                "Each task has 2-4 measurable acceptance criteria",
                "Validation report states pass/fail for dependency and DoD quality checks",
            ],
        },
    ]

    if intent in {"execute", "debug"}:
        templates.append(
            {
                "phase": "deploy",
                "title": "Apply approved task plan to Task Graph",
                "description": "Optionally append planned tasks to TG with idempotency and index verification.",
                "priority": "medium",
                "owner_role": "ops",
                "acceptance_criteria": [
                    "Applied TG tasks include origin_plan_id and created_by metadata",
                    "TG index rebuild verification returns pass status",
                ],
            }
        )

    if any(keyword in request_text.lower() for keyword in ["doc", "docs", "documentation", "report", "summary", "док", "отчет", "сводка"]):
        templates.append(
            {
                "phase": "docs",
                "title": "Document plan decisions and handoff details",
                "description": "Capture assumptions, links, and completion criteria for non-technical handoff.",
                "priority": "low",
                "owner_role": "lead",
                "acceptance_criteria": [
                    "A plain-language summary is attached to plan output",
                    "Each task title has a one-sentence explanation understandable by non-developers",
                ],
            }
        )

    return templates


def normalize_owner(role: str, default_role: str) -> str:
    value = role.strip().lower() if isinstance(role, str) else default_role
    if value in OWNER_ROLES:
        return value
    return default_role


def normalize_priority(value: str, default_priority: str) -> str:
    raw = value.strip().lower() if isinstance(value, str) else default_priority
    if raw in PRIORITY_ORDER:
        return raw
    return default_priority


def build_suggested_tasks(
    templates: List[Dict[str, Any]],
    max_tasks: int,
    kg_refs: List[str],
    rg_refs: List[str],
    owner_default: str,
    priority_default: str,
) -> List[Dict[str, Any]]:
    if max_tasks < 3:
        raise PlanIntoTasksError("--max-tasks must be >= 3 for execution-ready decomposition")

    selected = templates[: max_tasks]
    if len(selected) < 3:
        raise PlanIntoTasksError("Insufficient templates to build minimum 3 tasks")

    tasks: List[Dict[str, Any]] = []

    for idx, template in enumerate(selected, start=1):
        temp_id = f"TEMP-{idx}"
        phase = template.get("phase")
        if phase not in PHASE_ORDER:
            raise PlanIntoTasksError(f"Unsupported phase: {phase}")

        acceptance = template.get("acceptance_criteria")
        if not isinstance(acceptance, list):
            raise PlanIntoTasksError("Task template acceptance_criteria must be a list")
        validate_dod(acceptance)

        deps: List[str] = []
        if idx > 1:
            deps.append(f"TEMP-{idx-1}")

        task = {
            "temp_id": temp_id,
            "type": "task",
            "title": template.get("title", f"Planned Task {idx}"),
            "description": template.get("description", ""),
            "phase": phase,
            "status": "ready",
            "priority": normalize_priority(template.get("priority", priority_default), priority_default),
            "owner_role": normalize_owner(template.get("owner_role", owner_default), owner_default),
            "dependencies": deps,
            "blocks": [],
            "acceptance_criteria": acceptance,
            "inputs": {
                "kg_refs": kg_refs[:8],
                "rg_refs": rg_refs[:8],
                "sources": [],
            },
            "outputs": {
                "artifacts": [],
                "kg_updates": [],
                "rg_updates": [],
            },
            "tags": ["planned", f"phase:{phase}"],
        }
        tasks.append(task)

    # Deterministic sort.
    tasks.sort(
        key=lambda t: (
            PHASE_ORDER.get(str(t.get("phase", "deploy")), 999),
            PRIORITY_ORDER.get(str(t.get("priority", "medium")), 999),
            str(t.get("title", "")),
            str(t.get("temp_id", "")),
        )
    )

    return tasks


def validate_dependency_graph(tasks: List[Dict[str, Any]]) -> None:
    ids = {str(t.get("temp_id", "")) for t in tasks}
    if "" in ids:
        raise PlanIntoTasksError("Each planned task requires non-empty temp_id")

    adjacency: Dict[str, List[str]] = {tid: [] for tid in ids}

    for task in tasks:
        task_id = str(task.get("temp_id", ""))
        deps = task.get("dependencies")
        if not isinstance(deps, list):
            raise PlanIntoTasksError(f"Task {task_id} dependencies must be list")

        for dep in deps:
            if not isinstance(dep, str):
                raise PlanIntoTasksError(f"Task {task_id} contains non-string dependency")
            if dep == task_id:
                raise PlanIntoTasksError(f"Task {task_id} cannot depend on itself")
            if dep.startswith("TEMP-") and dep not in ids:
                raise PlanIntoTasksError(f"Task {task_id} has unresolved temp dependency: {dep}")
            if dep.startswith("TEMP-"):
                adjacency[dep].append(task_id)

    visiting: Set[str] = set()
    visited: Set[str] = set()

    def dfs(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise PlanIntoTasksError("Cycle detected in planned dependencies")
        visiting.add(node)
        for nxt in adjacency.get(node, []):
            dfs(nxt)
        visiting.remove(node)
        visited.add(node)

    for node in ids:
        dfs(node)


def plan_files(paths: GraphPaths) -> List[Path]:
    if not paths.plan_history_dir.exists():
        return []
    return sorted(paths.plan_history_dir.glob("task_plan.*.json"))


def load_plan_seq_values(paths: GraphPaths) -> List[int]:
    values: List[int] = []
    for file_path in plan_files(paths):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        seq = payload.get("plan_seq")
        if isinstance(seq, int) and seq >= 1:
            values.append(seq)
    return values


def rotate_plan_history(paths: GraphPaths, history_limit: int, dry_run: bool) -> None:
    files = plan_files(paths)
    if len(files) <= history_limit:
        return
    removable = files[: len(files) - history_limit]
    for old_path in removable:
        if dry_run:
            print(f"[DRY-RUN] remove {old_path}")
        else:
            old_path.unlink(missing_ok=True)


def persist_plan_history(
    paths: GraphPaths,
    base_payload: Dict[str, Any],
    history_limit: int,
    dry_run: bool,
) -> Tuple[Path, str, int]:
    paths.plan_history_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(2):
        seq_values = load_plan_seq_values(paths)
        next_seq = (max(seq_values) + 1) if seq_values else 1
        plan_id = f"PLAN-{new_ulid()}"

        payload = copy.deepcopy(base_payload)
        payload["plan_id"] = plan_id
        payload["plan_seq"] = next_seq

        history_name = f"task_plan.{timestamp_slug()}.{plan_id}.json"
        history_path = paths.plan_history_dir / history_name

        if dry_run:
            print(f"[DRY-RUN] write {history_path}")
            return history_path, plan_id, next_seq

        if history_path.exists():
            continue

        write_json_atomic(history_path, payload, dry_run=False)

        written = read_json(history_path)
        if written is None or written.get("plan_id") != plan_id:
            history_path.unlink(missing_ok=True)
            raise PlanIntoTasksError("Failed to validate written task plan file")

        seq_after = load_plan_seq_values(paths)
        if seq_after.count(next_seq) == 1:
            rotate_plan_history(paths, history_limit=history_limit, dry_run=False)
            return history_path, plan_id, next_seq

        history_path.unlink(missing_ok=True)
        if attempt == 0:
            continue
        raise PlanIntoTasksError("plan_seq allocation conflict after retry")

    raise PlanIntoTasksError("Unable to allocate plan_seq")


def build_index_rows_for_layer(layer: str, raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest = resolve_latest_view(raw_records)
    rows: List[Dict[str, Any]] = []

    for node_id in sorted(latest.keys()):
        node = latest[node_id]
        full_node = copy.deepcopy(node)
        full_node.pop("_line_no", None)

        content_node = copy.deepcopy(full_node)
        content_node.pop("updated_at", None)
        if layer == "TG":
            content_node.pop("log", None)

        rows.append(
            {
                "id": full_node.get("id"),
                "title": full_node.get("title", ""),
                "tags": full_node.get("tags", []),
                "updated_at": full_node.get("updated_at", ""),
                "type": full_node.get("type", ""),
                "layer": layer,
                "status": full_node.get("status") if layer == "TG" else None,
                "content_hash": sha256_hex(content_node),
                "full_hash": sha256_hex(full_node),
            }
        )
    return rows


def rebuild_indexes(paths: GraphPaths, dry_run: bool) -> None:
    raw = read_raw_layers(paths)
    write_jsonl_atomic(paths.kg_index, build_index_rows_for_layer("KG", raw["KG"]), dry_run)
    write_jsonl_atomic(paths.rg_index, build_index_rows_for_layer("RG", raw["RG"]), dry_run)
    write_jsonl_atomic(paths.tg_index, build_index_rows_for_layer("TG", raw["TG"]), dry_run)


def verify_indexes(paths: GraphPaths) -> None:
    raw = read_raw_layers(paths)
    expected = {
        "KG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in build_index_rows_for_layer("KG", raw["KG"])],
        "RG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in build_index_rows_for_layer("RG", raw["RG"])],
        "TG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in build_index_rows_for_layer("TG", raw["TG"])],
    }
    actual = {
        "KG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in read_jsonl(paths.kg_index)],
        "RG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in read_jsonl(paths.rg_index)],
        "TG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in read_jsonl(paths.tg_index)],
    }

    for layer in ("KG", "RG", "TG"):
        if expected[layer] != actual[layer]:
            raise PlanIntoTasksError(f"Index verification failed for {layer}")


def max_evidence_confidence(node: Dict[str, Any]) -> float:
    evidence = node.get("evidence")
    if not isinstance(evidence, list):
        return 0.0
    best = 0.0
    for item in evidence:
        if not isinstance(item, dict):
            continue
        conf = item.get("confidence")
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            continue
        best = max(best, conf_f)
    return round(best, 4)


def rg_status(node: Dict[str, Any]) -> str:
    outcome = node.get("outcome")
    if isinstance(outcome, dict):
        raw = outcome.get("status")
        return str(raw) if raw is not None else ""
    return ""


def build_post_response_context_pack(
    paths: GraphPaths,
    commit_id: str,
    changed_node_ids: List[str],
    goal_hint: str,
) -> Dict[str, Any]:
    raw_layers = read_raw_layers(paths)
    layers_latest = {
        layer: resolve_latest_view(records)
        for layer, records in raw_layers.items()
    }

    tg_nodes = [copy.deepcopy(node) for node in layers_latest["TG"].values()]
    tg_nodes.sort(key=lambda node: str(node.get("updated_at", "")), reverse=True)
    active_tasks = [node for node in tg_nodes if str(node.get("status", "")) in {"ready", "in_progress", "blocked", "review"}]
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

    kg_nodes = [copy.deepcopy(node) for node in layers_latest["KG"].values()]
    kg_nodes.sort(key=lambda node: str(node.get("updated_at", "")), reverse=True)
    knowledge_snapshot = [
        {
            "id": node.get("id"),
            "title": node.get("title"),
            "summary": node.get("summary", ""),
            "confidence": max_evidence_confidence(node),
        }
        for node in kg_nodes[:30]
    ]

    rg_nodes = [copy.deepcopy(node) for node in layers_latest["RG"].values()]
    rg_nodes.sort(key=lambda node: str(node.get("updated_at", "")), reverse=True)

    changed_rg_ids = [node_id for node_id in changed_node_ids if node_id.startswith("RG-")]
    prioritized: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for node_id in changed_rg_ids:
        node = layers_latest["RG"].get(node_id)
        if node is None or node_id in seen:
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
            "status": rg_status(node),
        }
        for node in prioritized[:10]
    ]

    recent_deltas: List[Dict[str, str]] = []
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
            raise PlanIntoTasksError("Context history write validation failed")

    write_json_atomic(paths.context_latest, context_pack, dry_run)

    history_files = sorted(paths.context_packs_dir.glob("context_pack.*.json"))
    if len(history_files) > history_limit:
        to_remove = history_files[: len(history_files) - history_limit]
        for old in to_remove:
            if dry_run:
                print(f"[DRY-RUN] remove {old}")
            else:
                old.unlink(missing_ok=True)

    return str(history_path)


def append_post_response_commit(
    paths: GraphPaths,
    commit_id: str,
    plan_id: str,
    plan_seq: int,
    changed_node_ids: List[str],
    context_pack_path: str,
    request_id: Optional[str],
    source_request_hash: Optional[str],
    dry_run: bool,
) -> None:
    commit_row = {
        "commit_id": commit_id,
        "mode": "plan_into_tasks_apply",
        "plan_id": plan_id,
        "plan_seq": plan_seq,
        "changed_node_ids": changed_node_ids,
        "context_pack_path": context_pack_path,
        "created_at": now_iso_utc(),
        "event": "post_response_commit",
    }
    if request_id:
        commit_row["request_id"] = request_id
    if source_request_hash:
        commit_row["source_request_hash"] = source_request_hash

    append_jsonl_atomic(paths.post_response_commits, [commit_row], dry_run)


def existing_origin_plan_ids(tg_latest: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    plan_map: Dict[str, List[str]] = {}
    for node in tg_latest.values():
        plan_id = node.get("origin_plan_id")
        if not isinstance(plan_id, str) or not plan_id:
            continue
        plan_map.setdefault(plan_id, []).append(str(node.get("id", "")))
    return plan_map


def parse_request_text(args: argparse.Namespace) -> str:
    if args.request_text and args.request_file:
        raise PlanIntoTasksError("Provide either --request-text or --request-file, not both")
    if not args.request_text and not args.request_file:
        raise PlanIntoTasksError("One of --request-text or --request-file is required")

    if args.request_text:
        return args.request_text

    request_path = Path(args.request_file).expanduser().resolve()
    if not request_path.exists():
        raise PlanIntoTasksError(f"request file not found: {request_path}")
    return request_path.read_text(encoding="utf-8")


def apply_planned_tasks(
    paths: GraphPaths,
    planned_tasks: List[Dict[str, Any]],
    plan_id: str,
    plan_seq: int,
    request_id: Optional[str],
    source_request_hash: Optional[str],
    strict_links: bool,
    force: bool,
    dry_run: bool,
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    raw_layers = read_raw_layers(paths)
    tg_latest = resolve_latest_view(raw_layers["TG"])
    existing_map = existing_origin_plan_ids(tg_latest)
    warnings: List[str] = []

    if plan_id in existing_map and not force:
        raise PlanIntoTasksError(
            f"Plan {plan_id} already applied to TG tasks {existing_map[plan_id]}. Use --force true to allow duplicate apply."
        )
    if plan_id in existing_map and force:
        warnings.append(f"Duplicate apply allowed by --force for existing origin_plan_id {plan_id}")

    temp_to_tg: Dict[str, str] = {str(task["temp_id"]): f"TG-TASK-{new_ulid()}" for task in planned_tasks}
    existing_tg_ids = set(tg_latest.keys())

    now = now_iso_utc()
    rows_to_append: List[Dict[str, Any]] = []

    for task in planned_tasks:
        temp_id = str(task["temp_id"])
        real_id = temp_to_tg[temp_id]

        resolved_deps: List[str] = []
        for dep in task.get("dependencies", []):
            if not isinstance(dep, str):
                continue
            if dep.startswith("TEMP-"):
                mapped = temp_to_tg.get(dep)
                if mapped:
                    resolved_deps.append(mapped)
                else:
                    if strict_links:
                        raise PlanIntoTasksError(f"Unresolved temp dependency {dep} in {temp_id}")
                    warnings.append(f"Dropped unresolved temp dependency {dep} in {temp_id}")
            elif dep.startswith("TG-"):
                if dep in existing_tg_ids:
                    resolved_deps.append(dep)
                else:
                    if strict_links:
                        raise PlanIntoTasksError(f"External dependency not found in TG: {dep}")
                    warnings.append(f"Dropped external dependency not found in TG: {dep}")
            else:
                if strict_links:
                    raise PlanIntoTasksError(f"Unsupported dependency format in {temp_id}: {dep}")
                warnings.append(f"Dropped unsupported dependency format in {temp_id}: {dep}")

        row = {
            "id": real_id,
            "type": "task",
            "title": task.get("title", ""),
            "description": task.get("description", ""),
            "status": task.get("status", "ready"),
            "priority": task.get("priority", "medium"),
            "owner_role": task.get("owner_role", "lead"),
            "dependencies": resolved_deps,
            "blocks": task.get("blocks", []),
            "acceptance_criteria": task.get("acceptance_criteria", []),
            "inputs": task.get("inputs", {"kg_refs": [], "rg_refs": [], "sources": []}),
            "outputs": task.get("outputs", {"artifacts": [], "kg_updates": [], "rg_updates": []}),
            "log": [{"at": now, "event": "created", "by": "plan-into-tasks"}],
            "links": [],
            "tags": list(dict.fromkeys((task.get("tags") or []) + ["planned", f"phase:{task.get('phase', 'implement')}"])),
            "phase": task.get("phase", "implement"),
            "rev": 1,
            "created_at": now,
            "updated_at": now,
            "created_by": "plan-into-tasks",
            "origin": "request_task_plan",
            "origin_plan_id": plan_id,
        }

        if request_id:
            row["request_id"] = request_id
        if source_request_hash:
            row["source_request_hash"] = source_request_hash
        row["plan_seq"] = plan_seq

        rows_to_append.append(row)

    append_jsonl_atomic(paths.tg_nodes, rows_to_append, dry_run)
    rebuild_indexes(paths, dry_run)
    if not dry_run:
        verify_indexes(paths)

    changed_ids = [row["id"] for row in rows_to_append]
    commit_id: Optional[str] = None
    context_pack_path: Optional[str] = None

    if changed_ids:
        commit_id = f"COMMIT-{new_ulid()}"
        context_pack = build_post_response_context_pack(
            paths=paths,
            commit_id=commit_id,
            changed_node_ids=changed_ids,
            goal_hint=str(planned_tasks[0].get("title", "Apply planned tasks to TG")) if planned_tasks else "Apply planned tasks to TG",
        )
        context_pack_path = write_context_pack(
            paths=paths,
            context_pack=context_pack,
            history_limit=CONTEXT_PACK_HISTORY_LIMIT_DEFAULT,
            dry_run=dry_run,
        )
        try:
            append_post_response_commit(
                paths=paths,
                commit_id=commit_id,
                plan_id=plan_id,
                plan_seq=plan_seq,
                changed_node_ids=changed_ids,
                context_pack_path=context_pack_path,
                request_id=request_id,
                source_request_hash=source_request_hash,
                dry_run=dry_run,
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise PlanIntoTasksError(
                "Context pack updated but commit-row append failed. "
                "Re-run plan-into-tasks with --apply true to backfill post_response_commits. "
                f"Original error: {exc}"
            ) from exc

    return changed_ids, warnings, commit_id, context_pack_path


def maybe_refresh_request_context(
    project_root: Path,
    request_text: str,
    intent: str,
    dry_run: bool,
) -> None:
    if dry_run:
        print("[DRY-RUN] refresh request context via build-context-pack")
        return

    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    candidate_scripts = [
        project_root / ".generated/skills/build-context-pack/scripts/build_context_pack.py",
        codex_home / "skills/build-context-pack/scripts/build_context_pack.py",
    ]

    script = None
    for candidate in candidate_scripts:
        if candidate.exists():
            script = candidate
            break

    if script is None:
        raise PlanIntoTasksError("--refresh-request-context requested, but build_context_pack.py was not found")

    cmd = [
        str(script),
        "--project-root",
        str(project_root),
        "--request-text",
        request_text,
        "--intent",
        intent,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PlanIntoTasksError(
            "Failed to refresh request context via build-context-pack: "
            + (proc.stderr.strip() or proc.stdout.strip())
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan user request into TG tasks")
    parser.add_argument("--project-root", required=True, help="Project root path")
    parser.add_argument("--graphs-dir", default="graphs", help="Graph directory under project root")
    parser.add_argument("--request-text", default=None, help="Request text")
    parser.add_argument("--request-file", default=None, help="Path to file containing request text")
    parser.add_argument("--intent", choices=sorted(INTENTS), default="auto")
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--apply", type=parse_bool, default=False)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--owner-default", default="lead")
    parser.add_argument("--priority-default", default="medium")
    parser.add_argument("--strict-links", type=parse_bool, default=False)
    parser.add_argument("--stdout-json", type=parse_bool, default=False)
    parser.add_argument("--plan-history-limit", type=int, default=200)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--force", type=parse_bool, default=False)
    parser.add_argument("--refresh-request-context", type=parse_bool, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.exists() or not project_root.is_dir():
        print(f"[ERROR] invalid project root: {project_root}")
        return 2

    if args.max_tasks < 3 or args.max_tasks > 25:
        print("[ERROR] --max-tasks must be between 3 and 25")
        return 2

    if args.plan_history_limit < 1:
        print("[ERROR] --plan-history-limit must be >= 1")
        return 2

    owner_default = args.owner_default.strip().lower()
    if owner_default not in OWNER_ROLES:
        print(f"[ERROR] --owner-default must be one of {sorted(OWNER_ROLES)}")
        return 2

    priority_default = args.priority_default.strip().lower()
    if priority_default not in PRIORITY_ORDER:
        print(f"[ERROR] --priority-default must be one of {sorted(PRIORITY_ORDER)}")
        return 2

    try:
        request_text_raw = parse_request_text(args)
    except PlanIntoTasksError as exc:
        print(f"[ERROR] {exc}")
        return 2

    request_text = normalize_request_text(request_text_raw)
    source_request_hash = sha256_hex(request_text)

    paths = GraphPaths(project_root=project_root, graphs_dir_name=args.graphs_dir)

    try:
        raw_layers = read_raw_layers(paths)
        layers_latest = {
            layer: resolve_latest_view(records)
            for layer, records in raw_layers.items()
        }

        intent = classify_intent(request_text, args.intent)
        entities, _scores = extract_entities(request_text, layers_latest)
        kg_refs, rg_refs, link_suggestions = collect_auto_links(entities, layers_latest)

        templates = default_task_templates(intent, request_text)
        planned_tasks = build_suggested_tasks(
            templates=templates,
            max_tasks=args.max_tasks,
            kg_refs=kg_refs,
            rg_refs=rg_refs,
            owner_default=owner_default,
            priority_default=priority_default,
        )

        validate_dependency_graph(planned_tasks)

        warnings: List[str] = []

        payload: Dict[str, Any] = {
            "generated_at": now_iso_utc(),
            "intent": intent,
            "tasks": planned_tasks,
            "warnings": warnings,
            "link_suggestions": link_suggestions,
            "source_request_hash": source_request_hash,
        }

        if args.request_id:
            payload["request_id"] = args.request_id

        history_path, plan_id, plan_seq = persist_plan_history(
            paths=paths,
            base_payload=payload,
            history_limit=args.plan_history_limit,
            dry_run=args.dry_run,
        )

        payload["plan_id"] = plan_id
        payload["plan_seq"] = plan_seq

        changed_ids: List[str] = []
        apply_warnings: List[str] = []
        apply_commit_id: Optional[str] = None
        apply_context_pack_path: Optional[str] = None

        if args.apply:
            changed_ids, apply_warnings, apply_commit_id, apply_context_pack_path = apply_planned_tasks(
                paths=paths,
                planned_tasks=planned_tasks,
                plan_id=plan_id,
                plan_seq=plan_seq,
                request_id=args.request_id,
                source_request_hash=source_request_hash,
                strict_links=args.strict_links,
                force=args.force,
                dry_run=args.dry_run,
            )
            warnings.extend(apply_warnings)
            if args.refresh_request_context:
                maybe_refresh_request_context(
                    project_root=project_root,
                    request_text=request_text,
                    intent=intent,
                    dry_run=args.dry_run,
                )
            if apply_commit_id:
                payload["commit_id"] = apply_commit_id
            if apply_context_pack_path:
                payload["context_pack_path"] = apply_context_pack_path

        payload["warnings"] = warnings

        if args.stdout_json:
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(f"[OK] plan_id: {plan_id}")
            print(f"[OK] plan_seq: {plan_seq}")
            print(f"[OK] intent: {intent}")
            print(f"[OK] tasks_planned: {len(planned_tasks)}")
            print(f"[OK] plan_history: {history_path}")
            if args.apply:
                print(f"[OK] apply_changed_tasks: {len(changed_ids)}")
                if apply_commit_id:
                    print(f"[OK] commit_id: {apply_commit_id}")
                if apply_context_pack_path:
                    print(f"[OK] context_pack_path: {apply_context_pack_path}")
            if warnings:
                print(f"[WARN] warnings: {len(warnings)}")
                for warning in warnings:
                    print(f"[WARN] {warning}")

        return 0

    except PlanIntoTasksError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] unexpected failure: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
