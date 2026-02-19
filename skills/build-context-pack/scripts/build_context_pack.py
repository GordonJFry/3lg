#!/usr/bin/env python3
"""Build request-time context packs from KG/RG/TG graph layers."""

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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
INTERNAL_ID_PATTERN = re.compile(r"\b(?:KG|RG|TG)-[A-Z0-9-]{8,}\b")
WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")

ACTIVE_TG_STATUSES = {"in_progress", "ready", "blocked", "review"}
INTENTS = {"auto", "plan", "execute", "explain", "debug", "research"}
INTENT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "debug": (
        "bug",
        "error",
        "fail",
        "failing",
        "trace",
        "exception",
        "debug",
        "ошиб",
        "баг",
        "исключен",
        "трейс",
        "не работает",
        "сломал",
    ),
    "execute": (
        "implement",
        "apply",
        "run",
        "execute",
        "create",
        "update",
        "fix",
        "сделай",
        "выполни",
        "запусти",
        "создай",
        "обнови",
        "исправ",
        "реализ",
        "примени",
    ),
    "plan": (
        "plan",
        "roadmap",
        "design",
        "spec",
        "architecture",
        "план",
        "дорожн",
        "архитектур",
        "дизайн",
        "спек",
        "задач",
    ),
    "research": (
        "research",
        "investigate",
        "analyze",
        "compare",
        "explore",
        "исслед",
        "разбер",
        "проанализ",
        "сравни",
        "изучи",
    ),
}


class ContextPackError(RuntimeError):
    """Raised for deterministic context-pack errors."""


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
            raise ContextPackError(f"Invalid JSON in {path} line {idx}: {exc}") from exc
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
    append_rows = list(rows)
    if not append_rows:
        return
    existing = read_jsonl(path)
    existing.extend(append_rows)
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
    def commits(self) -> Path:
        return self.graphs_dir / "post_response_commits.jsonl"

    @property
    def request_latest(self) -> Path:
        return self.graphs_dir / "request_context.latest.json"

    @property
    def request_history_dir(self) -> Path:
        return self.graphs_dir / "request_context_packs"

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

    all_have_rev = all(value is not None for value in rev_values)

    best_line = -1
    best_record: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[Any, ...]] = None

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
            best_line = line_no
            best_record = rec

    if best_record is None:
        raise ContextPackError("Unable to resolve latest record")

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
    max_entities: int = 30,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    request_tokens = set(tokenize(request_text))
    explicit_ids = set(re.findall(r"\b(?:KG|RG|TG)-[A-Z0-9-]{8,}\b", request_text))

    entity_scores: Dict[str, float] = {}
    entity_rows: List[Dict[str, Any]] = []

    for layer, nodes in layers_latest.items():
        for node_id, node in nodes.items():
            score = 0.0
            reason = ""

            if node_id in explicit_ids:
                score = 1.0
                reason = "id_match"
            else:
                text_tokens = set(tokenize(_node_search_text(node)))
                if not text_tokens:
                    continue
                overlap = request_tokens.intersection(text_tokens)
                if overlap:
                    coverage = len(overlap) / max(1, len(request_tokens))
                    density = len(overlap) / max(1, len(text_tokens))
                    score = min(0.99, (0.8 * coverage) + (0.2 * density))
                    reason = "lexical_overlap"

            if score <= 0.0:
                continue

            entity_scores[node_id] = max(entity_scores.get(node_id, 0.0), score)
            entity_rows.append(
                {
                    "node_id": node_id,
                    "layer": layer,
                    "title": node.get("title", ""),
                    "confidence": round(score, 4),
                    "reason": reason,
                }
            )

    deduped: Dict[str, Dict[str, Any]] = {}
    for row in entity_rows:
        node_id = row["node_id"]
        current = deduped.get(node_id)
        if current is None or row["confidence"] > current["confidence"]:
            deduped[node_id] = row

    entities_sorted = sorted(
        deduped.values(),
        key=lambda item: (-item["confidence"], item["layer"], item["node_id"]),
    )[:max_entities]

    return entities_sorted, entity_scores


def recency_boost(updated_at: Any) -> int:
    updated = parse_iso(updated_at)
    if not updated:
        return 0
    age = datetime.now(timezone.utc) - updated
    if age <= timedelta(days=1):
        return 2
    if age <= timedelta(days=7):
        return 1
    return 0


def rg_status(node: Dict[str, Any]) -> str:
    outcome = node.get("outcome")
    if isinstance(outcome, dict):
        raw = outcome.get("status")
        return str(raw) if raw is not None else ""
    return ""


def get_rg_refs(node: Dict[str, Any]) -> Tuple[Set[str], Set[str], Set[str]]:
    kg_refs: Set[str] = set()
    tg_refs: Set[str] = set()
    artifacts: Set[str] = set()

    refs = node.get("refs")
    if isinstance(refs, dict):
        for key, target in (("kg_refs", kg_refs), ("tg_refs", tg_refs), ("artifacts", artifacts)):
            vals = refs.get(key)
            if isinstance(vals, list):
                for val in vals:
                    if isinstance(val, str):
                        target.add(val)

    links = node.get("links")
    if isinstance(links, list):
        for val in links:
            if not isinstance(val, str):
                continue
            if val.startswith("KG-"):
                kg_refs.add(val)
            elif val.startswith("TG-"):
                tg_refs.add(val)
            elif val.startswith("artifact://") or val.startswith("ART-"):
                artifacts.add(val)

    return kg_refs, tg_refs, artifacts


def get_tg_refs(node: Dict[str, Any]) -> Tuple[Set[str], Set[str], Set[str]]:
    kg_refs: Set[str] = set()
    rg_refs: Set[str] = set()
    deps: Set[str] = set()

    inputs = node.get("inputs")
    if isinstance(inputs, dict):
        for key, target in (("kg_refs", kg_refs), ("rg_refs", rg_refs)):
            vals = inputs.get(key)
            if isinstance(vals, list):
                for val in vals:
                    if isinstance(val, str):
                        target.add(val)

    for key in ("dependencies", "blocks"):
        vals = node.get(key)
        if isinstance(vals, list):
            for val in vals:
                if isinstance(val, str):
                    deps.add(val)

    return kg_refs, rg_refs, deps


def score_sort_key(score: int, updated_at: Any, node_id: str) -> Tuple[int, float, str]:
    updated = parse_iso(updated_at)
    updated_epoch = updated.timestamp() if updated else float("-inf")
    return (-score, -updated_epoch, node_id)


def select_tg_tasks(
    tg_latest: Dict[str, Dict[str, Any]],
    entity_scores: Dict[str, float],
    accepted_rg_tg_refs: Set[str],
    max_tasks: int,
) -> List[Dict[str, Any]]:
    active_nodes = [
        node for node in tg_latest.values() if str(node.get("status", "")) in ACTIVE_TG_STATUSES
    ]

    seed_rows: List[Tuple[int, Dict[str, Any]]] = []
    for node in active_nodes:
        node_id = str(node.get("id", ""))
        score = 0
        if entity_scores.get(node_id, 0.0) > 0.0:
            score += 5
        if node_id in accepted_rg_tg_refs:
            score += 3
        tags = node.get("tags")
        if isinstance(tags, list) and any(tag in {"constraint", "risk"} for tag in tags):
            score += 2
        score += recency_boost(node.get("updated_at"))
        seed_rows.append((score, node))

    seed_rows.sort(key=lambda row: score_sort_key(row[0], row[1].get("updated_at"), str(row[1].get("id", ""))))

    if not seed_rows:
        return []

    selected_ids: List[str] = []
    selected_set: Set[str] = set()

    def add_with_closure(root_id: str) -> None:
        queue = [root_id]
        while queue and len(selected_ids) < max_tasks:
            current = queue.pop(0)
            if current in selected_set:
                continue
            node = tg_latest.get(current)
            if node is None:
                continue
            selected_set.add(current)
            selected_ids.append(current)
            _, _, deps = get_tg_refs(node)
            queue.extend(dep for dep in deps if dep not in selected_set)

    for score, node in seed_rows:
        if len(selected_ids) >= max_tasks:
            break
        node_id = str(node.get("id", ""))
        if not node_id:
            continue
        if score <= 0 and selected_ids:
            continue
        add_with_closure(node_id)

    if not selected_ids:
        add_with_closure(str(seed_rows[0][1].get("id", "")))

    selected_nodes = [tg_latest[node_id] for node_id in selected_ids if node_id in tg_latest]

    scored_nodes: List[Tuple[int, Dict[str, Any]]] = []
    for node in selected_nodes:
        node_id = str(node.get("id", ""))
        score = 0
        if entity_scores.get(node_id, 0.0) > 0.0:
            score += 5
        if node_id in accepted_rg_tg_refs:
            score += 3
        score += recency_boost(node.get("updated_at"))
        scored_nodes.append((score, node))

    scored_nodes.sort(key=lambda row: score_sort_key(row[0], row[1].get("updated_at"), str(row[1].get("id", ""))))
    return [row[1] for row in scored_nodes[:max_tasks]]


def select_kg_nodes(
    kg_latest: Dict[str, Dict[str, Any]],
    selected_tg: List[Dict[str, Any]],
    selected_rg: List[Dict[str, Any]],
    entity_scores: Dict[str, float],
    max_kg: int,
) -> List[Dict[str, Any]]:
    selected_tg_kg_refs: Set[str] = set()
    for task in selected_tg:
        kg_refs, _, _ = get_tg_refs(task)
        selected_tg_kg_refs.update(kg_refs)

    accepted_rg_kg_refs: Set[str] = set()
    for decision in selected_rg:
        if rg_status(decision) != "accepted":
            continue
        kg_refs, _, _ = get_rg_refs(decision)
        accepted_rg_kg_refs.update(kg_refs)

    rows: List[Tuple[int, Dict[str, Any]]] = []
    for node in kg_latest.values():
        node_id = str(node.get("id", ""))
        score = 0
        if entity_scores.get(node_id, 0.0) > 0.0:
            score += 5
        if node_id in selected_tg_kg_refs:
            score += 4
        if node_id in accepted_rg_kg_refs:
            score += 3

        tags = node.get("tags") if isinstance(node.get("tags"), list) else []
        node_type = str(node.get("type", ""))
        if node_type in {"constraint", "risk"} or any(tag in {"constraint", "risk"} for tag in tags):
            score += 2

        score += recency_boost(node.get("updated_at"))

        if score > 0:
            rows.append((score, node))

    if not rows:
        for node in kg_latest.values():
            score = recency_boost(node.get("updated_at"))
            rows.append((score, node))

    rows.sort(key=lambda row: score_sort_key(row[0], row[1].get("updated_at"), str(row[1].get("id", ""))))
    return [row[1] for row in rows[:max_kg]]


def select_rg_nodes(
    rg_latest: Dict[str, Dict[str, Any]],
    selected_tg: List[Dict[str, Any]],
    selected_kg: List[Dict[str, Any]],
    entity_scores: Dict[str, float],
    max_rg: int,
) -> List[Dict[str, Any]]:
    selected_tg_ids = {str(node.get("id", "")) for node in selected_tg}
    selected_kg_ids = {str(node.get("id", "")) for node in selected_kg}

    selected_tg_rg_refs: Set[str] = set()
    for task in selected_tg:
        _, rg_refs, _ = get_tg_refs(task)
        selected_tg_rg_refs.update(rg_refs)

    candidate_ids: Set[str] = set()
    for node_id, node in rg_latest.items():
        status = rg_status(node)
        kg_refs, tg_refs, _ = get_rg_refs(node)
        if status in {"accepted", "proposed"} and (kg_refs.intersection(selected_kg_ids) or tg_refs.intersection(selected_tg_ids)):
            candidate_ids.add(node_id)
        if entity_scores.get(node_id, 0.0) > 0.0:
            candidate_ids.add(node_id)

    if not candidate_ids:
        for node_id, node in rg_latest.items():
            if rg_status(node) in {"accepted", "proposed"}:
                candidate_ids.add(node_id)

    related_ids: Set[str] = set(candidate_ids)
    for node_id in list(candidate_ids):
        node = rg_latest.get(node_id)
        if not node:
            continue
        outcome = node.get("outcome")
        if isinstance(outcome, dict):
            supersedes = outcome.get("supersedes")
            if isinstance(supersedes, list):
                for old_id in supersedes:
                    if isinstance(old_id, str) and old_id in rg_latest:
                        related_ids.add(old_id)

    rows: List[Tuple[int, Dict[str, Any]]] = []
    for node_id in related_ids:
        node = rg_latest.get(node_id)
        if node is None:
            continue
        score = 0
        if entity_scores.get(node_id, 0.0) > 0.0:
            score += 5
        if node_id in selected_tg_rg_refs:
            score += 4
        if rg_status(node) == "accepted":
            score += 3

        tags = node.get("tags") if isinstance(node.get("tags"), list) else []
        if any(tag in {"constraint", "risk"} for tag in tags):
            score += 2

        score += recency_boost(node.get("updated_at"))
        rows.append((score, node))

    rows.sort(key=lambda row: score_sort_key(row[0], row[1].get("updated_at"), str(row[1].get("id", ""))))
    return [row[1] for row in rows[:max_rg]]


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
    return best


def critical_info_missing(intent: str, entities: List[Dict[str, Any]], selected_tg: List[Dict[str, Any]], selected_kg: List[Dict[str, Any]]) -> bool:
    if not entities and intent in {"plan", "execute", "debug", "research"}:
        return True
    if not selected_tg and intent in {"execute", "debug"}:
        return True
    if not selected_kg and intent != "explain":
        return True
    return False


def build_clarify_suggestion(
    intent: str,
    entities: List[Dict[str, Any]],
    selected_tg: List[Dict[str, Any]],
    selected_kg: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not entities:
        topic = "target entities and scope"
    elif not selected_tg:
        topic = "execution steps and dependencies"
    elif not selected_kg:
        topic = "constraints and factual grounding"
    else:
        topic = "missing project context"

    return {
        "title": f"Clarify: {topic}",
        "description": "Collect missing context required to retrieve a precise minimal Context Pack.",
        "priority": "medium",
        "owner_role": "lead",
        "tags": ["clarify"],
        "acceptance_criteria": [
            "Clarifying details are explicitly provided by user",
            "Updated retrieval returns non-empty relevant KG/TG for intent",
        ],
        "inputs": {
            "kg_refs": [row["node_id"] for row in entities if row["node_id"].startswith("KG-")][:5],
            "rg_refs": [row["node_id"] for row in entities if row["node_id"].startswith("RG-")][:5],
            "sources": [],
        },
        "outputs": {
            "artifacts": [],
            "kg_updates": [],
            "rg_updates": [],
        },
    }


def build_recent_deltas(
    paths: GraphPaths,
    max_deltas: int,
    requested_commit_id: Optional[str],
    previous_request_seq: Optional[int],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    commits = read_jsonl(paths.commits)
    if not commits:
        if previous_request_seq is None:
            return [], None
        return [
            {
                "layer": "REQUEST",
                "id": f"SEQ-{previous_request_seq}",
                "change": "previous_request_anchor_no_commits",
            }
        ], None

    anchor_commit_id: Optional[str] = None
    anchor_index: Optional[int] = None

    if requested_commit_id:
        for idx, rec in enumerate(commits):
            if rec.get("commit_id") == requested_commit_id:
                anchor_index = idx
                anchor_commit_id = requested_commit_id
                break

    if anchor_index is None:
        last = commits[-1]
        anchor_index = len(commits) - 1
        anchor_commit_id = str(last.get("commit_id")) if last.get("commit_id") else None

    selected_commits = commits[anchor_index + 1 :] if anchor_index is not None else commits

    rows: List[Dict[str, Any]] = []
    for commit in selected_commits:
        commit_id = str(commit.get("commit_id", ""))
        mode = str(commit.get("mode", ""))
        changed = commit.get("changed_node_ids")
        if isinstance(changed, list):
            for node_id in changed:
                if not isinstance(node_id, str):
                    continue
                layer = node_id.split("-", 1)[0] if "-" in node_id else "UNKNOWN"
                rows.append(
                    {
                        "layer": layer,
                        "id": node_id,
                        "change": f"{mode}:changed",
                        "commit_id": commit_id,
                    }
                )

    if not rows and previous_request_seq is not None:
        rows.append(
            {
                "layer": "REQUEST",
                "id": f"SEQ-{previous_request_seq}",
                "change": "previous_request_anchor",
                "commit_id": anchor_commit_id,
            }
        )

    return rows[:max_deltas], anchor_commit_id


def history_request_context_files(paths: GraphPaths) -> List[Path]:
    if not paths.request_history_dir.exists():
        return []
    files = sorted(paths.request_history_dir.glob("request_context.*.json"))
    return files


def load_history_seq_values(paths: GraphPaths) -> List[int]:
    values: List[int] = []
    for hist_file in history_request_context_files(paths):
        try:
            payload = json.loads(hist_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        seq = payload.get("request_seq")
        if isinstance(seq, int) and seq >= 1:
            values.append(seq)
    return values


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
    kg_rows = build_index_rows_for_layer("KG", raw["KG"])
    rg_rows = build_index_rows_for_layer("RG", raw["RG"])
    tg_rows = build_index_rows_for_layer("TG", raw["TG"])

    write_jsonl_atomic(paths.kg_index, kg_rows, dry_run)
    write_jsonl_atomic(paths.rg_index, rg_rows, dry_run)
    write_jsonl_atomic(paths.tg_index, tg_rows, dry_run)


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
            raise ContextPackError(f"Index verification failed for {layer}")


def apply_clarify_task(paths: GraphPaths, suggestion: Dict[str, Any], dry_run: bool) -> Optional[str]:
    task_id = f"TG-TASK-{new_ulid()}"
    now = now_iso_utc()

    task = {
        "id": task_id,
        "type": "task",
        "title": suggestion.get("title", "Clarify: missing context"),
        "description": suggestion.get("description", "Clarify missing context"),
        "status": "ready",
        "priority": suggestion.get("priority", "medium"),
        "owner_role": suggestion.get("owner_role", "lead"),
        "dependencies": [],
        "blocks": [],
        "acceptance_criteria": suggestion.get("acceptance_criteria", []),
        "inputs": suggestion.get("inputs", {"kg_refs": [], "rg_refs": [], "sources": []}),
        "outputs": suggestion.get("outputs", {"artifacts": [], "kg_updates": [], "rg_updates": []}),
        "log": [{"at": now, "event": "created", "by": "build-context-pack"}],
        "links": [],
        "tags": list(dict.fromkeys((suggestion.get("tags") or []) + ["clarify"])),
        "rev": 1,
        "created_at": now,
        "updated_at": now,
        "created_by": "build-context-pack",
        "origin": "clarify_suggestion",
    }

    append_jsonl_atomic(paths.tg_nodes, [task], dry_run)
    rebuild_indexes(paths, dry_run)
    if not dry_run:
        verify_indexes(paths)
    return task_id


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

    tg_nodes = [copy.deepcopy(node) for node in tg_latest.values()]
    tg_nodes.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    active_tasks = [node for node in tg_nodes if str(node.get("status", "")) in ACTIVE_TG_STATUSES]
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

    kg_nodes = [copy.deepcopy(node) for node in kg_latest.values()]
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

    rg_nodes = [copy.deepcopy(node) for node in rg_latest.values()]
    rg_nodes.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)

    changed_rg_ids = [node_id for node_id in changed_node_ids if node_id.startswith("RG-")]
    prioritized: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for node_id in changed_rg_ids:
        node = rg_latest.get(node_id)
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
            raise ContextPackError("Context history write validation failed")

    write_json_atomic(paths.context_latest, context_pack, dry_run)

    history_files = sorted(paths.context_packs_dir.glob("context_pack.*.json"))
    if len(history_files) > history_limit:
        for old in history_files[: len(history_files) - history_limit]:
            if dry_run:
                print(f"[DRY-RUN] remove {old}")
            else:
                old.unlink(missing_ok=True)

    return str(history_path)


def append_post_response_commit(
    paths: GraphPaths,
    commit_id: str,
    changed_node_ids: List[str],
    context_pack_path: str,
    request_id: Optional[str],
    request_seq: Optional[int],
    dry_run: bool,
) -> None:
    row: Dict[str, Any] = {
        "commit_id": commit_id,
        "mode": "build_context_pack_apply_clarify",
        "changed_node_ids": changed_node_ids,
        "context_pack_path": context_pack_path,
        "created_at": now_iso_utc(),
        "event": "post_response_commit",
    }
    if request_id:
        row["request_id"] = request_id
    if isinstance(request_seq, int) and request_seq >= 1:
        row["request_seq"] = request_seq
    append_jsonl_atomic(paths.commits, [row], dry_run)


def compact_request(request_text: str, max_chars: int = 140) -> str:
    collapsed = " ".join(request_text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def build_context_pack(
    request_text: str,
    intent: str,
    entities: List[Dict[str, Any]],
    selected_tg: List[Dict[str, Any]],
    selected_kg: List[Dict[str, Any]],
    selected_rg: List[Dict[str, Any]],
    recent_deltas: List[Dict[str, Any]],
) -> Dict[str, Any]:
    goal = selected_tg[0].get("title") if selected_tg else compact_request(request_text)

    task_snapshot = [
        {
            "id": node.get("id"),
            "title": node.get("title"),
            "status": node.get("status"),
            "deps": node.get("dependencies", []),
            "acceptance_criteria": node.get("acceptance_criteria", []),
        }
        for node in selected_tg
    ]

    knowledge_snapshot = [
        {
            "id": node.get("id"),
            "title": node.get("title"),
            "summary": node.get("summary", ""),
            "confidence": round(max_evidence_confidence(node), 4),
        }
        for node in selected_kg
    ]

    decision_snapshot = [
        {
            "id": node.get("id"),
            "title": node.get("title"),
            "decision": node.get("decision", ""),
            "status": rg_status(node),
        }
        for node in selected_rg
    ]

    return {
        "generated_at": now_iso_utc(),
        "intent": intent,
        "entities": entities,
        "goal": str(goal),
        "active_tasks": [node.get("id") for node in selected_tg],
        "task_snapshot": task_snapshot,
        "knowledge_snapshot": knowledge_snapshot,
        "decision_snapshot": decision_snapshot,
        "recent_deltas": recent_deltas,
    }


def rotate_history(paths: GraphPaths, history_limit: int, dry_run: bool) -> None:
    files = history_request_context_files(paths)
    if len(files) <= history_limit:
        return
    to_remove = files[: len(files) - history_limit]
    for file_path in to_remove:
        if dry_run:
            print(f"[DRY-RUN] remove {file_path}")
        else:
            file_path.unlink(missing_ok=True)


def persist_pack_with_request_seq(
    paths: GraphPaths,
    base_pack: Dict[str, Any],
    history_limit: int,
    write_latest: bool,
    dry_run: bool,
) -> Tuple[Path, str, int]:
    paths.request_history_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(2):
        seq_values = load_history_seq_values(paths)
        next_seq = (max(seq_values) + 1) if seq_values else 1
        request_id = f"REQ-{new_ulid()}"

        pack = copy.deepcopy(base_pack)
        pack["request_id"] = request_id
        pack["request_seq"] = next_seq

        history_name = f"request_context.{timestamp_slug()}.{request_id}.json"
        history_path = paths.request_history_dir / history_name

        if dry_run:
            print(f"[DRY-RUN] write {history_path}")
            if write_latest:
                print(f"[DRY-RUN] write {paths.request_latest}")
            return history_path, request_id, next_seq

        if history_path.exists():
            # Extremely unlikely with ULID, but deterministic guard.
            continue

        write_json_atomic(history_path, pack, dry_run=False)

        written = read_json(history_path)
        if written is None or written.get("request_id") != request_id:
            history_path.unlink(missing_ok=True)
            raise ContextPackError("Failed to validate written request context history file")

        seq_after = load_history_seq_values(paths)
        if seq_after.count(next_seq) == 1:
            if write_latest:
                write_json_atomic(paths.request_latest, pack, dry_run=False)
            rotate_history(paths, history_limit=history_limit, dry_run=False)
            return history_path, request_id, next_seq

        history_path.unlink(missing_ok=True)
        if attempt == 0:
            continue
        raise ContextPackError("request_seq allocation conflict after retry")

    raise ContextPackError("Unable to allocate request_seq")


def load_previous_request_seq(paths: GraphPaths) -> Optional[int]:
    latest = read_json(paths.request_latest)
    if latest is None:
        return None
    seq = latest.get("request_seq")
    return seq if isinstance(seq, int) and seq >= 1 else None


def parse_request_text(args: argparse.Namespace) -> str:
    if args.request_text and args.request_file:
        raise ContextPackError("Provide either --request-text or --request-file, not both")
    if not args.request_text and not args.request_file:
        raise ContextPackError("One of --request-text or --request-file is required")

    if args.request_text:
        return args.request_text

    request_path = Path(args.request_file).expanduser().resolve()
    if not request_path.exists():
        raise ContextPackError(f"request file not found: {request_path}")
    return request_path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build request-time context packs from KG/RG/TG")
    parser.add_argument("--project-root", required=True, help="Project root path")
    parser.add_argument("--graphs-dir", default="graphs", help="Graph directory under project root")
    parser.add_argument("--request-text", default=None, help="Raw user request text")
    parser.add_argument("--request-file", default=None, help="File containing user request text")
    parser.add_argument("--intent", choices=sorted(INTENTS), default="auto")

    parser.add_argument("--max-tasks", type=int, default=15)
    parser.add_argument("--max-kg", type=int, default=30)
    parser.add_argument("--max-rg", type=int, default=10)
    parser.add_argument("--max-deltas", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=200)

    parser.add_argument("--write-latest", type=parse_bool, default=True)
    parser.add_argument("--stdout-json", type=parse_bool, default=False)
    parser.add_argument("--suggest-clarify", type=parse_bool, default=True)
    parser.add_argument("--apply-clarify-task", type=parse_bool, default=False)
    parser.add_argument("--commit-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.exists() or not project_root.is_dir():
        print(f"[ERROR] invalid project root: {project_root}")
        return 2

    if args.max_tasks < 1 or args.max_kg < 1 or args.max_rg < 1 or args.max_deltas < 1 or args.history_limit < 1:
        print("[ERROR] limits must be >= 1")
        return 2

    try:
        request_text = parse_request_text(args)
    except ContextPackError as exc:
        print(f"[ERROR] {exc}")
        return 2

    paths = GraphPaths(project_root=project_root, graphs_dir_name=args.graphs_dir)

    try:
        raw_layers = read_raw_layers(paths)
        layers_latest = {
            layer: resolve_latest_view(records)
            for layer, records in raw_layers.items()
        }

        intent = classify_intent(request_text, args.intent)
        entities, entity_scores = extract_entities(request_text, layers_latest)

        accepted_rg_tg_refs: Set[str] = set()
        for rg_node in layers_latest["RG"].values():
            if rg_status(rg_node) != "accepted":
                continue
            _kg_refs, tg_refs, _artifacts = get_rg_refs(rg_node)
            accepted_rg_tg_refs.update(tg_refs)

        selected_tg = select_tg_tasks(
            tg_latest=layers_latest["TG"],
            entity_scores=entity_scores,
            accepted_rg_tg_refs=accepted_rg_tg_refs,
            max_tasks=args.max_tasks,
        )

        # First-pass RG based on TG/KG may need placeholder KG list.
        selected_rg_initial = select_rg_nodes(
            rg_latest=layers_latest["RG"],
            selected_tg=selected_tg,
            selected_kg=[],
            entity_scores=entity_scores,
            max_rg=args.max_rg,
        )

        selected_kg = select_kg_nodes(
            kg_latest=layers_latest["KG"],
            selected_tg=selected_tg,
            selected_rg=selected_rg_initial,
            entity_scores=entity_scores,
            max_kg=args.max_kg,
        )

        selected_rg = select_rg_nodes(
            rg_latest=layers_latest["RG"],
            selected_tg=selected_tg,
            selected_kg=selected_kg,
            entity_scores=entity_scores,
            max_rg=args.max_rg,
        )

        previous_seq = load_previous_request_seq(paths)
        recent_deltas, anchored_commit_id = build_recent_deltas(
            paths=paths,
            max_deltas=args.max_deltas,
            requested_commit_id=args.commit_id,
            previous_request_seq=previous_seq,
        )

        pack = build_context_pack(
            request_text=request_text,
            intent=intent,
            entities=entities,
            selected_tg=selected_tg,
            selected_kg=selected_kg,
            selected_rg=selected_rg,
            recent_deltas=recent_deltas,
        )

        if anchored_commit_id:
            pack["commit_id"] = anchored_commit_id

        should_suggest = args.suggest_clarify and intent != "explain"
        missing = critical_info_missing(intent, entities, selected_tg, selected_kg)

        clarify_applied_task_id: Optional[str] = None
        clarify_commit_id: Optional[str] = None
        clarify_context_pack_path: Optional[str] = None
        if should_suggest and missing:
            suggestion = build_clarify_suggestion(intent, entities, selected_tg, selected_kg)
            pack["clarify_suggestion"] = suggestion
            if args.apply_clarify_task:
                clarify_applied_task_id = apply_clarify_task(paths, suggestion, dry_run=args.dry_run)
                if clarify_applied_task_id:
                    pack["clarify_applied_task_id"] = clarify_applied_task_id
                    clarify_commit_id = f"COMMIT-{new_ulid()}"
                    pack["commit_id"] = clarify_commit_id
                    if len(pack["recent_deltas"]) < args.max_deltas:
                        pack["recent_deltas"].append(
                            {
                                "layer": "TG",
                                "id": clarify_applied_task_id,
                                "change": "clarify_task_created",
                            }
                        )

        history_path, request_id, request_seq = persist_pack_with_request_seq(
            paths=paths,
            base_pack=pack,
            history_limit=args.history_limit,
            write_latest=args.write_latest,
            dry_run=args.dry_run,
        )

        pack["request_id"] = request_id
        pack["request_seq"] = request_seq

        if clarify_applied_task_id and clarify_commit_id:
            changed_ids = [clarify_applied_task_id]
            context_pack = build_post_response_context_pack(
                paths=paths,
                commit_id=clarify_commit_id,
                changed_node_ids=changed_ids,
                goal_hint=str(pack.get("goal", "Clarify missing context")),
            )
            clarify_context_pack_path = write_context_pack(
                paths=paths,
                context_pack=context_pack,
                history_limit=args.history_limit,
                dry_run=args.dry_run,
            )
            try:
                append_post_response_commit(
                    paths=paths,
                    commit_id=clarify_commit_id,
                    changed_node_ids=changed_ids,
                    context_pack_path=clarify_context_pack_path,
                    request_id=request_id,
                    request_seq=request_seq,
                    dry_run=args.dry_run,
                )
            except Exception as exc:  # pylint: disable=broad-except
                raise ContextPackError(
                    "Post-response context updated but commit-row append failed. "
                    "Re-run build-context-pack with --apply-clarify-task true to backfill post_response_commits. "
                    f"Original error: {exc}"
                ) from exc

            pack["context_pack_path"] = clarify_context_pack_path

        if args.stdout_json:
            print(json.dumps(pack, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(f"[OK] request_id: {request_id}")
            print(f"[OK] request_seq: {request_seq}")
            print(f"[OK] intent: {intent}")
            print(
                "[OK] snapshot sizes: "
                f"tasks={len(pack['task_snapshot'])} "
                f"kg={len(pack['knowledge_snapshot'])} "
                f"rg={len(pack['decision_snapshot'])} "
                f"deltas={len(pack['recent_deltas'])}"
            )
            print(f"[OK] history: {history_path}")
            if args.write_latest:
                print(f"[OK] latest: {paths.request_latest}")
            if "clarify_suggestion" in pack:
                print("[OK] clarify_suggestion: generated")
            if clarify_applied_task_id:
                print(f"[OK] clarify_task_id: {clarify_applied_task_id}")
            if clarify_commit_id:
                print(f"[OK] commit_id: {clarify_commit_id}")
            if clarify_context_pack_path:
                print(f"[OK] context_pack_path: {clarify_context_pack_path}")

        return 0

    except ContextPackError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] unexpected failure: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
