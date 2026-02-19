#!/usr/bin/env python3
"""Commit conversation-derived knowledge into KG with deterministic upserts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
INTERNAL_ID_PATTERN = re.compile(r"\b(?:KG|RG|TG)-[A-Z0-9-]{8,}\b")
KG_ENTITY_ID_PATTERN = re.compile(r"\bKG-ENTITY-[A-Z0-9-]{8,}\b")
NON_ALNUM_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)

TYPE_ORDER = {
    "constraint": 0,
    "definition": 1,
    "fact": 2,
    "assumption": 3,
}

ACTIVE_TG_STATUSES = {"ready", "in_progress", "blocked", "review"}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}

TOPIC_NEUTRAL_TOKENS = {
    "not",
    "never",
    "no",
    "must",
    "must_not",
    "should",
    "should_not",
    "always",
    "only",
}

ASSUMPTION_MARKERS = [
    "might",
    "may",
    "could",
    "likely",
    "probably",
    "assume",
    "assuming",
    "uncertain",
    "appears",
    "seems",
    "if ",
    "возможно",
    "может",
    "скорее всего",
    "предполож",
    "не уверен",
    "если ",
]

DEFINITION_MARKERS = [
    " is ",
    " means ",
    " refers to ",
    " defined as ",
    " definition ",
    " это ",
    " означает ",
    " определяется как ",
    " определение ",
]

CONSTRAINT_STRONG_MARKERS = [
    "must",
    "must not",
    "never",
    "do not",
    "cannot",
    "can't",
    "always",
    "only",
    "должен",
    "должна",
    "должны",
    "нельзя",
    "запрещено",
    "никогда",
    "всегда",
    "только",
]

CONSTRAINT_WEAK_MARKERS = [
    "should",
    "should not",
    "следует",
    "не следует",
]

NEGATIVE_MARKERS = [
    "must not",
    "do not",
    "never",
    "cannot",
    "can't",
    "should not",
    "no ",
    " not ",
    " нельзя",
    " не ",
    "никогда",
    "запрещено",
]

INCOMPATIBLE_MODAL_PAIRS = {
    ("always", "never"),
    ("must", "must_not"),
    ("should", "should_not"),
}


class CommitKnowledgeError(RuntimeError):
    """Raised for deterministic commit-knowledge failures."""


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


def sha256_hex_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_raw_for_hash(text: str) -> str:
    return normalize_newlines(text).strip()


def normalize_for_dedupe(text: str) -> str:
    cleaned = normalize_newlines(text).strip().lower()
    cleaned = cleaned.replace("_", " ")
    cleaned = NON_ALNUM_RE.sub(" ", cleaned)
    cleaned = SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def normalize_key_text(text: str) -> str:
    normalized = normalize_for_dedupe(text)
    if normalized:
        return normalized
    raw = normalize_raw_for_hash(text)
    if not raw:
        return "empty"
    return f"item_{sha256_hex_text(raw)[:12]}"


def normalize_scope_fragment(text: str) -> str:
    return normalize_key_text(text).replace(" ", "_")


def compact_whitespace(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


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
            raise CommitKnowledgeError(f"Invalid JSONL in {path} line {idx}: {exc}") from exc
        if not isinstance(payload, dict):
            raise CommitKnowledgeError(f"Invalid JSONL object in {path} line {idx}")
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


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]], dry_run: bool) -> None:
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
    content = ("\n".join(lines) + "\n") if lines else ""
    write_text_atomic(path, content, dry_run)


def append_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]], dry_run: bool) -> None:
    append_rows = list(rows)
    if not append_rows:
        return
    existing = read_jsonl(path)
    existing.extend(append_rows)
    write_jsonl_atomic(path, existing, dry_run)


def unique_keep_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


@dataclass
class GraphPaths:
    project_root: Path
    graphs_dir_name: str

    @property
    def graphs_dir(self) -> Path:
        return (self.project_root / self.graphs_dir_name).resolve()

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
    def meta(self) -> Path:
        return self.graphs_dir / "meta.json"

    @property
    def conflicts(self) -> Path:
        return self.graphs_dir / "conflicts.jsonl"

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
    def knowledge_commits_dir(self) -> Path:
        return self.graphs_dir / "knowledge_commits"


def read_raw_layers(paths: GraphPaths) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "KG": read_jsonl(paths.kg_nodes),
        "RG": read_jsonl(paths.rg_nodes),
        "TG": read_jsonl(paths.tg_nodes),
    }


def _resolve_latest_for_id(records: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    rev_values: List[Optional[int]] = []
    for _, record in records:
        rev = record.get("rev")
        rev_values.append(rev if isinstance(rev, int) and rev >= 1 else None)

    all_have_rev = all(value is not None for value in rev_values)

    best_key: Optional[Tuple[Any, ...]] = None
    best_line = -1
    best_record: Optional[Dict[str, Any]] = None

    for line_no, record in records:
        updated = parse_iso(record.get("updated_at"))
        updated_epoch = updated.timestamp() if updated else float("-inf")
        rev = int(record.get("rev", 0)) if isinstance(record.get("rev"), int) else 0

        if all_have_rev:
            key = (rev, updated_epoch, line_no)
        else:
            key = (updated_epoch, line_no)

        if best_key is None or key > best_key:
            best_key = key
            best_line = line_no
            best_record = record

    if best_record is None:
        raise CommitKnowledgeError("Unable to resolve latest record for id")

    resolved = copy.deepcopy(best_record)
    resolved["_line_no"] = best_line
    return resolved


def resolve_latest_view(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, record in enumerate(records, start=1):
        node_id = record.get("id")
        if not isinstance(node_id, str) or not node_id:
            continue
        grouped.setdefault(node_id, []).append((idx, record))

    latest: Dict[str, Dict[str, Any]] = {}
    for node_id, grouped_records in grouped.items():
        latest[node_id] = _resolve_latest_for_id(grouped_records)
    return latest


def strip_internal_fields(node: Dict[str, Any]) -> Dict[str, Any]:
    clean = copy.deepcopy(node)
    clean.pop("_line_no", None)
    return clean


def extract_project_name(paths: GraphPaths) -> str:
    meta = read_json(paths.meta)
    if isinstance(meta, dict):
        name = meta.get("project_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return paths.project_root.name


def tokenize(text: str) -> List[str]:
    return [token for token in WORD_RE.findall(normalize_for_dedupe(text)) if token]


def resolve_entity_scope(conversation_text: str, kg_latest: Dict[str, Dict[str, Any]]) -> Optional[str]:
    explicit_ids = sorted(set(KG_ENTITY_ID_PATTERN.findall(conversation_text)))
    for entity_id in explicit_ids:
        node = kg_latest.get(entity_id)
        if node and str(node.get("type", "")) == "entity":
            return entity_id

    request_tokens = set(tokenize(conversation_text))
    if not request_tokens:
        return None

    scored: List[Tuple[int, str]] = []
    for node_id, node in kg_latest.items():
        if str(node.get("type", "")) != "entity":
            continue
        title = str(node.get("title", ""))
        title_tokens = set(tokenize(title))
        overlap = request_tokens.intersection(title_tokens)
        if len(overlap) >= 2:
            scored.append((len(overlap), node_id))

    if not scored:
        return None

    scored.sort(key=lambda row: (-row[0], row[1]))
    return scored[0][1]


def derive_scope(project_name: str, entity_id: Optional[str]) -> str:
    base = f"project::{normalize_scope_fragment(project_name)}"
    if entity_id:
        return f"{base}::entity::{entity_id}"
    return base


@dataclass
class SourceContext:
    source_key: str
    source_kind: str
    source_title: str
    source_path: Optional[str]
    source_digest: str
    source_request_hash_raw: str
    source_request_hash_norm: str


def build_source_context(args: argparse.Namespace, raw_input_text: str) -> SourceContext:
    raw_normalized = normalize_raw_for_hash(raw_input_text)
    norm_normalized = normalize_for_dedupe(raw_input_text)

    source_request_hash_raw = sha256_hex_text(raw_normalized)
    source_request_hash_norm = sha256_hex_text(norm_normalized)

    source_kind = "conversation_text"
    source_title = args.source_title.strip() if isinstance(args.source_title, str) and args.source_title.strip() else "Conversation input"
    source_path: Optional[str] = None

    if args.conversation_file:
        source_kind = "conversation_file"
        source_path = str(Path(args.conversation_file).expanduser().resolve())
        if not (isinstance(args.source_title, str) and args.source_title.strip()):
            source_title = Path(source_path).name

    if args.source_key:
        source_key = args.source_key.strip()
    elif args.request_id:
        source_key = f"request::{args.request_id.strip()}"
    elif source_path:
        source_key = f"file::{source_path}"
    else:
        source_key = "conversation::default"

    return SourceContext(
        source_key=source_key,
        source_kind=source_kind,
        source_title=source_title,
        source_path=source_path,
        source_digest=source_request_hash_raw,
        source_request_hash_raw=source_request_hash_raw,
        source_request_hash_norm=source_request_hash_norm,
    )


def segment_statements(text: str) -> List[Dict[str, Any]]:
    normalized = normalize_newlines(text)
    lines = normalized.splitlines(keepends=True)

    statements: List[Dict[str, Any]] = []
    offset = 0

    for line_no, line in enumerate(lines, start=1):
        line_without_newline = line.rstrip("\n")
        stripped = line_without_newline.strip()

        if not stripped:
            offset += len(line)
            continue

        if stripped in {"---", "```"}:
            offset += len(line)
            continue

        if not any(ch.isalnum() for ch in stripped):
            offset += len(line)
            continue

        start_in_line = line_without_newline.find(stripped)
        start_char = offset + max(start_in_line, 0)
        end_char = start_char + len(stripped)

        statements.append(
            {
                "text": stripped,
                "span": {
                    "start_line": line_no,
                    "end_line": line_no,
                    "start_char": start_char,
                    "end_char": end_char,
                },
                "excerpt": truncate_text(stripped, 200),
            }
        )

        offset += len(line)

    if not normalized.strip():
        return []

    if not statements:
        stripped = normalized.strip()
        statements.append(
            {
                "text": stripped,
                "span": {
                    "start_line": 1,
                    "end_line": max(1, normalized.count("\n") + 1),
                    "start_char": 0,
                    "end_char": len(stripped),
                },
                "excerpt": truncate_text(stripped, 200),
            }
        )

    return statements


def clean_statement_prefix(text: str) -> str:
    cleaned = re.sub(r"^\s*[-*]\s+", "", text)
    cleaned = re.sub(r"^\s*\d+[.)]\s+", "", cleaned)
    return cleaned.strip()


def detect_claim_type(statement_text: str) -> Tuple[str, float, str, str]:
    lower = f" {statement_text.lower()} "

    has_assumption = any(marker in lower for marker in ASSUMPTION_MARKERS)
    has_definition = any(marker in lower for marker in DEFINITION_MARKERS)
    has_constraint_strong = any(marker in lower for marker in CONSTRAINT_STRONG_MARKERS)
    has_constraint_weak = any(marker in lower for marker in CONSTRAINT_WEAK_MARKERS)

    claim_type = "fact"
    confidence = 0.72
    polarity = "positive"
    modal = "none"

    if has_assumption:
        claim_type = "assumption"
        confidence = 0.50
        modal = "uncertain"

    if has_definition and claim_type != "assumption":
        claim_type = "definition"
        confidence = 0.84
        modal = "none"

    if has_constraint_strong and claim_type != "assumption":
        claim_type = "constraint"
        confidence = 0.92
        if "must not" in lower or "do not" in lower or "cannot" in lower or "can't" in lower:
            modal = "must_not"
        elif "never" in lower:
            modal = "never"
        elif "always" in lower:
            modal = "always"
        else:
            modal = "must"

    elif has_constraint_weak and claim_type not in {"assumption", "constraint"}:
        claim_type = "constraint"
        confidence = 0.82
        modal = "should_not" if "should not" in lower else "should"

    if "?" in statement_text:
        confidence = max(0.35, confidence - 0.15)

    if any(marker in lower for marker in NEGATIVE_MARKERS):
        polarity = "negative"

    return claim_type, round(confidence, 4), polarity, modal


def topic_key_from_title(title: str) -> str:
    tokens = [
        tok
        for tok in normalize_key_text(title).split(" ")
        if tok and tok not in STOPWORDS and tok not in TOPIC_NEUTRAL_TOKENS
    ]
    if not tokens:
        tokens = normalize_key_text(title).split(" ")
    return " ".join(tokens[:12]).strip()


def make_knowledge_key(claim_type: str, scope: str, title: str) -> str:
    normalized_title = normalize_key_text(title)
    if claim_type == "definition":
        return f"def::{scope}::{normalized_title}"
    if claim_type == "constraint":
        return f"constraint::{scope}::{normalized_title}"
    if claim_type == "assumption":
        return f"assump::{scope}::{normalized_title}"
    return f"fact::{scope}::{normalized_title}"


def candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[float, int, str, str]:
    claim_type = str(candidate.get("type", "fact"))
    return (
        -float(candidate.get("confidence", 0.0)),
        TYPE_ORDER.get(claim_type, 99),
        str(candidate.get("title", "")),
        str(candidate.get("temp_id", "")),
    )


def build_candidates(
    statements: List[Dict[str, Any]],
    source_context: SourceContext,
    scope: str,
    source_node_id: Optional[str],
    entity_id: Optional[str],
    max_nodes: int,
    min_confidence: float,
    include_assumptions: bool,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
    warnings: List[str] = []

    raw_candidates: List[Dict[str, Any]] = []
    for idx, statement in enumerate(statements, start=1):
        text = clean_statement_prefix(str(statement.get("text", "")))
        if len(normalize_key_text(text).split(" ")) < 3:
            continue

        claim_type, confidence, polarity, modal = detect_claim_type(text)
        if claim_type == "assumption":
            if not include_assumptions:
                continue
            if confidence < 0.35:
                continue
        else:
            if confidence < min_confidence:
                continue

        title = truncate_text(compact_whitespace(text), 96)
        summary = truncate_text(compact_whitespace(text), 280)

        knowledge_key = make_knowledge_key(claim_type, scope, title)
        topic_key = topic_key_from_title(title)

        tags = [claim_type, "conversation_commit"]
        if claim_type == "assumption":
            tags.append("uncertain")
        if claim_type == "constraint":
            tags.append("policy")

        links: List[str] = []
        relationships: List[Dict[str, Any]] = []
        if entity_id:
            links.append(entity_id)
            relationships.append({"rel": "about", "target_id": entity_id})

        temp_id = f"TEMP-KG-{idx:03d}"
        claim_hash = sha256_hex_text(f"{claim_type}|{normalize_key_text(title)}|{normalize_key_text(summary)}|{scope}")

        evidence = {
            "source_id": source_node_id or "TEMP-SOURCE",
            "confidence": confidence,
            "source_digest": source_context.source_digest,
            "span": statement["span"],
            "excerpt": statement["excerpt"],
        }

        raw_candidates.append(
            {
                "temp_id": temp_id,
                "type": claim_type,
                "title": title,
                "summary": summary,
                "status": "active",
                "confidence": confidence,
                "claim_hash": claim_hash,
                "knowledge_key": knowledge_key,
                "topic_key": topic_key,
                "polarity": polarity,
                "modal": modal,
                "scope": scope,
                "tags": unique_keep_order(tags),
                "links": unique_keep_order(links),
                "relationships": relationships,
                "evidence": [evidence],
            }
        )

    deduped: Dict[str, Dict[str, Any]] = {}
    dedupe_collapsed = 0

    for candidate in raw_candidates:
        key = candidate["knowledge_key"]
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = candidate
            continue

        dedupe_collapsed += 1
        existing["evidence"] = existing["evidence"] + candidate.get("evidence", [])
        existing["evidence"] = existing["evidence"][:10]
        if float(candidate.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
            preserved_temp = existing.get("temp_id")
            preserved_evidence = existing.get("evidence", [])
            deduped[key] = candidate
            deduped[key]["temp_id"] = preserved_temp
            deduped[key]["evidence"] = preserved_evidence

    if dedupe_collapsed:
        warnings.append(f"Collapsed {dedupe_collapsed} duplicate candidate(s) by knowledge_key")

    candidates = sorted(deduped.values(), key=candidate_sort_key)[:max_nodes]

    stats = {
        "statements_scanned": len(statements),
        "raw_candidates": len(raw_candidates),
        "selected_candidates": len(candidates),
        "dedupe_collapsed": dedupe_collapsed,
    }

    return candidates, warnings, stats


def get_source_key(node: Dict[str, Any]) -> Optional[str]:
    direct = node.get("source_key")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    attrs = node.get("attributes")
    if isinstance(attrs, dict):
        nested = attrs.get("source_key")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def get_knowledge_key(node: Dict[str, Any]) -> Optional[str]:
    direct = node.get("knowledge_key")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    attrs = node.get("attributes")
    if isinstance(attrs, dict):
        nested = attrs.get("knowledge_key")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def semantic_source_view(node: Dict[str, Any]) -> Dict[str, Any]:
    attrs = node.get("attributes") if isinstance(node.get("attributes"), dict) else {}
    return {
        "source_key": get_source_key(node),
        "source_digest": attrs.get("source_digest"),
        "source_kind": attrs.get("source_kind"),
        "source_title": attrs.get("source_title"),
        "source_path": attrs.get("source_path"),
    }


def semantic_knowledge_view(node: Dict[str, Any]) -> Dict[str, Any]:
    attrs = node.get("attributes") if isinstance(node.get("attributes"), dict) else {}
    tags = node.get("tags") if isinstance(node.get("tags"), list) else []

    return {
        "type": node.get("type"),
        "title": normalize_key_text(str(node.get("title", ""))),
        "summary": normalize_key_text(str(node.get("summary", ""))),
        "status": node.get("status"),
        "knowledge_key": get_knowledge_key(node),
        "scope": attrs.get("scope"),
        "topic_key": attrs.get("topic_key"),
        "polarity": attrs.get("polarity"),
        "modal": attrs.get("modal"),
        "tags": sorted(str(tag) for tag in tags),
    }


def merge_evidence(existing: Any, incoming: Any, cap: int = 16) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if isinstance(existing, list):
        rows.extend(item for item in existing if isinstance(item, dict))
    if isinstance(incoming, list):
        rows.extend(item for item in incoming if isinstance(item, dict))

    dedup: Dict[str, Dict[str, Any]] = {}
    for item in rows:
        span = item.get("span") if isinstance(item.get("span"), dict) else {}
        key = "|".join(
            [
                str(item.get("source_id", "")),
                str(item.get("source_digest", "")),
                str(span.get("start_line", "")),
                str(span.get("start_char", "")),
                str(span.get("end_line", "")),
                str(span.get("end_char", "")),
            ]
        )
        dedup[key] = item

    merged = list(dedup.values())
    merged.sort(
        key=lambda item: (
            str(item.get("source_id", "")),
            int(item.get("span", {}).get("start_line", 0)) if isinstance(item.get("span"), dict) else 0,
            int(item.get("span", {}).get("start_char", 0)) if isinstance(item.get("span"), dict) else 0,
        )
    )
    return merged[:cap]


def source_node_payload(
    source_id: str,
    source_context: SourceContext,
    rev: int,
    created_at: str,
    updated_at: str,
    knowledge_commit_id: str,
    knowledge_seq: int,
    request_id: Optional[str],
) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {
        "kind": "conversation",
        "source_key": source_context.source_key,
        "source_digest": source_context.source_digest,
        "source_kind": source_context.source_kind,
        "source_title": source_context.source_title,
        "source_request_hash_raw": source_context.source_request_hash_raw,
        "source_request_hash_norm": source_context.source_request_hash_norm,
    }
    if source_context.source_path:
        attributes["source_path"] = source_context.source_path

    row: Dict[str, Any] = {
        "id": source_id,
        "type": "source",
        "title": f"Conversation Source: {source_context.source_title}",
        "summary": "Conversation source used for KG knowledge extraction commits.",
        "attributes": attributes,
        "relationships": [],
        "evidence": [],
        "status": "active",
        "tags": ["source", "conversation", "knowledge_commit"],
        "links": [],
        "source_key": source_context.source_key,
        "rev": rev,
        "created_at": created_at,
        "updated_at": updated_at,
        "created_by": "commit-knowledge",
        "origin": "conversation_knowledge_commit",
        "origin_knowledge_commit_id": knowledge_commit_id,
        "knowledge_seq": knowledge_seq,
    }
    if request_id:
        row["request_id"] = request_id
    return row


def build_existing_source_maps(kg_latest: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for node in kg_latest.values():
        if str(node.get("type", "")) != "source":
            continue
        source_key = get_source_key(node)
        if source_key:
            by_key[source_key] = strip_internal_fields(node)
    return by_key


def build_existing_knowledge_maps(kg_latest: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for node in kg_latest.values():
        claim_type = str(node.get("type", ""))
        if claim_type not in {"fact", "definition", "constraint", "assumption"}:
            continue
        knowledge_key = get_knowledge_key(node)
        if knowledge_key:
            by_key[knowledge_key] = strip_internal_fields(node)
    return by_key


def candidate_to_kg_node(
    candidate: Dict[str, Any],
    node_id: str,
    rev: int,
    created_at: str,
    updated_at: str,
    source_id: str,
    knowledge_commit_id: str,
    knowledge_seq: int,
    source_context: SourceContext,
    request_id: Optional[str],
) -> Dict[str, Any]:
    evidence_rows = []
    for evidence in candidate.get("evidence", []):
        if not isinstance(evidence, dict):
            continue
        row = copy.deepcopy(evidence)
        row["source_id"] = source_id
        row["source_digest"] = source_context.source_digest
        row["excerpt"] = truncate_text(compact_whitespace(str(row.get("excerpt", ""))), 200)
        span = row.get("span") if isinstance(row.get("span"), dict) else {}
        row["span"] = {
            "start_line": int(span.get("start_line", 1)),
            "end_line": int(span.get("end_line", int(span.get("start_line", 1)))),
            "start_char": int(span.get("start_char", 0)),
            "end_char": int(span.get("end_char", 0)),
        }
        row["confidence"] = float(row.get("confidence", candidate.get("confidence", 0.0)))
        evidence_rows.append(row)

    attributes = {
        "knowledge_key": candidate["knowledge_key"],
        "claim_hash": candidate["claim_hash"],
        "scope": candidate["scope"],
        "topic_key": candidate["topic_key"],
        "polarity": candidate["polarity"],
        "modal": candidate["modal"],
        "source_request_hash_raw": source_context.source_request_hash_raw,
        "source_request_hash_norm": source_context.source_request_hash_norm,
    }

    row: Dict[str, Any] = {
        "id": node_id,
        "type": candidate["type"],
        "title": candidate["title"],
        "summary": candidate["summary"],
        "attributes": attributes,
        "relationships": candidate.get("relationships", []),
        "evidence": evidence_rows,
        "status": "active",
        "tags": sorted(unique_keep_order(str(tag) for tag in candidate.get("tags", []))),
        "links": unique_keep_order([source_id] + [str(link) for link in candidate.get("links", [])]),
        "knowledge_key": candidate["knowledge_key"],
        "rev": rev,
        "created_at": created_at,
        "updated_at": updated_at,
        "created_by": "commit-knowledge",
        "origin": "conversation_knowledge_commit",
        "origin_knowledge_commit_id": knowledge_commit_id,
        "knowledge_seq": knowledge_seq,
        "source_request_hash_raw": source_context.source_request_hash_raw,
        "source_request_hash_norm": source_context.source_request_hash_norm,
    }

    if request_id:
        row["request_id"] = request_id

    return row


def is_modal_conflict(left_modal: str, right_modal: str) -> bool:
    if left_modal == right_modal:
        return False
    pair = (left_modal, right_modal)
    rev = (right_modal, left_modal)
    return pair in INCOMPATIBLE_MODAL_PAIRS or rev in INCOMPATIBLE_MODAL_PAIRS


def detect_conflicts(
    candidates: Sequence[Dict[str, Any]],
    kg_latest: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    topic_index: Dict[str, List[Dict[str, Any]]] = {}

    for node in kg_latest.values():
        claim_type = str(node.get("type", ""))
        if claim_type not in {"fact", "definition", "constraint", "assumption"}:
            continue

        attrs = node.get("attributes") if isinstance(node.get("attributes"), dict) else {}
        topic_key = attrs.get("topic_key")
        if not isinstance(topic_key, str) or not topic_key.strip():
            topic_key = topic_key_from_title(str(node.get("title", "")))

        topic_index.setdefault(topic_key, []).append(strip_internal_fields(node))

    conflicts: List[Dict[str, Any]] = []

    for candidate in candidates:
        topic_key = str(candidate.get("topic_key", "")).strip()
        if not topic_key:
            continue

        existing_nodes = topic_index.get(topic_key, [])
        for existing in existing_nodes:
            existing_key = get_knowledge_key(existing)
            if existing_key and existing_key == candidate["knowledge_key"]:
                continue

            existing_attrs = existing.get("attributes") if isinstance(existing.get("attributes"), dict) else {}
            existing_polarity = str(existing_attrs.get("polarity", "positive"))
            existing_modal = str(existing_attrs.get("modal", "none"))

            candidate_polarity = str(candidate.get("polarity", "positive"))
            candidate_modal = str(candidate.get("modal", "none"))

            opposite_polarity = existing_polarity != candidate_polarity
            modal_conflict = is_modal_conflict(existing_modal, candidate_modal)

            if not opposite_polarity and not modal_conflict:
                continue

            reason_parts = []
            if opposite_polarity:
                reason_parts.append("opposite_polarity")
            if modal_conflict:
                reason_parts.append("modal_incompatibility")

            conflicts.append(
                {
                    "candidate_temp_id": candidate["temp_id"],
                    "candidate_knowledge_key": candidate["knowledge_key"],
                    "existing_id": existing.get("id"),
                    "existing_knowledge_key": existing_key,
                    "topic_key": topic_key,
                    "reason": ",".join(reason_parts),
                }
            )

    conflicts.sort(
        key=lambda row: (
            str(row.get("topic_key", "")),
            str(row.get("candidate_temp_id", "")),
            str(row.get("existing_id", "")),
        )
    )
    return conflicts


def index_row_for_node(layer: str, node: Dict[str, Any]) -> Dict[str, Any]:
    full_view = strip_internal_fields(node)

    content_view = copy.deepcopy(full_view)
    content_view.pop("updated_at", None)
    if layer == "TG":
        content_view.pop("log", None)

    return {
        "id": full_view.get("id"),
        "title": full_view.get("title", ""),
        "tags": full_view.get("tags", []),
        "updated_at": full_view.get("updated_at", ""),
        "type": full_view.get("type", ""),
        "layer": layer,
        "status": full_view.get("status") if layer == "TG" else None,
        "content_hash": sha256_hex_obj(content_view),
        "full_hash": sha256_hex_obj(full_view),
    }


def build_index_rows(layer: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest = resolve_latest_view(records)
    rows = [index_row_for_node(layer, node) for _, node in sorted(latest.items())]
    return rows


def rebuild_indexes(paths: GraphPaths, dry_run: bool) -> None:
    raw_layers = read_raw_layers(paths)
    kg_rows = build_index_rows("KG", raw_layers["KG"])
    rg_rows = build_index_rows("RG", raw_layers["RG"])
    tg_rows = build_index_rows("TG", raw_layers["TG"])

    write_jsonl_atomic(paths.kg_index, kg_rows, dry_run)
    write_jsonl_atomic(paths.rg_index, rg_rows, dry_run)
    write_jsonl_atomic(paths.tg_index, tg_rows, dry_run)


def verify_indexes(paths: GraphPaths) -> None:
    raw_layers = read_raw_layers(paths)

    expected = {
        "KG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in build_index_rows("KG", raw_layers["KG"])],
        "RG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in build_index_rows("RG", raw_layers["RG"])],
        "TG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in build_index_rows("TG", raw_layers["TG"])],
    }

    actual = {
        "KG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in read_jsonl(paths.kg_index)],
        "RG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in read_jsonl(paths.rg_index)],
        "TG": [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in read_jsonl(paths.tg_index)],
    }

    for layer in ("KG", "RG", "TG"):
        if expected[layer] != actual[layer]:
            raise CommitKnowledgeError(f"Index verification failed for {layer}")


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
            conf_float = float(conf)
        except (TypeError, ValueError):
            continue
        best = max(best, conf_float)
    return round(best, 4)


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

    tg_nodes = [strip_internal_fields(node) for node in tg_latest.values()]
    tg_nodes.sort(key=lambda node: str(node.get("updated_at", "")), reverse=True)

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

    kg_nodes = [strip_internal_fields(node) for node in kg_latest.values()]
    kg_nodes.sort(key=lambda node: str(node.get("updated_at", "")), reverse=True)

    changed_kg_ids = [node_id for node_id in changed_node_ids if node_id.startswith("KG-")]
    changed_kg_nodes = [kg_latest[node_id] for node_id in changed_kg_ids if node_id in kg_latest]

    prioritized: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for node in changed_kg_nodes + kg_nodes:
        node_id = str(node.get("id", ""))
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        prioritized.append(node)

    knowledge_snapshot = [
        {
            "id": node.get("id"),
            "title": node.get("title"),
            "summary": node.get("summary", ""),
            "confidence": max_evidence_confidence(node),
        }
        for node in prioritized[:30]
    ]

    rg_nodes = [strip_internal_fields(node) for node in rg_latest.values()]
    rg_nodes = [
        node
        for node in rg_nodes
        if str(node.get("outcome", {}).get("status", "")) in {"accepted", "proposed", "superseded"}
    ]
    rg_nodes.sort(key=lambda node: str(node.get("updated_at", "")), reverse=True)

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

    if not dry_run:
        written = read_json(history_path)
        if written is None or written.get("commit_id") != context_pack.get("commit_id"):
            raise CommitKnowledgeError("Context history write validation failed")

    write_json_atomic(paths.context_latest, context_pack, dry_run)

    history_files = sorted(paths.context_packs_dir.glob("context_pack.*.json"))
    if len(history_files) > history_limit:
        to_remove = history_files[: len(history_files) - history_limit]
        for old_file in to_remove:
            if dry_run:
                print(f"[DRY-RUN] remove {old_file}")
            else:
                old_file.unlink(missing_ok=True)

    return str(history_path)


def append_commit_record(
    paths: GraphPaths,
    commit_id: str,
    changed_node_ids: List[str],
    context_pack_path: str,
    knowledge_commit_id: str,
    knowledge_seq: int,
    source_context: SourceContext,
    request_id: Optional[str],
    dry_run: bool,
) -> None:
    row: Dict[str, Any] = {
        "commit_id": commit_id,
        "mode": "commit_knowledge_apply",
        "changed_node_ids": changed_node_ids,
        "context_pack_path": context_pack_path,
        "created_at": now_iso_utc(),
        "event": "post_response_commit",
        "knowledge_commit_id": knowledge_commit_id,
        "knowledge_seq": knowledge_seq,
        "source_key": source_context.source_key,
        "source_request_hash_raw": source_context.source_request_hash_raw,
        "source_request_hash_norm": source_context.source_request_hash_norm,
    }
    if request_id:
        row["request_id"] = request_id

    append_jsonl_atomic(paths.post_commits, [row], dry_run)


def history_files(paths: GraphPaths) -> List[Path]:
    if not paths.knowledge_commits_dir.exists():
        return []
    return sorted(paths.knowledge_commits_dir.glob("knowledge_commit.*.json"))


def load_knowledge_seq_values(paths: GraphPaths) -> List[int]:
    values: List[int] = []
    for file_path in history_files(paths):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        seq = payload.get("knowledge_seq")
        if isinstance(seq, int) and seq >= 1:
            values.append(seq)
    return values


def rotate_knowledge_history(paths: GraphPaths, history_limit: int, dry_run: bool) -> None:
    files = history_files(paths)
    if len(files) <= history_limit:
        return

    to_remove = files[: len(files) - history_limit]
    for old in to_remove:
        if dry_run:
            print(f"[DRY-RUN] remove {old}")
        else:
            old.unlink(missing_ok=True)


def persist_knowledge_history(
    paths: GraphPaths,
    base_payload: Dict[str, Any],
    history_limit: int,
    dry_run: bool,
) -> Tuple[Path, str, int, Dict[str, Any]]:
    if not dry_run:
        paths.knowledge_commits_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(2):
        seq_values = load_knowledge_seq_values(paths)
        knowledge_seq = (max(seq_values) + 1) if seq_values else 1
        knowledge_commit_id = f"KGCMT-{new_ulid()}"

        payload = copy.deepcopy(base_payload)
        payload["knowledge_commit_id"] = knowledge_commit_id
        payload["knowledge_seq"] = knowledge_seq

        history_name = f"knowledge_commit.{timestamp_slug()}.{knowledge_commit_id}.json"
        history_path = paths.knowledge_commits_dir / history_name

        if dry_run:
            print(f"[DRY-RUN] write {history_path}")
            return history_path, knowledge_commit_id, knowledge_seq, payload

        if history_path.exists():
            continue

        write_json_atomic(history_path, payload, dry_run=False)
        written = read_json(history_path)
        if written is None or written.get("knowledge_commit_id") != knowledge_commit_id:
            history_path.unlink(missing_ok=True)
            raise CommitKnowledgeError("Failed to validate written knowledge history file")

        seq_after = load_knowledge_seq_values(paths)
        if seq_after.count(knowledge_seq) == 1:
            rotate_knowledge_history(paths, history_limit=history_limit, dry_run=False)
            return history_path, knowledge_commit_id, knowledge_seq, payload

        history_path.unlink(missing_ok=True)
        if attempt == 0:
            continue
        raise CommitKnowledgeError("knowledge_seq allocation conflict after retry")

    raise CommitKnowledgeError("Unable to allocate knowledge_seq")


def parse_conversation_text(args: argparse.Namespace) -> str:
    if args.conversation_text and args.conversation_file:
        raise CommitKnowledgeError("Provide either --conversation-text or --conversation-file, not both")
    if not args.conversation_text and not args.conversation_file:
        raise CommitKnowledgeError("One of --conversation-text or --conversation-file is required")

    if args.conversation_text:
        return args.conversation_text

    file_path = Path(args.conversation_file).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        raise CommitKnowledgeError(f"conversation file not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and commit conversation knowledge into KG")
    parser.add_argument("--project-root", required=True, help="Project root path")
    parser.add_argument("--graphs-dir", default="graphs", help="Graphs directory under project root")

    parser.add_argument("--conversation-text", default=None, help="Conversation text input")
    parser.add_argument("--conversation-file", default=None, help="Path to file with conversation text")

    parser.add_argument("--apply", type=parse_bool, default=False)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--max-nodes", type=int, default=20)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--source-key", default=None)
    parser.add_argument("--source-title", default=None)
    parser.add_argument("--include-assumptions", type=parse_bool, default=True)
    parser.add_argument("--plan-history-limit", type=int, default=200)
    parser.add_argument("--context-pack-history-limit", type=int, default=200)
    parser.add_argument("--stdout-json", type=parse_bool, default=False)
    parser.add_argument("--strict-empty", type=parse_bool, default=False)
    parser.add_argument("--apply-conflicts", type=parse_bool, default=False)
    parser.add_argument("--apply-conflict-task", type=parse_bool, default=False)
    return parser.parse_args()


def create_conflict_task(
    conflict_row: Dict[str, Any],
    kg_node_id: str,
    existing_id: str,
    now: str,
    source_context: SourceContext,
) -> Dict[str, Any]:
    task_id = f"TG-TASK-{new_ulid()}"
    title = f"Resolve knowledge conflict: {truncate_text(str(conflict_row.get('topic_key', 'topic')), 72)}"

    return {
        "id": task_id,
        "type": "task",
        "title": title,
        "description": "Review conflicting KG claims and decide authoritative constraint/fact.",
        "status": "ready",
        "priority": "high",
        "owner_role": "lead",
        "dependencies": [],
        "blocks": [],
        "acceptance_criteria": [
            "Conflict record is reviewed and marked resolved or superseded",
            "Resulting KG update references authoritative source evidence",
        ],
        "inputs": {
            "kg_refs": unique_keep_order([kg_node_id, existing_id]),
            "rg_refs": [],
            "sources": [source_context.source_key],
        },
        "outputs": {
            "artifacts": [],
            "kg_updates": [kg_node_id, existing_id],
            "rg_updates": [],
        },
        "log": [{"at": now, "event": "created", "by": "commit-knowledge"}],
        "links": unique_keep_order([kg_node_id, existing_id]),
        "tags": ["conflict_resolution", "clarify"],
        "rev": 1,
        "created_at": now,
        "updated_at": now,
        "created_by": "commit-knowledge",
        "origin": "knowledge_conflict",
    }


def build_conflict_record(
    conflict_row: Dict[str, Any],
    candidate_node_id: str,
    resolution_task_ids: List[str],
    now: str,
    source_context: SourceContext,
) -> Dict[str, Any]:
    conflict_id = f"CONFLICT-{new_ulid()}"

    evidence = [
        {
            "source_digest": source_context.source_digest,
            "reason": conflict_row.get("reason"),
            "topic_key": conflict_row.get("topic_key"),
            "candidate_temp_id": conflict_row.get("candidate_temp_id"),
            "candidate_knowledge_key": conflict_row.get("candidate_knowledge_key"),
            "existing_knowledge_key": conflict_row.get("existing_knowledge_key"),
        }
    ]

    existing_id = str(conflict_row.get("existing_id", ""))

    return {
        "id": conflict_id,
        "title": f"Detected knowledge conflict on topic '{conflict_row.get('topic_key', '')}'",
        "status": "open",
        "created_at": now,
        "updated_at": now,
        "node_ids": unique_keep_order([candidate_node_id, existing_id]),
        "evidence": evidence,
        "resolution_task_ids": resolution_task_ids,
        "notes": "Automatically detected by commit-knowledge.",
        "tags": ["auto_detected", "knowledge_conflict"],
    }


def find_existing_conflict_resolution_tasks(tg_latest: Dict[str, Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for node_id, node in tg_latest.items():
        tags = node.get("tags")
        if isinstance(tags, list) and "conflict_resolution" in tags:
            ids.append(node_id)
    return sorted(ids)


def ensure_iso_fields(rows: Iterable[Dict[str, Any]], label: str) -> None:
    for row in rows:
        for field in ("created_at", "updated_at"):
            value = row.get(field)
            if not isinstance(value, str) or ISO_PATTERN.match(value) is None:
                raise CommitKnowledgeError(f"{label} row has invalid {field}")


def main() -> int:
    args = parse_args()

    if args.max_nodes < 1 or args.max_nodes > 200:
        print("[ERROR] --max-nodes must be between 1 and 200")
        return 2

    if not (0.0 <= args.min_confidence <= 1.0):
        print("[ERROR] --min-confidence must be between 0 and 1")
        return 2

    if args.plan_history_limit < 1 or args.context_pack_history_limit < 1:
        print("[ERROR] history limits must be >= 1")
        return 2

    if args.apply_conflict_task and not args.apply_conflicts:
        print("[ERROR] --apply-conflict-task true requires --apply-conflicts true")
        return 2

    if args.apply_conflict_task and not args.apply:
        print("[ERROR] --apply-conflict-task true requires --apply true")
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
        raw_text = parse_conversation_text(args)
    except CommitKnowledgeError as exc:
        print(f"[ERROR] {exc}")
        return 2

    try:
        source_context = build_source_context(args, raw_text)

        raw_layers = read_raw_layers(paths)
        latest_layers = {layer: resolve_latest_view(rows) for layer, rows in raw_layers.items()}

        project_name = extract_project_name(paths)
        entity_id = resolve_entity_scope(raw_text, latest_layers["KG"])
        scope = derive_scope(project_name, entity_id)

        existing_sources = build_existing_source_maps(latest_layers["KG"])
        existing_source_node = existing_sources.get(source_context.source_key)
        source_node_id = str(existing_source_node.get("id")) if existing_source_node else None

        statements = segment_statements(raw_text)
        candidates, extraction_warnings, extraction_stats = build_candidates(
            statements=statements,
            source_context=source_context,
            scope=scope,
            source_node_id=source_node_id,
            entity_id=entity_id,
            max_nodes=args.max_nodes,
            min_confidence=args.min_confidence,
            include_assumptions=args.include_assumptions,
        )

        if not candidates and args.strict_empty:
            raise CommitKnowledgeError("No commit-worthy knowledge candidates found (--strict-empty true)")

        conflict_candidates = detect_conflicts(candidates, latest_layers["KG"])
        suggested_conflict_tasks = [
            {
                "title": f"Resolve conflict for {row['topic_key']}",
                "candidate_temp_id": row["candidate_temp_id"],
                "existing_id": row["existing_id"],
                "reason": row["reason"],
            }
            for row in conflict_candidates
        ]

        warnings: List[str] = list(extraction_warnings)
        if conflict_candidates:
            warnings.append(f"Detected {len(conflict_candidates)} conflict candidate(s)")

        base_payload: Dict[str, Any] = {
            "generated_at": now_iso_utc(),
            "source": {
                "source_key": source_context.source_key,
                "source_digest": source_context.source_digest,
                "source_kind": source_context.source_kind,
                "source_title": source_context.source_title,
                "source_node_id": source_node_id,
            },
            "candidates": candidates,
            "warnings": warnings,
            "stats": {
                **extraction_stats,
                "conflict_candidates": len(conflict_candidates),
                "apply_requested": bool(args.apply),
            },
            "source_request_hash_raw": source_context.source_request_hash_raw,
            "source_request_hash_norm": source_context.source_request_hash_norm,
        }

        if args.request_id:
            base_payload["request_id"] = args.request_id

        if conflict_candidates:
            base_payload["conflict_candidates"] = conflict_candidates
            base_payload["suggested_conflict_tasks"] = suggested_conflict_tasks

        history_path, knowledge_commit_id, knowledge_seq, payload = persist_knowledge_history(
            paths=paths,
            base_payload=base_payload,
            history_limit=args.plan_history_limit,
            dry_run=args.dry_run,
        )

        payload["knowledge_commit_id"] = knowledge_commit_id
        payload["knowledge_seq"] = knowledge_seq

        changed_kg_rows: List[Dict[str, Any]] = []
        changed_tg_rows: List[Dict[str, Any]] = []
        changed_conflict_rows: List[Dict[str, Any]] = []
        changed_node_ids: List[str] = []

        if args.apply:
            now = now_iso_utc()
            kg_latest = {k: strip_internal_fields(v) for k, v in latest_layers["KG"].items()}
            tg_latest = {k: strip_internal_fields(v) for k, v in latest_layers["TG"].items()}

            existing_source = build_existing_source_maps(kg_latest).get(source_context.source_key)
            if existing_source is None:
                source_id = f"KG-SOURCE-{new_ulid()}"
                source_row = source_node_payload(
                    source_id=source_id,
                    source_context=source_context,
                    rev=1,
                    created_at=now,
                    updated_at=now,
                    knowledge_commit_id=knowledge_commit_id,
                    knowledge_seq=knowledge_seq,
                    request_id=args.request_id,
                )
                changed_kg_rows.append(source_row)
                changed_node_ids.append(source_id)
                kg_latest[source_id] = source_row
            else:
                source_id = str(existing_source.get("id"))
                desired_source = source_node_payload(
                    source_id=source_id,
                    source_context=source_context,
                    rev=int(existing_source.get("rev", 1)),
                    created_at=str(existing_source.get("created_at", now)),
                    updated_at=now,
                    knowledge_commit_id=knowledge_commit_id,
                    knowledge_seq=knowledge_seq,
                    request_id=args.request_id,
                )
                if semantic_source_view(existing_source) != semantic_source_view(desired_source):
                    desired_source["rev"] = int(existing_source.get("rev", 1)) + 1
                    desired_source["created_at"] = str(existing_source.get("created_at", now))
                    changed_kg_rows.append(desired_source)
                    changed_node_ids.append(source_id)
                    kg_latest[source_id] = desired_source
                else:
                    source_id = str(existing_source.get("id"))

            source_obj = payload.get("source")
            if isinstance(source_obj, dict):
                source_obj["source_node_id"] = source_id

            payload_candidates = payload.get("candidates")
            if isinstance(payload_candidates, list):
                for payload_candidate in payload_candidates:
                    if not isinstance(payload_candidate, dict):
                        continue
                    evidence_rows = payload_candidate.get("evidence")
                    if not isinstance(evidence_rows, list):
                        continue
                    for evidence_row in evidence_rows:
                        if isinstance(evidence_row, dict):
                            evidence_row["source_id"] = source_id

            existing_knowledge = build_existing_knowledge_maps(kg_latest)
            candidate_applied_node_ids: Dict[str, str] = {}

            for candidate in sorted(candidates, key=candidate_sort_key):
                key = candidate["knowledge_key"]
                existing = existing_knowledge.get(key)

                if existing is None:
                    node_id = f"KG-{candidate['type'].upper()}-{new_ulid()}"
                    new_row = candidate_to_kg_node(
                        candidate=candidate,
                        node_id=node_id,
                        rev=1,
                        created_at=now,
                        updated_at=now,
                        source_id=source_id,
                        knowledge_commit_id=knowledge_commit_id,
                        knowledge_seq=knowledge_seq,
                        source_context=source_context,
                        request_id=args.request_id,
                    )
                    changed_kg_rows.append(new_row)
                    changed_node_ids.append(node_id)
                    candidate_applied_node_ids[candidate["temp_id"]] = node_id
                    existing_knowledge[key] = new_row
                    kg_latest[node_id] = new_row
                    continue

                node_id = str(existing.get("id"))
                desired = candidate_to_kg_node(
                    candidate=candidate,
                    node_id=node_id,
                    rev=int(existing.get("rev", 1)),
                    created_at=str(existing.get("created_at", now)),
                    updated_at=now,
                    source_id=source_id,
                    knowledge_commit_id=knowledge_commit_id,
                    knowledge_seq=knowledge_seq,
                    source_context=source_context,
                    request_id=args.request_id,
                )

                if semantic_knowledge_view(existing) == semantic_knowledge_view(desired):
                    candidate_applied_node_ids[candidate["temp_id"]] = node_id
                    continue

                desired["rev"] = int(existing.get("rev", 1)) + 1
                desired["created_at"] = str(existing.get("created_at", now))
                desired["evidence"] = merge_evidence(existing.get("evidence"), desired.get("evidence"))
                desired["tags"] = sorted(unique_keep_order([*(existing.get("tags") or []), *(desired.get("tags") or [])]))
                desired["links"] = unique_keep_order([*(existing.get("links") or []), *(desired.get("links") or [])])

                changed_kg_rows.append(desired)
                changed_node_ids.append(node_id)
                candidate_applied_node_ids[candidate["temp_id"]] = node_id
                existing_knowledge[key] = desired
                kg_latest[node_id] = desired

            if args.apply_conflicts and conflict_candidates:
                existing_resolution_tasks = find_existing_conflict_resolution_tasks(tg_latest)

                if not args.apply_conflict_task and not existing_resolution_tasks:
                    raise CommitKnowledgeError(
                        "--apply-conflicts requested but no existing TG task tagged conflict_resolution is available. "
                        "Use --apply-conflict-task true or create a tagged resolution task first."
                    )

                for conflict in conflict_candidates:
                    temp_id = str(conflict.get("candidate_temp_id", ""))
                    candidate_node_id = candidate_applied_node_ids.get(temp_id)
                    if not candidate_node_id:
                        continue

                    existing_id = str(conflict.get("existing_id", ""))
                    resolution_ids: List[str]

                    if args.apply_conflict_task:
                        task_row = create_conflict_task(
                            conflict_row=conflict,
                            kg_node_id=candidate_node_id,
                            existing_id=existing_id,
                            now=now,
                            source_context=source_context,
                        )
                        changed_tg_rows.append(task_row)
                        task_id = str(task_row["id"])
                        changed_node_ids.append(task_id)
                        resolution_ids = [task_id]
                    else:
                        resolution_ids = [existing_resolution_tasks[0]]

                    conflict_row = build_conflict_record(
                        conflict_row=conflict,
                        candidate_node_id=candidate_node_id,
                        resolution_task_ids=resolution_ids,
                        now=now,
                        source_context=source_context,
                    )
                    changed_conflict_rows.append(conflict_row)

            ensure_iso_fields(changed_kg_rows, "KG")
            ensure_iso_fields(changed_tg_rows, "TG")

            append_jsonl_atomic(paths.kg_nodes, changed_kg_rows, args.dry_run)
            append_jsonl_atomic(paths.tg_nodes, changed_tg_rows, args.dry_run)
            append_jsonl_atomic(paths.conflicts, changed_conflict_rows, args.dry_run)

            rebuild_indexes(paths, args.dry_run)
            if not args.dry_run:
                verify_indexes(paths)

            payload["stats"]["kg_rows_written"] = len(changed_kg_rows)
            payload["stats"]["tg_rows_written"] = len(changed_tg_rows)
            payload["stats"]["conflicts_written"] = len(changed_conflict_rows)

            if changed_node_ids:
                commit_id = f"COMMIT-{new_ulid()}"
                context_pack = build_post_response_context_pack(
                    paths=paths,
                    commit_id=commit_id,
                    changed_node_ids=changed_node_ids,
                    goal_hint="Commit extracted conversation knowledge",
                )

                context_pack_path = write_context_pack(
                    paths=paths,
                    context_pack=context_pack,
                    history_limit=args.context_pack_history_limit,
                    dry_run=args.dry_run,
                )

                try:
                    append_commit_record(
                        paths=paths,
                        commit_id=commit_id,
                        changed_node_ids=changed_node_ids,
                        context_pack_path=context_pack_path,
                        knowledge_commit_id=knowledge_commit_id,
                        knowledge_seq=knowledge_seq,
                        source_context=source_context,
                        request_id=args.request_id,
                        dry_run=args.dry_run,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    raise CommitKnowledgeError(
                        "Context pack updated but commit log append failed. "
                        "Re-run commit-knowledge with --apply true to backfill post_response_commits. "
                        f"Original error: {exc}"
                    ) from exc

                payload["commit_id"] = commit_id
                payload["context_pack_path"] = context_pack_path
            else:
                payload["warnings"] = payload.get("warnings", []) + ["Apply mode produced no KG/TG changes; commit/context sync skipped"]
                payload["stats"]["kg_rows_written"] = 0
                payload["stats"]["tg_rows_written"] = 0
                payload["stats"]["conflicts_written"] = 0

        if args.stdout_json:
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(f"[OK] knowledge_commit_id: {knowledge_commit_id}")
            print(f"[OK] knowledge_seq: {knowledge_seq}")
            print(f"[OK] candidates: {len(candidates)}")
            print(f"[OK] history: {history_path}")
            print(f"[OK] source_key: {source_context.source_key}")
            print(f"[OK] source_request_hash_raw: {source_context.source_request_hash_raw}")
            print(f"[OK] source_request_hash_norm: {source_context.source_request_hash_norm}")
            if args.apply:
                print(f"[OK] apply: true")
                print(f"[OK] kg_rows_written: {payload.get('stats', {}).get('kg_rows_written', 0)}")
                print(f"[OK] tg_rows_written: {payload.get('stats', {}).get('tg_rows_written', 0)}")
                print(f"[OK] conflicts_written: {payload.get('stats', {}).get('conflicts_written', 0)}")
                if payload.get("commit_id"):
                    print(f"[OK] commit_id: {payload['commit_id']}")
                if payload.get("context_pack_path"):
                    print(f"[OK] context_pack_path: {payload['context_pack_path']}")

            for warning in payload.get("warnings", []):
                print(f"[WARN] {warning}")

        return 0

    except CommitKnowledgeError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] unexpected failure: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
