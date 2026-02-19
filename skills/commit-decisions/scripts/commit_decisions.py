#!/usr/bin/env python3
"""Commit alternative-analysis decisions into RG with deterministic upsert semantics."""

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
ID_PATTERN = re.compile(r"\b(?:KG|RG|TG)-[A-Z0-9-]{8,}\b")
WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
ALT_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$")
ALT_OPTION_RE = re.compile(r"^\s*(?:option|вариант|опция)\s+[a-z0-9а-яё]+\s*[:\-]\s*(.+?)\s*$", flags=re.IGNORECASE)
INLINE_OPTIONS_RE = re.compile(
    r"(?:option|вариант|опция)\s+[a-z0-9а-яё]+\s*[:\-]\s*(.+?)(?=(?:\s+(?:option|вариант|опция)\s+[a-z0-9а-яё]+\s*[:\-])|$)",
    flags=re.IGNORECASE,
)
TRAILING_SELECTION_MARKER_RE = re.compile(r"\.\s*(recommended|selected|chosen|pick|выбрано|рекомендовано|предпочтительно)\s*:\s*.*$", flags=re.IGNORECASE)

OUTCOME_STATUSES = {"proposed", "accepted", "rejected", "superseded"}
ASSUMPTION_MARKERS = [
    "assume",
    "assuming",
    "might",
    "may",
    "could",
    "uncertain",
    "if ",
    "предполож",
    "возможно",
    "может",
    "если ",
]
RISK_MARKERS = [
    "risk",
    "tradeoff",
    "downside",
    "cost",
    "liability",
    "failure",
    "danger",
    "риск",
    "компромисс",
    "минус",
    "опасн",
    "затрат",
]
VALIDATION_MARKERS = [
    "validate",
    "test",
    "verify",
    "measure",
    "check",
    "assert",
    "провер",
    "тест",
    "валид",
    "измер",
]


class CommitDecisionsError(RuntimeError):
    """Raised for deterministic commit-decisions failures."""


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
    lowered = normalize_newlines(text).strip().lower()
    lowered = lowered.replace("_", " ")
    lowered = NON_WORD_RE.sub(" ", lowered)
    lowered = WHITESPACE_RE.sub(" ", lowered).strip()
    return lowered


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


def compact_ws(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def tokenize(text: str) -> List[str]:
    return [tok for tok in WORD_PATTERN.findall(normalize_for_dedupe(text)) if tok]


def unique_keep_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


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
            raise CommitDecisionsError(f"Invalid JSONL in {path} line {idx}: {exc}") from exc
        if not isinstance(payload, dict):
            raise CommitDecisionsError(f"Invalid JSONL object in {path} line {idx}")
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


def strip_internal(node: Dict[str, Any]) -> Dict[str, Any]:
    clean = copy.deepcopy(node)
    clean.pop("_line_no", None)
    return clean


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
    def post_commits(self) -> Path:
        return self.graphs_dir / "post_response_commits.jsonl"

    @property
    def context_latest(self) -> Path:
        return self.graphs_dir / "context_pack.latest.json"

    @property
    def context_packs_dir(self) -> Path:
        return self.graphs_dir / "context_packs"

    @property
    def decision_commits_dir(self) -> Path:
        return self.graphs_dir / "decision_commits"

    @property
    def meta(self) -> Path:
        return self.graphs_dir / "meta.json"


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

    best_key: Optional[Tuple[Any, ...]] = None
    best_record: Optional[Dict[str, Any]] = None
    best_line = -1

    for line_no, rec in records:
        updated = parse_iso(rec.get("updated_at"))
        updated_epoch = updated.timestamp() if updated else float("-inf")
        rev = int(rec.get("rev", 0)) if isinstance(rec.get("rev"), int) else 0

        key = (rev, updated_epoch, line_no) if all_have_rev else (updated_epoch, line_no)

        if best_key is None or key > best_key:
            best_key = key
            best_record = rec
            best_line = line_no

    if best_record is None:
        raise CommitDecisionsError("Unable to resolve latest view")

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
    for node_id, recs in grouped.items():
        latest[node_id] = _resolve_latest_for_id(recs)
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
            raise CommitDecisionsError(f"Index verification failed for {layer}")


def extract_project_name(paths: GraphPaths) -> str:
    meta = read_json(paths.meta)
    if isinstance(meta, dict):
        name = meta.get("project_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return paths.project_root.name


def parse_request_text(args: argparse.Namespace) -> str:
    if args.decision_text and args.decision_file:
        raise CommitDecisionsError("Provide either --decision-text or --decision-file, not both")
    if not args.decision_text and not args.decision_file:
        raise CommitDecisionsError("One of --decision-text or --decision-file is required")

    if args.decision_text:
        return args.decision_text

    file_path = Path(args.decision_file).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        raise CommitDecisionsError(f"decision file not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


def parse_csv_ids(raw: Optional[str], prefix: str) -> List[str]:
    if not raw:
        return []
    values = [item.strip() for item in raw.split(",") if item.strip()]
    for value in values:
        if not value.startswith(prefix + "-"):
            raise CommitDecisionsError(f"Invalid {prefix} reference: {value}")
    return unique_keep_order(values)


def parse_explicit_ids_from_text(text: str) -> Tuple[List[str], List[str], List[str]]:
    ids = unique_keep_order(ID_PATTERN.findall(text))
    kg: List[str] = []
    rg: List[str] = []
    tg: List[str] = []
    for node_id in ids:
        if node_id.startswith("KG-"):
            kg.append(node_id)
        elif node_id.startswith("RG-"):
            rg.append(node_id)
        elif node_id.startswith("TG-"):
            tg.append(node_id)
    return kg, rg, tg


def node_search_text(node: Dict[str, Any]) -> str:
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
        for value in attrs.values():
            if isinstance(value, (str, int, float, bool)):
                parts.append(str(value))

    return " ".join(parts)


def lexical_auto_links(
    text: str,
    nodes: Dict[str, Dict[str, Any]],
    prefix: str,
    max_links: int = 6,
) -> List[Tuple[str, float]]:
    tokens = set(tokenize(text))
    if not tokens:
        return []

    scored: List[Tuple[str, float]] = []
    for node_id, node in nodes.items():
        if not node_id.startswith(prefix + "-"):
            continue
        node_tokens = set(tokenize(node_search_text(node)))
        if not node_tokens:
            continue
        overlap = tokens.intersection(node_tokens)
        if not overlap:
            continue
        coverage = len(overlap) / max(1, len(tokens))
        density = len(overlap) / max(1, len(node_tokens))
        score = min(0.99, 0.8 * coverage + 0.2 * density)
        if score >= 0.15:
            scored.append((node_id, score))

    scored.sort(key=lambda row: (-row[1], row[0]))
    return scored[:max_links]


def parse_alternatives_from_text(text: str) -> List[Dict[str, Any]]:
    alternatives: List[Dict[str, Any]] = []
    seen_titles: Set[str] = set()

    def add_alternative(title_raw: str) -> None:
        title = compact_ws(TRAILING_SELECTION_MARKER_RE.sub("", title_raw)).rstrip(".;")
        if not title:
            return
        normalized = normalize_key_text(title)
        if normalized in seen_titles:
            return
        seen_titles.add(normalized)
        alternatives.append({"title": title, "pros": [], "cons": []})

    inline_matches = INLINE_OPTIONS_RE.findall(normalize_newlines(text))
    for match in inline_matches:
        add_alternative(str(match))

    for raw_line in normalize_newlines(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = ALT_OPTION_RE.match(line)
        if match:
            add_alternative(match.group(1))
            continue

        match = ALT_BULLET_RE.match(line)
        if match:
            content = compact_ws(match.group(1))
            lowered = content.lower()
            if lowered.startswith("pros:") or lowered.startswith("cons:"):
                continue
            add_alternative(content)

    if not alternatives:
        lowered = normalize_newlines(text).lower()
        if " vs " in lowered or " или " in lowered:
            parts = [
                compact_ws(part)
                for part in re.split(r"\b(?:vs|или)\b", text, flags=re.IGNORECASE)
                if compact_ws(part)
            ]
            for part in parts:
                add_alternative(truncate_text(part, 120))

    return alternatives


def parse_alternatives_from_file(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise CommitDecisionsError(f"alternatives file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CommitDecisionsError(f"Invalid JSON in alternatives file: {exc}") from exc

    def row_from_value(value: Any, fallback_title: Optional[str] = None) -> Dict[str, Any]:
        if isinstance(value, str):
            title = compact_ws(value)
            if not title:
                raise CommitDecisionsError("Alternative title cannot be empty")
            return {"title": title, "pros": [], "cons": []}
        if isinstance(value, dict):
            title_raw = value.get("title", fallback_title)
            if not isinstance(title_raw, str) or not compact_ws(title_raw):
                raise CommitDecisionsError("Alternative object must include non-empty title")
            title = compact_ws(title_raw)
            pros = value.get("pros", [])
            cons = value.get("cons", [])
            pros_list = [compact_ws(str(item)) for item in pros if str(item).strip()] if isinstance(pros, list) else []
            cons_list = [compact_ws(str(item)) for item in cons if str(item).strip()] if isinstance(cons, list) else []
            return {"title": title, "pros": pros_list, "cons": cons_list}
        raise CommitDecisionsError("Alternatives must be strings or objects")

    # Ordered list input: preserve order.
    if isinstance(payload, list):
        return [row_from_value(value) for value in payload]

    # Dictionary input can be unordered: materialize deterministically.
    if isinstance(payload, dict):
        if "alternatives" in payload:
            alt_value = payload.get("alternatives")
            if not isinstance(alt_value, list):
                raise CommitDecisionsError("alternatives key must be a list")
            return [row_from_value(value) for value in alt_value]

        materialized: List[Dict[str, Any]] = []
        for key, value in payload.items():
            materialized.append(row_from_value(value, fallback_title=str(key)))

        materialized.sort(key=lambda row: (normalize_key_text(row["title"]), row["title"]))
        return materialized

    raise CommitDecisionsError("Alternatives JSON must be array or object")


def enrich_alternatives_with_option_ids(alternatives: List[Dict[str, Any]], max_alternatives: int) -> List[Dict[str, Any]]:
    if len(alternatives) > max_alternatives:
        alternatives = alternatives[:max_alternatives]

    enriched: List[Dict[str, Any]] = []
    seen_norm_titles: Set[str] = set()

    for idx, row in enumerate(alternatives, start=1):
        title = compact_ws(str(row.get("title", "")))
        if not title:
            continue

        norm_title = normalize_key_text(title)
        if norm_title in seen_norm_titles:
            continue
        seen_norm_titles.add(norm_title)

        pros = row.get("pros", [])
        cons = row.get("cons", [])
        pros_list = [compact_ws(str(item)) for item in pros if str(item).strip()] if isinstance(pros, list) else []
        cons_list = [compact_ws(str(item)) for item in cons if str(item).strip()] if isinstance(cons, list) else []

        enriched.append(
            {
                "option_id": f"OPT-{idx}",
                "title": title,
                "pros": pros_list,
                "cons": cons_list,
            }
        )

    # Keep original parse/file order; never reorder by normalization here.
    for idx, row in enumerate(enriched, start=1):
        row["option_id"] = f"OPT-{idx}"

    return enriched


def resolve_selected_option(
    alternatives: List[Dict[str, Any]],
    selected_arg: Optional[str],
    text: str,
) -> str:
    if not alternatives:
        return "undecided"

    by_id = {str(row["option_id"]): row for row in alternatives}
    by_norm_title: Dict[str, List[Dict[str, Any]]] = {}
    for row in alternatives:
        key = normalize_key_text(str(row["title"]))
        by_norm_title.setdefault(key, []).append(row)

    if selected_arg:
        selected = selected_arg.strip()
        if selected in by_id:
            return selected

        key = normalize_key_text(selected)
        matches = by_norm_title.get(key, [])
        if len(matches) == 1:
            return str(matches[0]["option_id"])
        raise CommitDecisionsError(f"Unable to resolve --selected-option '{selected}' to a unique option_id")

    lowered = text.lower()
    for marker in [
        "selected:",
        "chosen:",
        "recommended:",
        "choose:",
        "pick:",
        "выбрано:",
        "выбран:",
        "рекомендовано:",
        "предпочтительно:",
        "выбери:",
    ]:
        idx = lowered.find(marker)
        if idx == -1:
            continue
        remainder = text[idx + len(marker) :].splitlines()[0]
        key = normalize_key_text(remainder)
        matches = by_norm_title.get(key, [])
        if len(matches) == 1:
            return str(matches[0]["option_id"])

    return "undecided"


def collect_rationale(text: str, selected_option: str, alternatives: List[Dict[str, Any]]) -> List[str]:
    rationale: List[str] = []

    if selected_option != "undecided":
        selected_title = ""
        for row in alternatives:
            if str(row["option_id"]) == selected_option:
                selected_title = str(row["title"])
                break
        if selected_title:
            rationale.append(f"Selected option {selected_option} ({selected_title}) based on provided alternatives.")

    for line in normalize_newlines(text).splitlines():
        stripped = compact_ws(line)
        if not stripped:
            continue
        lowered = stripped.lower()
        if (
            "because" in lowered
            or "потому что" in lowered
            or lowered.startswith("reason")
            or lowered.startswith("rationale")
            or lowered.startswith("причина")
            or lowered.startswith("обосн")
        ):
            rationale.append(stripped)

    if not rationale:
        rationale.append("Decision recorded from provided alternatives; rationale details should be expanded if needed.")

    return rationale[:8]


def collect_assumptions(text: str) -> List[str]:
    assumptions: List[str] = []
    for line in normalize_newlines(text).splitlines():
        stripped = compact_ws(line)
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(marker in lowered for marker in ASSUMPTION_MARKERS):
            assumptions.append(stripped)
    return assumptions[:8]


def collect_risks(text: str) -> List[str]:
    risks: List[str] = []
    for line in normalize_newlines(text).splitlines():
        stripped = compact_ws(line)
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(marker in lowered for marker in RISK_MARKERS):
            risks.append(stripped)
    return risks[:8]


def collect_validation(text: str, selected_option: str) -> List[Dict[str, Any]]:
    validations: List[Dict[str, Any]] = []
    for line in normalize_newlines(text).splitlines():
        stripped = compact_ws(line)
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(marker in lowered for marker in VALIDATION_MARKERS):
            validations.append({"check": stripped, "status": "pending"})

    if not validations:
        validations.append(
            {
                "check": f"Validate feasibility and impact of selected option {selected_option} before marking accepted.",
                "status": "pending",
            }
        )
    return validations[:8]


def derive_title(text: str, alternatives: List[Dict[str, Any]]) -> str:
    for line in normalize_newlines(text).splitlines():
        stripped = compact_ws(line)
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("decision:"):
            candidate = compact_ws(stripped.split(":", 1)[1])
            if candidate:
                return truncate_text(candidate, 120)
        if len(stripped) >= 10:
            return truncate_text(stripped, 120)

    if alternatives:
        return truncate_text(f"Choose among {len(alternatives)} alternatives", 120)
    return "Decision record"


def derive_context(text: str) -> str:
    return truncate_text(compact_ws(normalize_newlines(text)), 400)


def derive_decision_statement(selected_option: str, alternatives: List[Dict[str, Any]]) -> str:
    if selected_option == "undecided":
        return "Decision remains proposed with no final selection yet."

    for row in alternatives:
        if str(row["option_id"]) == selected_option:
            return f"Select {selected_option}: {row['title']}"

    return "Decision selection specified by option id."


def normalize_alternatives_for_hash(alternatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in alternatives:
        normalized.append(
            {
                "option_id": row.get("option_id"),
                "title": normalize_key_text(str(row.get("title", ""))),
                "pros": [normalize_key_text(str(item)) for item in row.get("pros", [])],
                "cons": [normalize_key_text(str(item)) for item in row.get("cons", [])],
            }
        )
    return normalized


def normalized_refs(refs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        "kg_refs": sorted(unique_keep_order(refs.get("kg_refs", []))),
        "tg_refs": sorted(unique_keep_order(refs.get("tg_refs", []))),
        "artifacts": sorted(unique_keep_order(refs.get("artifacts", []))),
    }


def compute_decision_payload_hash(candidate: Dict[str, Any]) -> str:
    subset = {
        "context": normalize_key_text(str(candidate.get("context", ""))),
        "decision": normalize_key_text(str(candidate.get("decision", ""))),
        "selected_option": str(candidate.get("selected_option", "")),
        "alternatives": normalize_alternatives_for_hash(candidate.get("alternatives", [])),
        "rationale": [normalize_key_text(str(item)) for item in candidate.get("rationale", [])],
        "refs": normalized_refs(candidate.get("refs", {})),
        "outcome_status": str(candidate.get("outcome", {}).get("status", "")),
    }
    return sha256_hex_obj(subset)


def get_decision_key(node: Dict[str, Any]) -> Optional[str]:
    value = node.get("decision_key")
    if isinstance(value, str) and value.strip():
        return value.strip()
    attrs = node.get("attributes")
    if isinstance(attrs, dict):
        nested = attrs.get("decision_key")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def get_decision_payload_hash(node: Dict[str, Any]) -> Optional[str]:
    attrs = node.get("attributes")
    if isinstance(attrs, dict):
        value = attrs.get("decision_payload_hash")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def get_selected_option(node: Dict[str, Any]) -> str:
    attrs = node.get("attributes")
    if isinstance(attrs, dict):
        selected = attrs.get("selected_option")
        if isinstance(selected, str) and selected.strip():
            return selected.strip()

    outcome = node.get("outcome")
    if isinstance(outcome, dict):
        selected = outcome.get("selected_option")
        if isinstance(selected, str) and selected.strip():
            return selected.strip()

    return "undecided"


def semantic_candidate_for_hash(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "context": candidate.get("context", ""),
        "decision": candidate.get("decision", ""),
        "selected_option": candidate.get("selected_option", "undecided"),
        "alternatives": candidate.get("alternatives", []),
        "rationale": candidate.get("rationale", []),
        "refs": candidate.get("refs", {"kg_refs": [], "tg_refs": [], "artifacts": []}),
        "outcome": candidate.get("outcome", {"status": "proposed"}),
    }


def resolve_scope(
    project_name: str,
    explicit_kg_refs: List[str],
    kg_latest: Dict[str, Dict[str, Any]],
) -> str:
    entity_id: Optional[str] = None
    for ref in explicit_kg_refs:
        node = kg_latest.get(ref)
        if node and str(node.get("type", "")) == "entity":
            entity_id = ref
            break

    base = f"project::{normalize_scope_fragment(project_name)}"
    if entity_id:
        return f"{base}::entity::{entity_id}"
    return base


def parse_refs(
    args: argparse.Namespace,
    decision_text: str,
    latest_layers: Dict[str, Dict[str, Dict[str, Any]]],
) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []

    explicit_kg_from_text, _explicit_rg_from_text, explicit_tg_from_text = parse_explicit_ids_from_text(decision_text)
    explicit_kg_from_args = parse_csv_ids(args.kg_refs, "KG")
    explicit_tg_from_args = parse_csv_ids(args.tg_refs, "TG")

    kg_refs = unique_keep_order(explicit_kg_from_args + explicit_kg_from_text)
    tg_refs = unique_keep_order(explicit_tg_from_args + explicit_tg_from_text)

    link_suggestions: List[Dict[str, Any]] = []

    if args.auto_link:
        kg_scored = lexical_auto_links(decision_text, latest_layers["KG"], "KG", max_links=8)
        tg_scored = lexical_auto_links(decision_text, latest_layers["TG"], "TG", max_links=8)

        for node_id, score in kg_scored:
            link_suggestions.append(
                {
                    "node_id": node_id,
                    "layer": "KG",
                    "confidence": round(score, 4),
                    "title": latest_layers["KG"].get(node_id, {}).get("title", ""),
                }
            )
            if node_id not in kg_refs:
                kg_refs.append(node_id)

        for node_id, score in tg_scored:
            link_suggestions.append(
                {
                    "node_id": node_id,
                    "layer": "TG",
                    "confidence": round(score, 4),
                    "title": latest_layers["TG"].get(node_id, {}).get("title", ""),
                }
            )
            if node_id not in tg_refs:
                tg_refs.append(node_id)

    refs = {
        "kg_refs": unique_keep_order(kg_refs),
        "tg_refs": unique_keep_order(tg_refs),
        "artifacts": [],
    }

    if not refs["kg_refs"]:
        warnings.append("No KG references resolved")
    if not refs["tg_refs"]:
        warnings.append("No TG references resolved")

    return refs, link_suggestions, warnings


def ensure_refs_exist(refs: Dict[str, List[str]], latest_layers: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    for kg in refs.get("kg_refs", []):
        if kg not in latest_layers["KG"]:
            raise CommitDecisionsError(f"KG reference not found: {kg}")
    for tg in refs.get("tg_refs", []):
        if tg not in latest_layers["TG"]:
            raise CommitDecisionsError(f"TG reference not found: {tg}")


def create_followup_task(
    decision_id: str,
    decision_key: str,
    refs: Dict[str, List[str]],
    now: str,
) -> Dict[str, Any]:
    task_id = f"TG-TASK-{new_ulid()}"
    return {
        "id": task_id,
        "type": "task",
        "title": "Finalize proposed decision selection",
        "description": "Resolve undecided option and mark RG decision outcome accepted/rejected explicitly.",
        "status": "ready",
        "priority": "high",
        "owner_role": "lead",
        "dependencies": [],
        "blocks": [],
        "acceptance_criteria": [
            "A concrete option_id is selected from decision alternatives",
            "RG outcome status is explicitly updated from proposed",
        ],
        "inputs": {
            "kg_refs": refs.get("kg_refs", []),
            "rg_refs": [decision_id],
            "sources": [],
        },
        "outputs": {
            "artifacts": [],
            "kg_updates": [],
            "rg_updates": [decision_id],
        },
        "log": [{"at": now, "event": "created", "by": "commit-decisions"}],
        "links": unique_keep_order([decision_id] + refs.get("kg_refs", []) + refs.get("tg_refs", [])),
        "tags": ["decision_followup"],
        "rev": 1,
        "created_at": now,
        "updated_at": now,
        "created_by": "commit-decisions",
        "origin": "decision_followup",
        "decision_key": decision_key,
    }


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
            raise CommitDecisionsError("Context history write validation failed")

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


def append_commit_row(
    paths: GraphPaths,
    commit_id: str,
    decision_commit_id: str,
    decision_seq: int,
    decision_id: str,
    decision_key: str,
    outcome_status: str,
    changed_node_ids: List[str],
    context_pack_path: str,
    source_hash_raw: str,
    source_hash_norm: str,
    request_id: Optional[str],
    dry_run: bool,
) -> None:
    row: Dict[str, Any] = {
        "commit_id": commit_id,
        "mode": "commit_decisions_apply",
        "decision_commit_id": decision_commit_id,
        "decision_seq": decision_seq,
        "decision_id": decision_id,
        "decision_key": decision_key,
        "outcome_status": outcome_status,
        "changed_node_ids": changed_node_ids,
        "context_pack_path": context_pack_path,
        "source_request_hash_raw": source_hash_raw,
        "source_request_hash_norm": source_hash_norm,
        "created_at": now_iso_utc(),
        "event": "post_response_commit",
    }
    if request_id:
        row["request_id"] = request_id

    append_jsonl_atomic(paths.post_commits, [row], dry_run)


def decision_commit_files(paths: GraphPaths) -> List[Path]:
    if not paths.decision_commits_dir.exists():
        return []
    return sorted(paths.decision_commits_dir.glob("decision_commit.*.json"))


def load_decision_seq_values(paths: GraphPaths) -> List[int]:
    values: List[int] = []
    for file_path in decision_commit_files(paths):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        seq = payload.get("decision_seq")
        if isinstance(seq, int) and seq >= 1:
            values.append(seq)
    return values


def rotate_decision_history(paths: GraphPaths, history_limit: int, dry_run: bool) -> None:
    files = decision_commit_files(paths)
    if len(files) <= history_limit:
        return
    to_remove = files[: len(files) - history_limit]
    for file_path in to_remove:
        if dry_run:
            print(f"[DRY-RUN] remove {file_path}")
        else:
            file_path.unlink(missing_ok=True)


def persist_decision_history(
    paths: GraphPaths,
    base_payload: Dict[str, Any],
    history_limit: int,
    dry_run: bool,
) -> Tuple[Path, str, int, Dict[str, Any]]:
    if not dry_run:
        paths.decision_commits_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(2):
        seq_values = load_decision_seq_values(paths)
        decision_seq = (max(seq_values) + 1) if seq_values else 1
        decision_commit_id = f"DECCMT-{new_ulid()}"

        payload = copy.deepcopy(base_payload)
        payload["decision_commit_id"] = decision_commit_id
        payload["decision_seq"] = decision_seq

        history_name = f"decision_commit.{timestamp_slug()}.{decision_commit_id}.json"
        history_path = paths.decision_commits_dir / history_name

        if dry_run:
            print(f"[DRY-RUN] write {history_path}")
            return history_path, decision_commit_id, decision_seq, payload

        if history_path.exists():
            continue

        write_json_atomic(history_path, payload, dry_run=False)
        written = read_json(history_path)
        if written is None or written.get("decision_commit_id") != decision_commit_id:
            history_path.unlink(missing_ok=True)
            raise CommitDecisionsError("Failed to validate written decision history file")

        seq_after = load_decision_seq_values(paths)
        if seq_after.count(decision_seq) == 1:
            rotate_decision_history(paths, history_limit=history_limit, dry_run=False)
            return history_path, decision_commit_id, decision_seq, payload

        history_path.unlink(missing_ok=True)
        if attempt == 0:
            continue
        raise CommitDecisionsError("decision_seq allocation conflict after retry")

    raise CommitDecisionsError("Unable to allocate decision_seq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Commit alternative selection decisions into RG")
    parser.add_argument("--project-root", required=True, help="Project root path")
    parser.add_argument("--graphs-dir", default="graphs", help="Graph directory under project root")

    parser.add_argument("--decision-text", default=None, help="Decision text input")
    parser.add_argument("--decision-file", default=None, help="Path to file containing decision text")
    parser.add_argument("--apply", type=parse_bool, default=False)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--outcome-status", choices=sorted(OUTCOME_STATUSES), default="proposed")
    parser.add_argument("--max-alternatives", type=int, default=12)
    parser.add_argument("--auto-link", type=parse_bool, default=True)
    parser.add_argument("--require-kg-tg", type=parse_bool, default=True)

    parser.add_argument("--selected-option", default=None)
    parser.add_argument("--alternatives-file", default=None)
    parser.add_argument("--kg-refs", default=None)
    parser.add_argument("--tg-refs", default=None)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--decision-key", default=None)
    parser.add_argument("--source-title", default=None)
    parser.add_argument("--source-key", default=None)
    parser.add_argument("--decision-history-limit", type=int, default=200)
    parser.add_argument("--context-pack-history-limit", type=int, default=200)
    parser.add_argument("--stdout-json", type=parse_bool, default=False)
    parser.add_argument("--strict-empty", type=parse_bool, default=False)
    parser.add_argument("--apply-followup-task", type=parse_bool, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_alternatives < 1 or args.max_alternatives > 50:
        print("[ERROR] --max-alternatives must be between 1 and 50")
        return 2

    if args.decision_history_limit < 1 or args.context_pack_history_limit < 1:
        print("[ERROR] history limits must be >= 1")
        return 2

    if args.apply_followup_task and not args.apply:
        print("[ERROR] --apply-followup-task true requires --apply true")
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
        decision_text_raw = parse_request_text(args)
    except CommitDecisionsError as exc:
        print(f"[ERROR] {exc}")
        return 2

    try:
        decision_text = normalize_newlines(decision_text_raw)
        source_request_hash_raw = sha256_hex_text(normalize_raw_for_hash(decision_text_raw))
        source_request_hash_norm = sha256_hex_text(normalize_for_dedupe(decision_text_raw))

        raw_layers = read_raw_layers(paths)
        latest_layers = {layer: resolve_latest_view(rows) for layer, rows in raw_layers.items()}

        parsed_alternatives = parse_alternatives_from_text(decision_text)
        if args.alternatives_file:
            parsed_alternatives = parse_alternatives_from_file(args.alternatives_file)

        alternatives = enrich_alternatives_with_option_ids(parsed_alternatives, args.max_alternatives)

        if args.apply and not alternatives:
            raise CommitDecisionsError("Apply mode requires at least one parsed alternative")

        if not alternatives and args.strict_empty:
            raise CommitDecisionsError("No alternatives found (--strict-empty true)")

        selected_option = resolve_selected_option(alternatives, args.selected_option, decision_text)

        if args.outcome_status == "accepted" and selected_option == "undecided":
            raise CommitDecisionsError("Cannot set outcome-status=accepted when selected option is undecided")

        refs, link_suggestions, link_warnings = parse_refs(args, decision_text, latest_layers)

        project_name = extract_project_name(paths)
        scope = resolve_scope(project_name, refs.get("kg_refs", []), latest_layers["KG"])

        title = derive_title(decision_text, alternatives)
        context = derive_context(decision_text)
        decision_stmt = derive_decision_statement(selected_option, alternatives)
        rationale = collect_rationale(decision_text, selected_option, alternatives)
        assumptions = collect_assumptions(decision_text)
        risks = collect_risks(decision_text)
        validation = collect_validation(decision_text, selected_option)

        outcome: Dict[str, Any] = {
            "status": args.outcome_status,
            "supersedes": [],
            "notes": "Decision recorded by commit-decisions.",
        }

        decision_key = args.decision_key.strip() if args.decision_key and args.decision_key.strip() else f"decision::{scope}::{normalize_key_text(title)}"

        decision_candidate: Dict[str, Any] = {
            "decision_key": decision_key,
            "title": title,
            "context": context,
            "decision": decision_stmt,
            "alternatives": alternatives,
            "selected_option": selected_option,
            "rationale": rationale,
            "assumptions": assumptions,
            "risks": risks,
            "validation": validation,
            "refs": refs,
            "links": unique_keep_order(refs.get("kg_refs", []) + refs.get("tg_refs", [])),
            "tags": unique_keep_order(["decision", "alternatives", f"outcome:{args.outcome_status}"]),
            "outcome": outcome,
        }

        payload_hash = compute_decision_payload_hash(decision_candidate)
        decision_candidate["decision_payload_hash"] = payload_hash

        source_obj: Dict[str, Any] = {
            "source_key": args.source_key.strip() if args.source_key and args.source_key.strip() else "decision::default",
            "source_digest": source_request_hash_raw,
            "source_kind": "decision_text" if args.decision_text else "decision_file",
            "source_title": args.source_title.strip() if args.source_title and args.source_title.strip() else "Decision input",
        }

        warnings: List[str] = []
        warnings.extend(link_warnings)

        if selected_option == "undecided":
            warnings.append("Selected option unresolved; outcome remains proposed unless explicitly changed")

        followup_task_suggestion: Optional[Dict[str, Any]] = None
        if selected_option == "undecided":
            followup_task_suggestion = {
                "title": "Finalize proposed decision selection",
                "description": "Select concrete option_id and update decision outcome explicitly.",
                "tags": ["decision_followup"],
            }

        base_payload: Dict[str, Any] = {
            "generated_at": now_iso_utc(),
            "source": source_obj,
            "decision_candidate": decision_candidate,
            "warnings": warnings,
            "stats": {
                "alternatives_parsed": len(alternatives),
                "link_suggestions": len(link_suggestions),
                "apply_requested": bool(args.apply),
                "selected_option": selected_option,
            },
            "source_request_hash_raw": source_request_hash_raw,
            "source_request_hash_norm": source_request_hash_norm,
            "link_suggestions": link_suggestions,
        }

        if args.request_id:
            base_payload["request_id"] = args.request_id
        if followup_task_suggestion is not None:
            base_payload["followup_task_suggestion"] = followup_task_suggestion

        history_path, decision_commit_id, decision_seq, payload = persist_decision_history(
            paths=paths,
            base_payload=base_payload,
            history_limit=args.decision_history_limit,
            dry_run=args.dry_run,
        )

        payload["decision_commit_id"] = decision_commit_id
        payload["decision_seq"] = decision_seq

        changed_rg_rows: List[Dict[str, Any]] = []
        changed_tg_rows: List[Dict[str, Any]] = []
        changed_node_ids: List[str] = []
        decision_id: Optional[str] = None

        if args.apply:
            # Apply-time strictness only.
            if args.require_kg_tg:
                if not refs.get("kg_refs"):
                    raise CommitDecisionsError("Apply requires non-empty KG refs (--require-kg-tg true)")
                if not refs.get("tg_refs"):
                    raise CommitDecisionsError("Apply requires non-empty TG refs (--require-kg-tg true)")

            ensure_refs_exist(refs, latest_layers)

            rg_latest = {k: strip_internal(v) for k, v in latest_layers["RG"].items()}

            existing_by_key: Dict[str, Dict[str, Any]] = {}
            for node in rg_latest.values():
                if str(node.get("type", "")) != "decision":
                    continue
                key = get_decision_key(node)
                if not key:
                    continue
                existing_by_key[key] = node

            now = now_iso_utc()
            existing = existing_by_key.get(decision_key)

            if existing is not None:
                decision_id = str(existing.get("id"))
                prev_hash = get_decision_payload_hash(existing)
                prev_rev = int(existing.get("rev", 1))
                prev_selected = get_selected_option(existing)

                if prev_hash == payload_hash:
                    warnings.append("Decision semantic payload unchanged; RG write skipped")
                else:
                    outcome_copy = copy.deepcopy(outcome)
                    if prev_selected != selected_option:
                        outcome_copy["notes"] = f"supersedes prior choice in rev {prev_rev}"
                        outcome_copy["supersedes_rev"] = prev_rev

                    row = {
                        "id": decision_id,
                        "type": "decision",
                        "title": title,
                        "context": context,
                        "decision": decision_stmt,
                        "rationale": rationale,
                        "alternatives": alternatives,
                        "assumptions": assumptions,
                        "risks": risks,
                        "validation": validation,
                        "outcome": outcome_copy,
                        "links": unique_keep_order(refs.get("kg_refs", []) + refs.get("tg_refs", [])),
                        "refs": refs,
                        "tags": unique_keep_order(["decision", "alternatives", f"outcome:{args.outcome_status}"]),
                        "decision_key": decision_key,
                        "attributes": {
                            "decision_key": decision_key,
                            "decision_payload_hash": payload_hash,
                            "selected_option": selected_option,
                            "source_request_hash_raw": source_request_hash_raw,
                            "source_request_hash_norm": source_request_hash_norm,
                        },
                        "rev": prev_rev + 1,
                        "created_at": str(existing.get("created_at", now)),
                        "updated_at": now,
                        "created_by": "commit-decisions",
                        "origin": "decision_commit",
                        "origin_decision_commit_id": decision_commit_id,
                        "decision_seq": decision_seq,
                    }
                    if args.request_id:
                        row["request_id"] = args.request_id

                    changed_rg_rows.append(row)
                    changed_node_ids.append(decision_id)
            else:
                decision_id = f"RG-DEC-{new_ulid()}"
                row = {
                    "id": decision_id,
                    "type": "decision",
                    "title": title,
                    "context": context,
                    "decision": decision_stmt,
                    "rationale": rationale,
                    "alternatives": alternatives,
                    "assumptions": assumptions,
                    "risks": risks,
                    "validation": validation,
                    "outcome": outcome,
                    "links": unique_keep_order(refs.get("kg_refs", []) + refs.get("tg_refs", [])),
                    "refs": refs,
                    "tags": unique_keep_order(["decision", "alternatives", f"outcome:{args.outcome_status}"]),
                    "decision_key": decision_key,
                    "attributes": {
                        "decision_key": decision_key,
                        "decision_payload_hash": payload_hash,
                        "selected_option": selected_option,
                        "source_request_hash_raw": source_request_hash_raw,
                        "source_request_hash_norm": source_request_hash_norm,
                    },
                    "rev": 1,
                    "created_at": now,
                    "updated_at": now,
                    "created_by": "commit-decisions",
                    "origin": "decision_commit",
                    "origin_decision_commit_id": decision_commit_id,
                    "decision_seq": decision_seq,
                }
                if args.request_id:
                    row["request_id"] = args.request_id

                changed_rg_rows.append(row)
                changed_node_ids.append(decision_id)

            if decision_id and args.apply_followup_task and selected_option == "undecided":
                task_row = create_followup_task(decision_id=decision_id, decision_key=decision_key, refs=refs, now=now_iso_utc())
                changed_tg_rows.append(task_row)
                changed_node_ids.append(str(task_row["id"]))

            for row in changed_rg_rows:
                for field in ("created_at", "updated_at"):
                    value = row.get(field)
                    if not isinstance(value, str) or ISO_PATTERN.match(value) is None:
                        raise CommitDecisionsError(f"RG row invalid {field}")
            for row in changed_tg_rows:
                for field in ("created_at", "updated_at"):
                    value = row.get(field)
                    if not isinstance(value, str) or ISO_PATTERN.match(value) is None:
                        raise CommitDecisionsError(f"TG row invalid {field}")

            append_jsonl_atomic(paths.rg_nodes, changed_rg_rows, args.dry_run)
            append_jsonl_atomic(paths.tg_nodes, changed_tg_rows, args.dry_run)

            rebuild_indexes(paths, args.dry_run)
            if not args.dry_run:
                verify_indexes(paths)

            payload["stats"]["rg_rows_written"] = len(changed_rg_rows)
            payload["stats"]["tg_rows_written"] = len(changed_tg_rows)

            if changed_node_ids and decision_id:
                commit_id = f"COMMIT-{new_ulid()}"
                context_pack = build_post_response_context_pack(
                    paths=paths,
                    commit_id=commit_id,
                    changed_node_ids=changed_node_ids,
                    goal_hint=title,
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
                        decision_commit_id=decision_commit_id,
                        decision_seq=decision_seq,
                        decision_id=decision_id,
                        decision_key=decision_key,
                        outcome_status=args.outcome_status,
                        changed_node_ids=changed_node_ids,
                        context_pack_path=context_pack_path,
                        source_hash_raw=source_request_hash_raw,
                        source_hash_norm=source_request_hash_norm,
                        request_id=args.request_id,
                        dry_run=args.dry_run,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    raise CommitDecisionsError(
                        "Context pack updated but commit row append failed. "
                        "Re-run commit-decisions with --apply true to backfill post_response_commits. "
                        f"Original error: {exc}"
                    ) from exc

                payload["commit_id"] = commit_id
                payload["context_pack_path"] = context_pack_path
                payload["decision_id"] = decision_id
                payload["decision_key"] = decision_key
            else:
                warnings.append("Apply mode produced no RG/TG changes; commit/context sync skipped")
                payload["stats"]["rg_rows_written"] = 0
                payload["stats"]["tg_rows_written"] = 0

        payload["warnings"] = warnings

        if args.stdout_json:
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(f"[OK] decision_commit_id: {decision_commit_id}")
            print(f"[OK] decision_seq: {decision_seq}")
            print(f"[OK] history: {history_path}")
            print(f"[OK] alternatives: {len(alternatives)}")
            print(f"[OK] selected_option: {selected_option}")
            if args.apply:
                print(f"[OK] rg_rows_written: {payload.get('stats', {}).get('rg_rows_written', 0)}")
                print(f"[OK] tg_rows_written: {payload.get('stats', {}).get('tg_rows_written', 0)}")
                if payload.get("decision_id"):
                    print(f"[OK] decision_id: {payload['decision_id']}")
                if payload.get("decision_key"):
                    print(f"[OK] decision_key: {payload['decision_key']}")
                if payload.get("commit_id"):
                    print(f"[OK] commit_id: {payload['commit_id']}")
                if payload.get("context_pack_path"):
                    print(f"[OK] context_pack_path: {payload['context_pack_path']}")
            if warnings:
                for warning in warnings:
                    print(f"[WARN] {warning}")

        return 0

    except CommitDecisionsError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] unexpected failure: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
