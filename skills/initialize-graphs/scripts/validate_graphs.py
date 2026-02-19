#!/usr/bin/env python3
"""Validate graph store invariants and report errors/warnings/info."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from init_graphs import GraphPaths, validate_graph_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate KG/RG/TG graph store invariants.")
    parser.add_argument("--project-root", required=True, help="Absolute or relative project root path")
    parser.add_argument("--graphs-dir", default="graphs", help="Graph directory under project root")
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help="Treat missing artifact:// files as warnings",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview checks without mutation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.exists() or not project_root.is_dir():
        print(f"[ERROR] project root not found: {project_root}")
        return 2

    if args.dry_run:
        print("[DRY-RUN] would validate JSONL schema invariants, link integrity, index drift, and context freshness")

    paths = GraphPaths(project_root=project_root, graphs_dir=(project_root / args.graphs_dir).resolve())

    try:
        errors, warnings, infos = validate_graph_store(paths=paths, strict_artifacts=args.strict_artifacts)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}")
        return 1

    for message in infos:
        print(f"[INFO] {message}")
    for message in warnings:
        print(f"[WARN] {message}")
    for message in errors:
        print(f"[ERROR] {message}")

    if errors:
        print(f"[FAIL] validation failed with {len(errors)} error(s)")
        return 1

    print(f"[OK] validation passed ({len(warnings)} warning(s), {len(infos)} info message(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
