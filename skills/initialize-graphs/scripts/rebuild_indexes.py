#!/usr/bin/env python3
"""Rebuild or verify derived KG/RG/TG index files from append-only node logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from init_graphs import GraphPaths, rebuild_indexes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild graph indexes from node logs.")
    parser.add_argument("--project-root", required=True, help="Absolute or relative project root path")
    parser.add_argument("--graphs-dir", default="graphs", help="Graph directory under project root")
    parser.add_argument("--verify-only", action="store_true", help="Do not write indexes; only detect drift")
    parser.add_argument("--dry-run", action="store_true", help="Preview write operations")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.exists() or not project_root.is_dir():
        print(f"[ERROR] project root not found: {project_root}")
        return 2

    paths = GraphPaths(project_root=project_root, graphs_dir=(project_root / args.graphs_dir).resolve())
    try:
        drift, counts = rebuild_indexes(paths=paths, verify_only=args.verify_only, dry_run=args.dry_run)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}")
        return 1

    if args.verify_only:
        if drift:
            print("[FAIL] index drift detected")
            return 1
        print("[OK] indexes match derived view")
        return 0

    print("[OK] indexes rebuilt")
    print(f"[OK] counts: KG={counts.get('KG', 0)} RG={counts.get('RG', 0)} TG={counts.get('TG', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
