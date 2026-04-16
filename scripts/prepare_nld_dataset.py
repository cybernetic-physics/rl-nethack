#!/usr/bin/env python3
"""
Prepare a local NLD dataset root from one or more already-downloaded zip archives.

This does not fetch remote data. It assumes the user has placed one or more
NLD zip shards on disk and wants them extracted, validated, and optionally
registered with `nle.dataset`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.nld_dataset_prep import (
    discover_altorg_roots,
    extract_zip_archives,
    summarize_altorg_root,
)
from src.nld_long_sequence_import import register_nld_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a local NLD dataset root")
    parser.add_argument("--zip", action="append", default=[], help="Local NLD zip archive; pass multiple times")
    parser.add_argument("--extract-dir", required=True, help="Directory to extract archives into")
    parser.add_argument("--dataset-name", default="nld-aa-local", help="Dataset registration name")
    parser.add_argument("--dbfilename", default="ttyrecs.db", help="SQLite metadata DB used by nle.dataset")
    parser.add_argument("--dataset-type", default="altorg", choices=["altorg", "nledata"], help="Dataset registration type")
    parser.add_argument("--register", action="store_true", help="Register the discovered root with nle.dataset")
    parser.add_argument("--root-path", default=None, help="Explicit extracted root to validate/register instead of discovery")
    args = parser.parse_args()

    result: dict[str, object] = {}
    if args.zip:
        result["extraction"] = extract_zip_archives(args.zip, args.extract_dir)

    if args.root_path:
        candidate_roots = [args.root_path]
    else:
        candidate_roots = discover_altorg_roots(args.extract_dir)
    result["candidate_roots"] = candidate_roots

    if not candidate_roots:
        print(json.dumps(result, indent=2))
        print("ERROR: no altorg-style dataset root found after extraction", file=sys.stderr)
        return 1

    chosen_root = candidate_roots[0]
    result["selected_root"] = chosen_root
    result["summary"] = summarize_altorg_root(chosen_root)

    if args.register:
        result["registration"] = register_nld_directory(
            chosen_root,
            args.dataset_name,
            dataset_type=args.dataset_type,
            dbfilename=args.dbfilename,
        )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
