#!/usr/bin/env python3
"""Select a metadata-filtered subset from long-sequence JSONL."""

from __future__ import annotations

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Select a metadata-filtered subset from long-sequence JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-equals", nargs="*", default=None,
                        help="Metadata equality filters as key=value")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sort-by", type=str, default=None,
                        help="Optional metadata field to sort descending before applying limit")
    return parser.parse_args()


def parse_filters(raw_filters):
    filters = {}
    for item in raw_filters or []:
        if "=" not in item:
            raise ValueError(f"Invalid filter {item!r}; expected key=value")
        key, value = item.split("=", 1)
        filters[key] = value
    return filters


def normalize(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def main():
    args = parse_args()
    filters = parse_filters(args.metadata_equals)
    rows = []
    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            if any(normalize(metadata.get(k)) != v for k, v in filters.items()):
                continue
            rows.append(row)

    if args.sort_by:
        rows.sort(key=lambda row: row.get("metadata", {}).get(args.sort_by, 0), reverse=True)

    if args.limit is not None:
        rows = rows[: args.limit]

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(json.dumps({"selected_rows": len(rows), "output": args.output, "filters": filters}, indent=2))


if __name__ == "__main__":
    main()
