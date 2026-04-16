#!/usr/bin/env python3
"""Back-convert long-sequence rows into episode-style rows."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.long_sequence_backconvert import extract_episode_rows_from_long_sequence_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract episode rows from a long-sequence JSONL corpus")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = extract_episode_rows_from_long_sequence_path(args.input, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
