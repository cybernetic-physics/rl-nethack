#!/usr/bin/env python3
"""Build selective preference or weighted-SFT datasets from long-sequence rows."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.long_sequence_preferences import (
    build_kto_style_rows,
    build_pairwise_preference_rows,
    build_weighted_sft_rows,
    load_long_sequence_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a selective positive/negative dataset from long-sequence rows")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positive-limit", type=int, default=None)
    parser.add_argument("--negative-limit", type=int, default=None)
    parser.add_argument("--min-repeat-run", type=int, default=3)
    parser.add_argument("--teacher-margin-threshold", type=float, default=0.0)
    parser.add_argument(
        "--output-format",
        choices=("kto", "weighted-sft", "pairwise"),
        default="kto",
        help="Output schema: KTO-style preference rows or ShareGPT weighted-SFT rows",
    )
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=-0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_long_sequence_rows(args.input)
    if args.output_format == "weighted-sft":
        output_rows = build_weighted_sft_rows(
            rows,
            positive_limit=args.positive_limit,
            negative_limit=args.negative_limit,
            min_repeat_run=args.min_repeat_run,
            teacher_margin_threshold=args.teacher_margin_threshold,
            positive_weight=args.positive_weight,
            negative_weight=args.negative_weight,
        )
    elif args.output_format == "pairwise":
        output_rows = build_pairwise_preference_rows(
            rows,
            negative_limit=args.negative_limit,
            teacher_margin_threshold=args.teacher_margin_threshold,
        )
    else:
        output_rows = build_kto_style_rows(
            rows,
            positive_limit=args.positive_limit,
            negative_limit=args.negative_limit,
            min_repeat_run=args.min_repeat_run,
            teacher_margin_threshold=args.teacher_margin_threshold,
        )
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")
    if args.output_format == "weighted-sft":
        positives = sum(1 for row in output_rows if float(row.get("sample_weight", 0.0)) > 0.0)
        negatives = sum(1 for row in output_rows if float(row.get("sample_weight", 0.0)) < 0.0)
    elif args.output_format == "pairwise":
        positives = len(output_rows)
        negatives = len(output_rows)
    else:
        positives = sum(1 for row in output_rows if row["label"])
        negatives = sum(1 for row in output_rows if not row["label"])
    print(
        json.dumps(
            {
                "rows": len(output_rows),
                "positives": positives,
                "negatives": negatives,
                "output_format": args.output_format,
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
