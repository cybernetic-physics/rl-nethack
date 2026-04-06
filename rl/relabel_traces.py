from __future__ import annotations

import argparse
import json

import numpy as np

from rl.feature_encoder import ACTION_SET
from rl.io_utils import atomic_write_text
from rl.train_bc import _normalize_teacher_paths, _teacher_logits_for_rows, load_trace_rows


def relabel_trace_actions(
    input_path: str,
    output_path: str,
    *,
    bc_model_path: str,
) -> dict:
    rows = load_trace_rows(input_path)
    teacher_paths = _normalize_teacher_paths(bc_model_path, None)
    if not teacher_paths:
        raise ValueError("At least one bc_model_path is required")
    ensembled_logits = _teacher_logits_for_rows(rows, teacher_paths)
    predicted_actions = [ACTION_SET[int(np.argmax(row_logits))] for row_logits in ensembled_logits]

    relabeled_rows = []
    changed = 0
    action_counts: dict[str, int] = {}
    for row, predicted_action in zip(rows, predicted_actions):
        relabeled = dict(row)
        relabeled["original_action"] = row.get("action")
        relabeled["action"] = predicted_action
        relabeled_rows.append(relabeled)
        changed += int(predicted_action != row.get("action"))
        action_counts[predicted_action] = action_counts.get(predicted_action, 0) + 1

    atomic_write_text(output_path, "".join(json.dumps(row) + "\n" for row in relabeled_rows))
    return {
        "input_path": input_path,
        "output_path": output_path,
        "teacher_bc_model_path": bc_model_path,
        "teacher_bc_model_paths": teacher_paths,
        "rows": len(relabeled_rows),
        "changed_rows": changed,
        "changed_rate": round(changed / max(1, len(relabeled_rows)), 4),
        "action_counts": action_counts,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Relabel trace actions with a BC teacher")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--bc-model-path", type=str, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = relabel_trace_actions(args.input, args.output, bc_model_path=args.bc_model_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
