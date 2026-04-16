import json

from src.long_sequence_corpus import build_token_budgeted_corpus


def _row(episode_id, step_index, *, tokens, outcome="unknown", is_win=False, depth=1, source="test"):
    return {
        "conversations": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ],
        "metadata": {
            "episode_id": episode_id,
            "step_index": step_index,
            "context_tokens_estimate": tokens,
            "target_context_tokens": tokens,
            "outcome": outcome,
            "is_win": is_win,
            "depth": depth,
            "turn": step_index,
            "source": source,
        },
    }


def test_build_token_budgeted_corpus_prioritizes_full_wins_and_stride(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    manifest_path = tmp_path / "manifest.json"

    rows = []
    for step in range(4):
        rows.append(_row("win_ep", step, tokens=80_000, outcome="win", is_win=True, depth=12))
    for step in range(6):
        rows.append(_row("loss_ep", step, tokens=40_000, outcome="loss", is_win=False, depth=9))
    for step in range(10):
        rows.append(_row("medium_ep", step, tokens=8_000, outcome="unknown", is_win=False, depth=3))

    with input_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    result = build_token_budgeted_corpus(
        [str(input_path)],
        output_path=str(output_path),
        manifest_path=str(manifest_path),
        target_tokens=300_000,
        full_episode_fraction=0.40,
        very_long_fraction=0.35,
        long_fraction=0.20,
        medium_fraction=0.05,
        very_long_min_tokens=65_536,
        long_min_tokens=16_384,
        medium_min_tokens=4_096,
        very_long_stride=2,
        long_stride=2,
        medium_stride=3,
    )

    selected_rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    selected_episode_ids = {row["metadata"]["episode_id"] for row in selected_rows}
    selected_tiers = {row["metadata"]["corpus_sampling_tier"] for row in selected_rows}

    assert result["selected_rows"] == len(selected_rows)
    assert "win_ep" in selected_episode_ids
    assert "full_episode" in selected_tiers
    assert "long" in selected_tiers
    assert result["tier_selected_tokens"]["full_episode"] >= 0
    assert result["selected_tokens"] > 0

    full_win_rows = [
        row for row in selected_rows
        if row["metadata"]["episode_id"] == "win_ep"
        and row["metadata"]["corpus_sampling_tier"] == "full_episode"
    ]
    assert len(full_win_rows) == 4


def test_build_token_budgeted_corpus_writes_manifest(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    rows = [
        _row("win_ep", 0, tokens=70_000, outcome="win", is_win=True, depth=10),
        _row("win_ep", 1, tokens=71_000, outcome="win", is_win=True, depth=10),
        _row("loss_ep", 0, tokens=20_000, outcome="loss", depth=8),
        _row("other_ep", 0, tokens=6_000, outcome="unknown", depth=2),
    ]
    with input_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    result = build_token_budgeted_corpus(
        [str(input_path)],
        output_path=str(output_path),
        target_tokens=100_000,
    )
    manifest = json.loads(open(result["manifest_path"], "r").read())
    assert manifest["target_tokens"] == 100_000
    assert manifest["selected_rows"] == result["selected_rows"]
    assert manifest["selected_tokens"] == result["selected_tokens"]
    assert "win" in manifest["outcome_selected_tokens"]
