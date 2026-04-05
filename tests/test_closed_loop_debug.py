import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.closed_loop_debug import (
    build_golden_episode,
    evaluate_golden_episode,
    load_golden_episode,
)
from src.state_encoder import StateEncoder


class TestGoldenEpisode:
    def setup_method(self):
        self.encoder = StateEncoder()

    def test_build_and_load_golden_episode(self, tmp_path):
        out = tmp_path / "golden.jsonl"
        result = build_golden_episode(
            seed=42,
            max_steps=3,
            encoder=self.encoder,
            output_path=str(out),
        )
        assert result["examples"] > 0
        assert out.is_file()

        rows = load_golden_episode(str(out))
        assert len(rows) == result["examples"]
        first = rows[0]
        assert "messages" in first
        assert "prompt" in first
        assert "target" in first
        assert "obs_hash" in first
        assert "next_obs_hash" in first
        assert "message_hash" in first

    def test_evaluate_golden_episode_no_server(self, tmp_path):
        out = tmp_path / "golden.jsonl"
        build_golden_episode(
            seed=42,
            max_steps=2,
            encoder=self.encoder,
            output_path=str(out),
        )
        result = evaluate_golden_episode(
            path=str(out),
            server_url="http://127.0.0.1:19999",
        )
        assert result["server_available"] is False
        assert "accuracy" in result
        assert isinstance(result["comparisons"], list)
