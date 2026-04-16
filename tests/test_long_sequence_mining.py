import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.long_sequence_mining import build_gold_wins_rows, compute_episode_score


def make_row(episode_id: str, *, is_win: bool, achieve: int = 0, maxlvl: int = 0, turns: int = 0):
    return {
        "conversations": [],
        "metadata": {
            "source_episode_id": episode_id,
            "is_win": is_win,
            "achieve": achieve,
            "maxlvl": maxlvl,
            "turns": turns,
        },
    }


def test_compute_episode_score_prefers_wins_then_depth():
    losing = [make_row("a", is_win=False, maxlvl=20, turns=1000)]
    winning = [make_row("b", is_win=True, maxlvl=10, turns=500)]
    assert compute_episode_score(winning) > compute_episode_score(losing)


def test_build_gold_wins_rows_ranks_episodes():
    rows = [
        make_row("loss-deep", is_win=False, maxlvl=25, turns=2000),
        make_row("win-shallow", is_win=True, maxlvl=5, turns=500),
        make_row("win-deep", is_win=True, maxlvl=20, turns=1500),
    ]
    selected = build_gold_wins_rows(rows, max_episodes=2)
    episode_ids = [row["metadata"]["source_episode_id"] for row in selected]
    assert "win-deep" in episode_ids
    assert "win-shallow" in episode_ids
    assert "loss-deep" not in episode_ids
