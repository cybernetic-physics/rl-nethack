from __future__ import annotations

import json
import os
import tempfile

from rl.sequence_world_model import load_sequence_world_model
from rl.sequence_world_model_dataset import build_sequence_world_model_examples, sequence_examples_to_arrays
from rl.sequence_world_model_eval import evaluate_sequence_world_model
from rl.sequence_model_selection import summarize_sequence_model_reports
from rl.sequence_benchmark import summarize_sequence_benchmark
from rl.sequence_planner_eval import evaluate_sequence_planner_replay
from rl.sequence_planner_multiseed import evaluate_sequence_planner_multiseed
from rl.sequence_planner_tune import tune_sequence_planner_replay
from rl.train_sequence_world_model import train_sequence_world_model
from rl.world_model_calibration import expected_calibration_error, fit_temperature_for_binary_logits
from rl.world_model_planner import plan_action_sequence_with_cem, score_action_candidates


def _toy_rows():
    rows = []
    for episode in range(3):
        for step in range(8):
            rows.append(
                {
                    "episode_id": f"ep{episode}",
                    "step": step,
                    "seed": episode,
                    "task": "explore",
                    "action": "east" if step % 2 == 0 else "west",
                    "allowed_actions": ["east", "west", "wait"],
                    "reward": 1.0 if step % 2 == 0 else 0.0,
                    "done": step == 7,
                    "observation_version": "v4",
                    "prompt": f"episode {episode} step {step}",
                    "planner_trace": [
                        {"action": "east", "total": 1.0},
                        {"action": "west", "total": 0.0},
                        {"action": "wait", "total": 0.1},
                    ],
                    "feature_vector": [
                        float(step),
                        float(step % 2),
                        float((step + 1) % 2),
                        1.0 if step % 2 == 0 else 0.0,
                        0.0 if step % 2 == 0 else 1.0,
                    ],
                }
            )
    return rows


def test_sequence_world_model_examples_build_windows():
    examples = build_sequence_world_model_examples(_toy_rows(), context_len=3, rollout_horizon=2, observation_version="v4")
    assert len(examples) > 0
    arrays = sequence_examples_to_arrays(examples)
    assert arrays["features"].shape[1:] == (5, 5)
    assert arrays["actions"].shape[1] == 4
    assert len(arrays["prompts"][0]) == 5
    assert arrays["rollout_start_indices"][0] == 2
    assert arrays["values"].shape[1] == 2
    assert arrays["planner_action_scores"].shape[2] == 13


def test_sequence_world_model_train_eval_and_plan_smoke():
    rows = _toy_rows()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "sequence_world_model.pt")
        trace_path = os.path.join(tmpdir, "trace.jsonl")
        with open(trace_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        result = train_sequence_world_model(
            rows,
            model_path,
            context_len=3,
            rollout_horizon=2,
            epochs=3,
            hidden_size=32,
            latent_dim=16,
            observation_version="v4",
            text_encoder_backend="hash",
            text_embedding_dim=16,
            overshooting_loss_coef=0.1,
            overshooting_distance=2,
            planner_policy_loss_coef=0.1,
            planner_policy_warmup_epochs=2,
            adaptive_loss_balance=True,
            selection_metric="planner_compromise_proxy",
        )
        assert result["model_type"] == "sequence_world_model"
        assert result["num_train_examples"] > 0
        assert "val_summary" in result
        assert result["planner_policy_warmup_epochs"] == 2
        assert result["adaptive_loss_balance"] is True
        assert result["selection_metric"] == "planner_compromise_proxy"
        assert "feature" in result["loss_balance_log_vars"]

        inference = load_sequence_world_model(model_path)
        observed = inference.observe([rows[0]["feature_vector"]], prompt_texts=[rows[0]["prompt"]])
        assert observed["latent"].shape[0] == 1
        assert observed["planner_action_logits"].shape[1] == 13
        rollout = inference.rollout(rows[0]["feature_vector"], [0, 2], prompt_text=rows[0]["prompt"])
        assert rollout["pred_features"].shape == (2, 5)
        assert rollout["pred_latent_uncertainty"].shape == (2,)
        sampled_rollout = inference.rollout(
            rows[0]["feature_vector"],
            [0, 2],
            prompt_text=rows[0]["prompt"],
            deterministic=False,
            num_samples=3,
        )
        assert sampled_rollout["reward_disagreement"].shape == (2,)
        assert sampled_rollout["value_disagreement"].shape == (2,)
        assert sampled_rollout["num_samples"] == 3

        eval_result = evaluate_sequence_world_model(
            model_path,
            trace_path,
            context_len=3,
            rollout_horizon=2,
            observation_version="v4",
        )
        assert eval_result["num_examples"] > 0
        assert "horizon_metrics" in eval_result
        assert eval_result["done_temperature"] > 0.0
        assert "overshooting_kl_mean" in eval_result
        assert "value_mae" in eval_result
        assert "planner_action_mae" in eval_result
        assert "planner_policy_ce" in eval_result
        assert "planner_policy_top1" in eval_result
        assert eval_result["planner_policy_target_temperature"] > 0.0
        assert "planner_pairwise_loss" in eval_result
        assert "planner_pairwise_accuracy" in eval_result

        plan = plan_action_sequence_with_cem(
            inference,
            rows[0]["feature_vector"],
            prompt_text=rows[0]["prompt"],
            planning_horizon=3,
            population_size=16,
            iterations=2,
            uncertainty_coef=0.25,
            disagreement_coef=0.1,
            rollout_samples=3,
        )
        assert len(plan["best_actions"]) == 3
        assert isinstance(plan["best_score"], float)
        assert plan["uncertainty_coef"] == 0.25
        assert plan["disagreement_coef"] == 0.1
        assert plan["rollout_samples"] == 3
        scored = score_action_candidates(
            inference,
            rows[0]["feature_vector"],
            ["east", "west", "wait"],
            prompt_text=rows[0]["prompt"],
            allowed_actions=rows[0]["allowed_actions"],
            planning_horizon=3,
            population_size=8,
            iterations=1,
            uncertainty_coef=0.25,
            disagreement_coef=0.1,
            rollout_samples=3,
        )
        assert [row["action"] for row in scored]
        planner_eval = evaluate_sequence_planner_replay(
            model_path,
            trace_path,
            planning_horizon=3,
            population_size=8,
            iterations=1,
            max_rows=4,
            root_prior_coef=0.5,
            uncertainty_coef=0.25,
            disagreement_coef=0.1,
            rollout_samples=3,
        )
        assert planner_eval["rows_evaluated"] > 0
        assert "mean_rank_correlation" in planner_eval
        assert planner_eval["root_prior_coef"] == 0.5
        assert planner_eval["uncertainty_coef"] == 0.25
        assert planner_eval["disagreement_coef"] == 0.1
        assert planner_eval["rollout_samples"] == 3
        planner_tune = tune_sequence_planner_replay(
            model_path,
            trace_path,
            planning_horizon=3,
            population_size=8,
            iterations=1,
            max_rows=4,
            bootstrap_value_coefs=[0.25, 0.5],
            root_prior_coefs=[0.0, 0.5],
            uncertainty_coefs=[0.0, 0.25],
            disagreement_coefs=[0.0, 0.1],
            rollout_samples=3,
        )
        assert planner_tune["best"] is not None
        assert len(planner_tune["trials"]) == 16
        planner_multiseed = evaluate_sequence_planner_multiseed(
            model_path,
            trace_path,
            planning_horizon=3,
            population_size=8,
            iterations=1,
            max_rows=4,
            seeds=[1, 2],
            bootstrap_value_coefs=[0.25],
            root_prior_coefs=[0.0, 0.5],
            uncertainty_coefs=[0.0, 0.25],
            disagreement_coefs=[0.0, 0.1],
            rollout_samples=3,
        )
        assert planner_multiseed["summary"]["mean_rank_correlation"] >= -1.0
        assert len(planner_multiseed["per_seed"]) == 2


def test_binary_temperature_scaling_helpers():
    labels = [0, 0, 1, 1]
    logits = [-2.0, -0.5, 0.4, 2.0]
    ece = expected_calibration_error(labels, [0.1, 0.4, 0.6, 0.9])
    fit = fit_temperature_for_binary_logits(labels, logits)
    assert ece >= 0.0
    assert fit["temperature"] > 0.0


def test_sequence_model_selection_summary(tmp_path):
    report_a = tmp_path / "a_train.json"
    report_b = tmp_path / "b_train.json"
    report_a.write_text(json.dumps({"val_summary": {"feature_mse": 0.2, "reward_mae": 0.3, "value_mae": 0.4, "planner_action_mae": 0.5, "planner_policy_ce": 1.2, "planner_pairwise_loss": 0.9}}))
    report_b.write_text(json.dumps({"val_summary": {"feature_mse": 0.1, "reward_mae": 0.4, "value_mae": 0.2, "planner_action_mae": 0.6, "planner_policy_ce": 1.0, "planner_pairwise_loss": 0.7}}))
    summary = summarize_sequence_model_reports([str(report_a), str(report_b)])
    assert summary["best_by_feature_mse"]["report_path"] == str(report_b)
    assert summary["best_by_reward_mae"]["report_path"] == str(report_a)
    assert summary["best_by_value_mae"]["report_path"] == str(report_b)
    assert summary["best_by_planner_policy_ce"]["report_path"] == str(report_b)
    assert summary["best_by_planner_pairwise_loss"]["report_path"] == str(report_b)


def test_sequence_benchmark_summary(tmp_path):
    report_a = tmp_path / "a_train.json"
    report_b = tmp_path / "b_train.json"
    planner_a = tmp_path / "a_planner.json"
    planner_b = tmp_path / "b_planner.json"
    report_a.write_text(json.dumps({"val_summary": {"feature_mse": 0.2, "reward_mae": 0.3}}))
    report_b.write_text(json.dumps({"val_summary": {"feature_mse": 0.1, "reward_mae": 0.4}}))
    planner_a.write_text(json.dumps({"exact_match_rate": 0.2, "mean_rank_correlation": 0.1, "mean_teacher_gap_from_predicted_best": 0.4}))
    planner_b.write_text(json.dumps({"exact_match_rate": 0.3, "mean_rank_correlation": 0.05, "mean_teacher_gap_from_predicted_best": 0.5}))
    summary = summarize_sequence_benchmark([(str(report_a), str(planner_a)), (str(report_b), str(planner_b))])
    assert summary["best_predictive_feature_mse"]["report_path"] == str(report_b)
    assert summary["best_planner_exact_match_rate"]["planner_eval_path"] == str(planner_b)
    assert summary["best_planner_teacher_gap"]["planner_eval_path"] == str(planner_a)


def test_sequence_benchmark_summary_with_multiseed_planner_json(tmp_path):
    report_a = tmp_path / "a_train.json"
    report_b = tmp_path / "b_train.json"
    planner_a = tmp_path / "a_multiseed.json"
    planner_b = tmp_path / "b_multiseed.json"
    report_a.write_text(json.dumps({"val_summary": {"feature_mse": 0.2, "reward_mae": 0.3}}))
    report_b.write_text(json.dumps({"val_summary": {"feature_mse": 0.1, "reward_mae": 0.4}}))
    planner_a.write_text(json.dumps({"summary": {"mean_exact_match_rate": 0.35, "std_exact_match_rate": 0.08, "mean_rank_correlation": 0.2, "std_rank_correlation": 0.03, "mean_teacher_gap": 0.25, "std_teacher_gap": 0.02}}))
    planner_b.write_text(json.dumps({"summary": {"mean_exact_match_rate": 0.3, "std_exact_match_rate": 0.04, "mean_rank_correlation": 0.18, "std_rank_correlation": 0.01, "mean_teacher_gap": 0.2, "std_teacher_gap": 0.03}}))
    summary = summarize_sequence_benchmark([(str(report_a), str(planner_a)), (str(report_b), str(planner_b))])
    assert summary["best_planner_exact_match_rate"]["planner_eval_path"] == str(planner_a)
    assert summary["best_planner_rank_correlation"]["planner_eval_path"] == str(planner_a)
    assert summary["best_planner_teacher_gap"]["planner_eval_path"] == str(planner_b)
    assert summary["best_planner_rank_correlation_std"]["planner_eval_path"] == str(planner_b)
