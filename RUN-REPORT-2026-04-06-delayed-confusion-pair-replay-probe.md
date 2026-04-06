## Purpose

Test whether exact confusion-pair replay weighting should be delayed until after the early teacher-preservation phase instead of being active from the start.

The previous exact confusion-pair replay probe improved late retention but lost the exact teacher tie early. This follow-up asks whether the same exact `east <-> south` replay weighting becomes healthier if it stays off until the branch reaches the prior best-learned point.

## Comparable Baseline

- trusted teacher artifact:
  - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- trusted held-out split:
  - `/tmp/x100_v4_heldout_traces.jsonl`
- observation version:
  - `v4`
- corrected replay baseline:
  - `appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_4k_a`
- corrected replay baseline result:
  - warm-start `0.9875`
  - best learned `0.9875` at `768`
  - retained late `0.9125` at `4352`
  - final `0.9125` at `4608`
- exact confusion-pair replay-weighting probe:
  - `appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_paires3q_4k_a`
- exact confusion-pair replay-weighting result:
  - warm-start `0.9875`
  - best learned `0.975` at `512`
  - retained late `0.95` at `4352`
  - final `0.9375` at `4608`

## Hypothesis

Exact replay weighting may be solving a real late-stability problem, but it may be turning on too early and perturbing the warm-start bridge.

If the exact confusion-pair boosts stay off until `768` env steps, then the branch may:

- preserve the exact teacher tie early,
- keep the late-stability gain from exact failure-family weighting,
- and convert the prior stabilizer-only result into a healthier comparable short gate.

## Code Paths Touched

- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/evaluate.py](/home/luc/rl-nethack/rl/evaluate.py)
- [rl/improver_report.py](/home/luc/rl-nethack/rl/improver_report.py)
- [cli.py](/home/luc/rl-nethack/cli.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## Change

Added `teacher_replay_confusion_pair_start_env_steps`.

Exact replay confusion-pair boosts can now stay disabled until a configured env-step threshold, while leaving:

- replay-to-raw-student-logits supervision unchanged,
- rollout-time teacher fallback unchanged,
- and the exact pair selector unchanged.

This probe used:

- `teacher_replay_confusion_pair_boosts=east->south=3.0,south->east=3.0`
- `teacher_replay_confusion_pair_start_env_steps=768`

## Validation

Targeted checks:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k 'active_replay_confusion_pair_boosts or parse_teacher_confusion_pair_boosts or weight_replay_losses_by_confusion_pairs or trainer_scaffold_includes_teacher_reg_args or build_appo_config_respects_value_stability_args or trainer_scaffold_includes_trace_eval_args or improver_report_links_teacher_and_best_trace_metadata or cli_rl_train_appo_forwards_teacher_prior_controls or forward_replay_action_logits'
```

Result:

- `10 passed`

Full scaffold suite:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `89 passed`

## Exact Commands Run

Short gate:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_paires3s768_4k_a \
  --train-dir train_dir/rl \
  --num-workers 2 \
  --num-envs-per-worker 4 \
  --rollout-length 16 \
  --recurrence 16 \
  --batch-size 256 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --learning-rate 0.0001 \
  --gamma 0.99 \
  --gae-lambda 0.9 \
  --value-loss-coeff 0.1 \
  --reward-scale 0.005 \
  --entropy-coeff 0.01 \
  --ppo-clip-ratio 0.1 \
  --train-for-env-steps 4096 \
  --scheduler rule_based \
  --reward-source hand_shaped \
  --episodic-explore-bonus-scale 0.0 \
  --episodic-explore-bonus-mode state_hash \
  --enabled-skills explore \
  --observation-version v4 \
  --env-max-episode-steps 200 \
  --model-hidden-size 1024 \
  --model-num-layers 3 \
  --bc-init-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --teacher-bc-path /tmp/x100_v4_distill_textdistil_c020_t2_h512.pt,/tmp/x100_v4_distill_textdistil_c025_t2_h512.pt \
  --teacher-prior-bc-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --teacher-report-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt.teacher_report.json \
  --teacher-loss-coef 0.01 \
  --teacher-loss-type ce \
  --teacher-loss-final-coef 0.005 \
  --teacher-loss-warmup-env-steps 128 \
  --teacher-loss-decay-env-steps 768 \
  --teacher-replay-trace-input /tmp/x100_v4_train_traces.jsonl \
  --teacher-replay-coef 0.02 \
  --teacher-replay-final-coef 0.005 \
  --teacher-replay-warmup-env-steps 128 \
  --teacher-replay-decay-env-steps 768 \
  --teacher-replay-batch-size 128 \
  --teacher-replay-priority-power 1.0 \
  --teacher-replay-source-mode uniform \
  --teacher-replay-confusion-pair-boosts 'east->south=3.0,south->east=3.0' \
  --teacher-replay-confusion-pair-start-env-steps 768 \
  --teacher-policy-blend-coef 0.0 \
  --teacher-policy-fallback-confidence 0.55 \
  --trace-eval-input /tmp/x100_v4_heldout_traces.jsonl \
  --trace-eval-interval-env-steps 128 \
  --trace-eval-top-k 5 \
  --save-every-sec 5 \
  --save-best-every-sec 5 \
  --no-rnn
```

Retained late checkpoint eval:

```bash
uv run python - <<'PY'
from rl.trace_eval import evaluate_trace_policy
path='train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_paires3s768_4k_a/checkpoint_p0/checkpoint_000000017_4352.pth'
res=evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl','appo',appo_experiment='appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_paires3s768_4k_a',appo_train_dir='train_dir/rl',appo_checkpoint_path=path,summary_only=True)
print(res['summary'])
PY
```

## Artifacts

- branch:
  - [appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_paires3s768_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_paires3s768_4k_a)
- improver report:
  - [improver_report.json](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_paires3s768_4k_a/improver_report.json)

## Benchmark Regime

- Level 0 replay-logic change plus Level 2 short online gate
- trusted deterministic held-out trace benchmark
- same corrected replay-to-raw-student-logits branch as baseline
- only the activation schedule for exact replay confusion-pair weighting changed

## Primary Results

- warm-start:
  - `0.9875`
- best learned checkpoint:
  - `0.9875` at `768`
- retained late checkpoint:
  - `0.725` at `4352`
- final checkpoint:
  - `0.7125` at `4608`

## Supporting Metrics

- invalid action rate:
  - `0.0` at warm-start, best learned, retained late, and final
- warm-start action counts:
  - `north=23`
  - `east=27`
  - `south=19`
  - `west=11`
- best learned action counts:
  - `north=23`
  - `east=27`
  - `south=19`
  - `west=11`
- retained late action counts:
  - `north=31`
  - `east=17`
  - `south=32`
- final action counts:
  - `north=33`
  - `east=15`
  - `south=31`
  - `west=1`
- teacher policy:
  - `blend_coef=0.0`
  - `fallback_confidence=0.55`
  - `logit_residual_scale=1.0`
  - `disagreement_margin=0.0`
- replay config:
  - `source_mode=uniform`
  - `priority_power=1.0`
  - `current_disagreement_boost=1.0`
  - `confusion_pair_boosts=east->south=3.0,south->east=3.0`
  - `confusion_pair_start_env_steps=768`
- teacher/student divergence metric:
  - not separately surfaced in `improver_report.json` for this branch
- fallback / override rate:
  - not separately surfaced in `improver_report.json` for this branch

## Preservation Or Improvement

This is preservation-only at best, and not a successful stabilizer.

The branch restored the exact early teacher tie, but only by postponing the weighted replay signal until the point where the earlier corrected baseline had already reached its best checkpoint. After that point, the run collapsed well below every relevant corrected replay comparator.

## Interpretation

The exact confusion-pair replay weighting family now shows a clear internal tradeoff:

- active early:
  - better late retention, weaker early teacher matching
- delayed to preserve early teacher matching:
  - restored early tie, severe late collapse

That means the benefit from exact confusion-pair weighting is not simply “too much too early.” Its late-stability effect appears to depend on shaping the branch from the start. Delaying it converts the branch into ordinary early preservation followed by drift.

## What Held Up

- replay-to-raw-student-logits remains the correct replay supervision target
- exact failure-family replay weighting is a real selector, not just a broad disagreement proxy
- the new schedule control behaved as intended and cleanly reproduced the early teacher tie

## What Failed

- delayed activation destroyed the late-stability benefit of exact confusion-pair weighting
- the branch never produced a learned checkpoint above the trusted teacher
- the final policy drifted hard toward `north` / `south` behavior despite zero invalid actions

## Recommended Next Move

Stop tuning simple activation schedules inside this replay-weighting family.

The corrected replay path is now clean enough to support stronger next branches, but this result says the remaining bottleneck is not “when to turn on the same small replay weighting trick.” The next high-EV move is a different improvement mechanism on top of the corrected replay base, such as:

- a true teacher-as-base residual policy,
- richer teacher/student replay data than the tiny teacher-only replay file,
- or a different conservative online improver that can use exact failure-family signals without relying on raw weighting alone.
