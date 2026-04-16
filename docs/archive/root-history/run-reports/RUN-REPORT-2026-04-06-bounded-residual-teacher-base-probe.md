## Purpose

Test whether rollout-time teacher shaping should use a bounded residual around the teacher logits instead of the earlier global blend / scale variants.

The corrected replay baseline already established that replay CE should target raw student logits, not the teacher-shaped acted policy. The next high-EV question is how the acted policy itself should stay teacher-based online. This probe asks whether the right bias is:

- teacher logits as the base policy,
- student logits as a bounded delta,
- plus the existing conservative confidence fallback.

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
  - retained late `0.9125` at `4096`
  - final `0.9125` at `4608`
- earlier global logit-residual probe:
  - `appo_v4_distill_ensemble_l3pure_splitresid_r030_f055_4k_a`
- earlier global logit-residual result:
  - warm-start `0.9875`
  - best learned `0.9875` at `256`
  - retained late `0.9625` at `4096`
  - final `0.9625` at `4608`
- historically stronger but confounded split-base branch before replay audit:
  - `appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a`
- historical pre-audit result:
  - warm-start `0.9875`
  - best learned `0.9875` at `512`
  - retained late `0.9875` at `4352`
  - final `0.975` at `4608`

## Hypothesis

The earlier “residual” probe still used an unbounded global interpolation toward student logits. That is too free.

If rollout-time teacher shaping instead uses:

- `teacher_logits + clipped(student_logits - teacher_logits)`

with a small per-action cap, then the branch may:

- preserve exact teacher-aligned behavior better than global blend / scale,
- keep the corrected replay path clean,
- and recover the late stability of the old split-base line without replay-prior confounds.

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

Added `teacher_policy_residual_logit_cap`.

Rollout-time teacher shaping can now clip each action logit residual around the teacher prior before action selection. This is a true bounded teacher-as-base policy, not a probability blend and not an unbounded student interpolation.

This probe used:

- `teacher_policy_logit_residual_scale=1.0`
- `teacher_policy_residual_logit_cap=0.5`
- `teacher_policy_fallback_confidence=0.55`
- corrected replay-to-raw-student-logits supervision unchanged

## Validation

Targeted checks:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k 'teacher_policy_blend_and_fallback_helpers or cli_rl_train_appo_forwards_teacher_prior_controls or trainer_scaffold_includes_teacher_reg_args or build_appo_config_respects_value_stability_args or improver_report_links_teacher_and_best_trace_metadata'
```

Result:

- `5 passed`

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
  --experiment appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_rescap05_4k_a \
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
  --teacher-policy-blend-coef 0.0 \
  --teacher-policy-fallback-confidence 0.55 \
  --teacher-policy-residual-logit-cap 0.5 \
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
path='train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_rescap05_4k_a/checkpoint_p0/checkpoint_000000017_4352.pth'
res=evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl','appo',appo_experiment='appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_rescap05_4k_a',appo_train_dir='train_dir/rl',appo_checkpoint_path=path,summary_only=True)
print(res['summary'])
PY
```

## Artifacts

- branch:
  - [appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_rescap05_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_rescap05_4k_a)
- improver report:
  - [improver_report.json](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_rescap05_4k_a/improver_report.json)

## Benchmark Regime

- Level 0 teacher-policy logic change plus Level 2 short online gate
- trusted deterministic held-out trace benchmark
- same corrected replay-to-raw-student-logits branch as baseline
- only the rollout-time teacher-as-base policy form changed

## Primary Results

- warm-start:
  - `0.9875`
- best learned checkpoint:
  - `0.9875` at `512`
- retained late checkpoint:
  - `0.9875` at `4352`
- final checkpoint:
  - `0.9875` at `4608`

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
  - `north=23`
  - `east=27`
  - `south=19`
  - `west=11`
- final action counts:
  - `north=23`
  - `east=27`
  - `south=19`
  - `west=11`
- teacher policy:
  - `prior_checkpoint_path=/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
  - `logit_residual_scale=1.0`
  - `residual_logit_cap=0.5`
  - `blend_coef=0.0`
  - `fallback_confidence=0.55`
  - `disagreement_margin=0.0`
- replay config:
  - `source_mode=uniform`
  - `priority_power=1.0`
  - `current_disagreement_boost=1.0`
  - `confusion_pair_boosts=''`
- teacher/student divergence metric:
  - not separately surfaced in `improver_report.json` for this branch
- fallback / override rate:
  - not separately surfaced in `improver_report.json` for this branch

## Preservation Or Improvement

This is a preservation branch, not yet a teacher-beating improver.

It does not beat the trusted teacher. But it is a real branch improvement over the corrected replay baseline, because a learned checkpoint at `512` and both late checkpoints all recover from `0.9125` back to the full trusted `0.9875` teacher match.

## Interpretation

This is the strongest clean post-audit result so far for the constrained improver line.

The important distinction from earlier negative branches is architectural:

- probability blending was too weak and unstable,
- global logit scaling was healthier but still leaked too much student drift,
- replay weighting alone only traded early preservation against late stability,
- bounded residual teacher-as-base control restores exact late teacher preservation without replay-prior confounds.

So the teacher should remain the acted base policy, but the student should be allowed only a small bounded logit delta around it.

## What Held Up

- corrected replay-to-raw-student-logits supervision remained healthy
- the bounded residual cap cleanly integrated with the existing teacher fallback path
- the branch preserved exact trusted teacher behavior through the end of the short run
- this cleanly recovers the spirit of the old split-base line without relying on the replay-prior bug

## What Failed

- the branch still did not produce a learned checkpoint above the trusted teacher
- the best checkpoint is still an early-preservation tie, not a genuine learned improvement

## Recommended Next Move

Treat this as the new corrected stabilization baseline for teacher-as-base APPO, not as the final answer.

The next high-EV move is no longer another replay-weight tweak. It is to build genuine improvement mechanisms on top of this bounded residual teacher base, for example:

- richer teacher/student replay data,
- selective relabeling keyed to real regression states,
- or a medium-run test only if the goal is to validate this as the default clean preservation branch rather than to claim teacher-beating improvement.
