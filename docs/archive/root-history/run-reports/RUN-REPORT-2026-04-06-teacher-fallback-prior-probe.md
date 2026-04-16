## Purpose

Test a teacher-as-base online improver variant by applying a rollout-time teacher prior inside the APPO actor path, then compare whether:

- a fallback to auxiliary distillation teachers helps,
- a fallback to the exact trusted `0.9875` teacher helps,
- and whether either branch produces a real teacher-beating learned checkpoint on the trusted held-out trace benchmark.

## Comparable Baseline

- trusted teacher artifact:
  - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- trusted offline score:
  - `0.9875`
- comparable mask-aware online branch:
  - `appo_v4_distill_ensemble_l3pure_maskteacher_4k_a`
- baseline warm-start:
  - `0.9875`
- baseline best learned checkpoint:
  - `0.9875` at `256` env steps
- baseline late retained checkpoints:
  - `0.8875` at `4096`
  - `0.8875` at `4608`

## Hypothesis

The online learner is drifting because rollout actions are still chosen by a free policy, even after the teacher-loss path was fixed.

If the actor uses a conservative teacher prior at action selection time, then:

- the state distribution should stay closer to the trusted teacher manifold,
- late checkpoint collapse should weaken,
- and a teacher-fallback branch might turn the current preservation-only APPO line into a more stable improver.

## Code Paths Touched

- [cli.py](/home/luc/rl-nethack/cli.py)
- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
- [rl/evaluate.py](/home/luc/rl-nethack/rl/evaluate.py)
- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## Change

Added a rollout-time teacher prior to the Sample Factory actor path.

The implementation does three things:

1. stores raw unnormalized `v4` observations inside the actor so the action mask can be recovered at action-selection time
2. loads frozen BC teachers directly into the actor path when configured
3. applies:
   - optional probability blending with the teacher
   - optional low-confidence fallback to the teacher

For this probe, only the fallback path was used:

- `teacher_policy_blend_coef = 0.0`
- `teacher_policy_fallback_confidence = 0.55`

## Validation

Targeted regressions:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k 'cli_rl_train_appo_forwards_teacher_prior_controls or teacher_policy_blend_and_fallback_helpers or trainer_scaffold_includes_teacher_reg_args or build_appo_config_respects_value_stability_args'
```

Result:

- `4 passed`

Full scaffold suite:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `80 passed`

## Exact Commands Run

### 1. Auxiliary-teacher fallback probe

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_teachergate_f055_4k_a \
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
  --trace-eval-input /tmp/x100_v4_heldout_traces.jsonl \
  --trace-eval-interval-env-steps 128 \
  --trace-eval-top-k 5 \
  --save-every-sec 5 \
  --save-best-every-sec 5 \
  --no-rnn
```

Late retained checkpoint evals:

```bash
uv run python - <<'PY'
from rl.trace_eval import evaluate_trace_policy
res = evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl', 'appo', appo_experiment='appo_v4_distill_ensemble_l3pure_teachergate_f055_4k_a', appo_train_dir='train_dir/rl', appo_checkpoint_path='train_dir/rl/appo_v4_distill_ensemble_l3pure_teachergate_f055_4k_a/checkpoint_p0/checkpoint_000000016_4096.pth', summary_only=True)
print(res['summary'])
PY
```

```bash
uv run python - <<'PY'
from rl.trace_eval import evaluate_trace_policy
res = evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl', 'appo', appo_experiment='appo_v4_distill_ensemble_l3pure_teachergate_f055_4k_a', appo_train_dir='train_dir/rl', appo_checkpoint_path='train_dir/rl/appo_v4_distill_ensemble_l3pure_teachergate_f055_4k_a/checkpoint_p0/checkpoint_000000018_4608.pth', summary_only=True)
print(res['summary'])
PY
```

### 2. Exact-teacher fallback probe

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_selfgate_f055_4k_a \
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
  --teacher-bc-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
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
  --trace-eval-input /tmp/x100_v4_heldout_traces.jsonl \
  --trace-eval-interval-env-steps 128 \
  --trace-eval-top-k 5 \
  --save-every-sec 5 \
  --save-best-every-sec 5 \
  --no-rnn
```

Late retained checkpoint evals:

```bash
uv run python - <<'PY'
from rl.trace_eval import evaluate_trace_policy
res = evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl', 'appo', appo_experiment='appo_v4_distill_ensemble_l3pure_selfgate_f055_4k_a', appo_train_dir='train_dir/rl', appo_checkpoint_path='train_dir/rl/appo_v4_distill_ensemble_l3pure_selfgate_f055_4k_a/checkpoint_p0/checkpoint_000000016_4096.pth', summary_only=True)
print(res['summary'])
PY
```

```bash
uv run python - <<'PY'
from rl.trace_eval import evaluate_trace_policy
res = evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl', 'appo', appo_experiment='appo_v4_distill_ensemble_l3pure_selfgate_f055_4k_a', appo_train_dir='train_dir/rl', appo_checkpoint_path='train_dir/rl/appo_v4_distill_ensemble_l3pure_selfgate_f055_4k_a/checkpoint_p0/checkpoint_000000018_4608.pth', summary_only=True)
print(res['summary'])
PY
```

## Artifacts

Aux-teacher fallback branch:

- [appo_v4_distill_ensemble_l3pure_teachergate_f055_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_teachergate_f055_4k_a)

Exact-teacher fallback branch:

- [appo_v4_distill_ensemble_l3pure_selfgate_f055_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_selfgate_f055_4k_a)

## Primary Results

### 1. Auxiliary-teacher fallback was actively bad

- warm-start:
  - `0.9875`
- best learned checkpoint:
  - `0.9875` at `256`
- late retained checkpoints:
  - `0.7625` at `4096`
  - `0.7125` at `4608`

Interpretation:

- fallback to the auxiliary distillation teachers made late drift dramatically worse than the current mask-aware baseline
- this is strong negative evidence that the fallback base must match the trusted teacher artifact closely

### 2. Exact-teacher fallback improved late stability, but still failed the promotion gate

- warm-start:
  - `0.9875`
- best learned checkpoint:
  - `0.975` at `512`
- late retained checkpoints:
  - `0.9625` at `4096`
  - `0.95` at `4608`

Interpretation:

- using the exact trusted teacher as the fallback base materially reduced late collapse
- compared to the current mask-aware baseline, this is a real medium-horizon stability gain:
  - baseline late retained checkpoints: `0.8875`, `0.8875`
  - exact-teacher fallback late retained checkpoints: `0.9625`, `0.95`
- but it still did not produce a teacher-beating learned checkpoint
- and it actually weakened the early best learned checkpoint from `0.9875` to `0.975`

## Supporting Metrics

Aux-teacher fallback:

- warm-start clone equal to teacher: `yes`
- learned checkpoint above trusted teacher: `no`
- preservation-only: `yes`
- late drift reduced: `no`

Exact-teacher fallback:

- warm-start clone equal to teacher: `yes`
- learned checkpoint above trusted teacher: `no`
- preservation-only: `yes`
- late drift reduced: `yes`

## What Held Up

- the new actor-path teacher-prior plumbing works in:
  - training
  - checkpoint evaluation
  - trace-based ranking
- the exact trusted teacher is a meaningful rollout-time safety anchor
- teacher-as-base style constraints deserve to remain a real mainline family

## What Failed

- fallback alone did not create a real improver
- using the wrong fallback teacher hurt badly
- even the correct fallback teacher reduced the best learned checkpoint below the no-fallback mask-aware branch

## Recommended Next Move

Do not promote either fallback-only branch to a medium or large run.

The exact-teacher fallback is worth keeping as infrastructure because it clearly improves late stability, but the next branch should separate:

1. the teacher used as the rollout fallback base
2. the auxiliary teacher used for replay / distillation loss

The highest-value follow-up is:

- keep the exact trusted teacher as the fallback base,
- restore or improve the stronger auxiliary replay/distillation supervision separately,
- and test either:
  - a smaller residual override on top of the teacher base, or
  - a selective fallback gate triggered by disagreement / off-support signals rather than raw max-probability alone.
