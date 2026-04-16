## Purpose

Test whether the teacher-as-base line improves if rollout-time fallback and teacher supervision are decoupled.

The concrete branch keeps:

- the exact trusted `0.9875` teacher as the rollout fallback base,
- the auxiliary distilled teachers as the replay / CE supervision source,
- and asks whether that split produces either:
  - a real teacher-beating learned checkpoint, or
  - a materially stronger late-stability branch than the prior fallback probes.

## Comparable Baseline

- trusted teacher artifact:
  - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- trusted offline score:
  - `0.9875`
- comparable mask-aware online branch:
  - `appo_v4_distill_ensemble_l3pure_maskteacher_4k_a`
- mask-aware baseline:
  - warm-start `0.9875`
  - best learned `0.9875` at `256`
  - late retained `0.8875` at `4096`
  - late retained `0.8875` at `4608`
- prior exact-teacher fallback branch:
  - `appo_v4_distill_ensemble_l3pure_selfgate_f055_4k_a`
- prior exact-teacher fallback result:
  - warm-start `0.9875`
  - best learned `0.975` at `512`
  - late retained `0.9625` at `4096`
  - late retained `0.95` at `4608`

## Hypothesis

The earlier fallback probe suggested that rollout-time teacher base and supervision teacher should not be forced to use the same artifact.

If the exact trusted teacher is used only as the conservative rollout fallback base, while the auxiliary distilled teachers remain the supervision source, then:

- early trace quality should recover relative to the single-teacher fallback branch,
- late stability should remain stronger than the no-fallback mask-aware baseline,
- and the branch may become a better platform for a later residual or selective-override improver.

## Code Paths Touched

- [cli.py](/home/luc/rl-nethack/cli.py)
- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
- [rl/evaluate.py](/home/luc/rl-nethack/rl/evaluate.py)
- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## Change

Added a separate `teacher_prior_bc_path` so rollout-time fallback can use a different teacher artifact than teacher replay / CE supervision.

Behavior:

- if `teacher_prior_bc_path` is set, the actor-path teacher prior loads from that path
- if it is not set, the old behavior is preserved and the prior falls back to `teacher_bc_path`
- training, evaluation, and scaffolded launcher paths all thread the new control through consistently

## Validation

Targeted regressions:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k 'resolve_teacher_prior_bc_paths_prefers_explicit_prior or cli_rl_train_appo_forwards_teacher_prior_controls or trainer_scaffold_includes_teacher_reg_args or build_appo_config_respects_value_stability_args or teacher_policy_blend_and_fallback_helpers'
```

Result:

- `5 passed`

Full scaffold suite:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `81 passed`

## Exact Commands Run

Training run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a \
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
res = evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl', 'appo', appo_experiment='appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a', appo_train_dir='train_dir/rl', appo_checkpoint_path='train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a/checkpoint_p0/checkpoint_000000017_4352.pth', summary_only=True)
print(res['summary'])
PY
```

```bash
uv run python - <<'PY'
from rl.trace_eval import evaluate_trace_policy
res = evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl', 'appo', appo_experiment='appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a', appo_train_dir='train_dir/rl', appo_checkpoint_path='train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a/checkpoint_p0/checkpoint_000000018_4608.pth', summary_only=True)
print(res['summary'])
PY
```

## Artifacts

- split-base branch:
  - [appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a)

## Benchmark Regime

- trusted metric:
  - deterministic held-out trace match
- trace split:
  - `/tmp/x100_v4_heldout_traces.jsonl`
- observation version:
  - `v4`
- warm-start bridge:
  - BC init from `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`

## Primary Results

- step-0 warm-start:
  - `0.9875`
- best learned checkpoint:
  - `0.9875` at `512`
- retained late checkpoint:
  - `0.9875` at `4352`
- final retained checkpoint:
  - `0.975` at `4608`

## Supporting Metrics

- invalid action rate:
  - `0.0` at warm-start, best learned, and retained late checkpoints
- action-count summary:
  - warm-start and `4352` checkpoint both matched:
    - `north=23`
    - `east=27`
    - `south=19`
    - `west=11`
  - final `4608` checkpoint shifted to:
    - `north=23`
    - `east=28`
    - `south=18`
    - `west=11`
- teacher source artifact:
  - rollout fallback base:
    - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
  - replay / CE supervision teachers:
    - `/tmp/x100_v4_distill_textdistil_c020_t2_h512.pt`
    - `/tmp/x100_v4_distill_textdistil_c025_t2_h512.pt`
- teacher-student divergence metric:
  - not yet separately logged in this branch
- fallback rate:
  - not yet separately logged in this branch
- preservation or real improvement:
  - preservation / stabilization only

## Interpretation

This branch still failed the promotion gate.

It did not produce a learned checkpoint above the trusted `0.9875` teacher baseline, so it is not a teacher-beating improver.

But it did materially strengthen the teacher-as-base story:

- unlike the single-teacher exact-fallback branch, the best learned checkpoint recovered to `0.9875` instead of dropping to `0.975`
- unlike the mask-aware baseline, a retained late checkpoint at `4352` also matched `0.9875`
- unlike the auxiliary-teacher fallback branch, it did not collapse catastrophically

So the split-base design is a better stabilizer than either:

- no rollout fallback, or
- single-artifact fallback that forces the same teacher to be both the conservative base and the supervision source

## What Held Up

- decoupling fallback teacher from supervision teacher is useful and worth keeping
- the exact trusted teacher remains the right conservative rollout base
- auxiliary teachers remain usable as a separate supervision source
- the branch is the cleanest late-stability result yet in the current constrained-APPO family

## What Failed

- no learned checkpoint beat the trusted baseline
- the branch is still preservation-oriented rather than genuinely improving
- the final retained checkpoint still drifted from `0.9875` down to `0.975`
- important support metrics such as fallback rate and explicit teacher-student divergence are still not instrumented

## Recommended Next Move

Do not promote this branch to a medium or large run.

Keep the split-base teacher path as infrastructure, then test one of these on top of it:

1. residual teacher override rather than raw fallback-only behavior
2. disagreement-triggered or off-support-triggered fallback instead of confidence-only fallback
3. prioritized teacher replay or selective relabeling focused on states where the final checkpoint still diverges from the tied `4352` checkpoint
