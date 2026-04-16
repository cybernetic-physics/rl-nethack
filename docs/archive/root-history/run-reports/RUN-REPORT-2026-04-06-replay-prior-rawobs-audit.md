## Purpose

Audit whether teacher-prior logic was leaking incorrectly into teacher replay updates.

The suspected bug was:

- on-policy forward passes store `_teacher_prior_raw_obs` on the actor-critic,
- replay updates call `forward_tail` directly on replay features,
- so replay loss could accidentally reuse stale on-policy raw observations when teacher prior or fallback is enabled.

This is a correctness question first, not a tuning question.

## Comparable Baseline

- trusted teacher artifact:
  - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- trusted held-out score:
  - `0.9875`
- strongest current constrained baseline before this audit:
  - `appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a`
- baseline result:
  - warm-start `0.9875`
  - best learned `0.9875` at `512`
  - retained late `0.9875` at `4352`
  - final retained `0.975` at `4608`

## Hypothesis

If replay loss is currently being computed through a stale teacher-prior context, then the branch is getting an invalid training signal.

Fixing that bug should:

- preserve correctness,
- reveal the true contribution of replay under the split-base fallback setup,
- and potentially change the late-stability picture substantially.

## Code Paths Touched

- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## Change

Added `_forward_replay_action_logits(...)` to compute replay logits with teacher prior disabled unless replay raw observations are explicitly available.

Behavior:

- saves the current `_teacher_prior_raw_obs`
- clears it during replay forward passes
- restores it afterwards
- preserves action-mask application on replay logits

This prevents replay CE from accidentally using stale on-policy raw observations.

## Validation

Targeted regressions:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k "forward_replay_action_logits_ignores_stale_teacher_prior_raw_obs or replay_priority_weights_target_requested_rows or scheduled_teacher_replay_coef_decays_linearly or cli_rl_train_appo_forwards_teacher_prior_controls or trainer_scaffold_includes_teacher_reg_args"
```

Result:

- `5 passed`

Full scaffold suite:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `84 passed`

## Exact Commands Run

Short gate with no hyperparameter changes, only the bug fix:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_splitgate_f055_fixreplayraw_4k_a \
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

Retained late checkpoint eval:

```bash
uv run python - <<'PY'
from rl.trace_eval import evaluate_trace_policy
path='train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_fixreplayraw_4k_a/checkpoint_p0/checkpoint_000000016_4096.pth'
res=evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl','appo',appo_experiment='appo_v4_distill_ensemble_l3pure_splitgate_f055_fixreplayraw_4k_a',appo_train_dir='train_dir/rl',appo_checkpoint_path=path,summary_only=True)
print(res['summary'])
PY
```

## Artifacts

- branch:
  - [appo_v4_distill_ensemble_l3pure_splitgate_f055_fixreplayraw_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_fixreplayraw_4k_a)
- improver report:
  - [improver_report.json](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_fixreplayraw_4k_a/improver_report.json)

## Benchmark Regime

- Level 0 correctness fix plus Level 2 short online gate
- same trusted held-out trace split
- same observation version `v4`
- same short-run split-base configuration except for the replay-prior bug fix

## Primary Results

- warm-start:
  - `0.9875`
- best learned checkpoint:
  - `0.9875` at `256`
- retained late checkpoint:
  - `0.8625` at `4096`
- final retained checkpoint:
  - `0.8375` at `4608`

## Supporting Metrics

- invalid action rate:
  - `0.0` throughout the reported checkpoints
- final action counts:
  - `north=24`
  - `east=15`
  - `south=20`
  - `west=21`
- retained `4096` action counts:
  - `north=24`
  - `east=17`
  - `south=21`
  - `west=18`

## Interpretation

This is strong negative evidence on branch behavior, but useful positive evidence on correctness.

The split-base branch became much worse once replay loss stopped inheriting stale teacher-prior raw observations.

That means the previous late-stability win was at least partly supported by an invalid coupling between:

- the on-policy teacher-prior path
- and replay supervision

So the branch is now cleaner but weaker.

## What Held Up

- the replay-prior raw-observation leak was real and is now fixed
- the warm-start bridge remained exact at `0.9875`
- the bug fix did not break evaluation, action masking, or training infrastructure

## What Failed

- the corrected branch lost the prior late-stability behavior
- no learned checkpoint beat the baseline
- late checkpoints regressed far below the trusted constrained baseline

## Preservation Or Improvement

This was not improvement. It was a correctness fix plus a negative short-run result.

## Recommended Next Move

- Keep the correctness fix.
- Treat the old split-base late-stability result as partially confounded by replay-prior leakage.
- Do not promote the corrected branch as the new constrained baseline.
- The next improver step should not assume that current replay CE is strong enough on its own once the invalid coupling is removed.
