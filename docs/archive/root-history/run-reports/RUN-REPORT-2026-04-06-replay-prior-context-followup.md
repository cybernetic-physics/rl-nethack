## Purpose

Follow up the replay-prior audit with the next correctness hypothesis:

- replay CE should not inherit stale on-policy `_teacher_prior_raw_obs`,
- but it also should not run with teacher prior context fully disabled,
- instead it should use the replay feature batch itself as the temporary teacher-prior raw-observation context.

This isolates whether the split-base replay path wants:

- stale on-policy context,
- no prior context,
- or the replay-state context that actually corresponds to the replay batch.

## Comparable Baseline

- trusted teacher artifact:
  - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- trusted held-out score:
  - `0.9875`
- original split-base constrained baseline:
  - `appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a`
- original split-base result:
  - warm-start `0.9875`
  - best learned `0.9875` at `512`
  - retained late `0.9875` at `4352`
  - final retained `0.975` at `4608`
- cleared-prior audit branch:
  - `appo_v4_distill_ensemble_l3pure_splitgate_f055_fixreplayraw_4k_a`
- cleared-prior audit result:
  - warm-start `0.9875`
  - best learned `0.9875` at `256`
  - retained late `0.8625` at `4096`
  - final retained `0.8375` at `4608`

## Hypothesis

If replay CE is meant to supervise the current constrained policy on replay states, then the replay forward pass should see replay-state prior context, not:

- stale on-policy context,
- and not an empty prior context.

Using `replay_features` as temporary `_teacher_prior_raw_obs` should be the cleaner and more correct replay path.

## Code Paths Touched

- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## Change

Updated `_forward_replay_action_logits(...)` so replay CE now:

- saves any existing `_teacher_prior_raw_obs`,
- sets `_teacher_prior_raw_obs = replay_features` for the replay forward pass,
- restores the old value afterward, or removes the attribute if it was absent,
- keeps replay action masking unchanged.

The regression test now verifies:

- stale prior context is not reused,
- replay logits do use replay-feature context,
- and the original context is restored afterward.

## Validation

Targeted replay-prior regressions:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k "forward_replay_action_logits or scheduled_teacher_replay_coef or teacher_policy"
```

Result:

- `3 passed`

Full scaffold suite:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `84 passed`

## Exact Commands Run

Short gate with no hyperparameter changes, only the replay-context fix:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawctx_4k_a \
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

## Artifacts

- branch:
  - [appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawctx_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawctx_4k_a)
- improver report:
  - [improver_report.json](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawctx_4k_a/improver_report.json)

## Benchmark Regime

- Level 0 correctness fix plus Level 2 short online gate
- same trusted held-out trace split
- same observation version `v4`
- same split-base constrained setup except for replay prior context during replay CE

## Primary Results

- warm-start:
  - `0.9875`
- best learned checkpoint:
  - `0.9625` at `512`
- final checkpoint:
  - `0.7000` at `4608`

## Supporting Metrics

- invalid action rate:
  - `0.0` at warm-start, best learned, and final
- best learned action counts:
  - `north=24`
  - `east=25`
  - `south=19`
  - `west=12`
- final action counts:
  - `north=35`
  - `east=16`
  - `south=29`

## Interpretation

This is stronger negative evidence than the cleared-prior audit.

Replay CE is still not getting healthier just by making the teacher-prior context more semantically correct.

The branch is now:

- more correct than the stale-context version,
- more semantically aligned than the no-context version,
- but materially worse on the trusted benchmark than both.

That means the replay problem is deeper than “which raw observation tensor should teacher prior read during replay?”

The likely issue is that replay CE itself is still flowing through the rollout-time teacher-prior / fallback path, which may be the wrong target for replay supervision.

## What Held Up

- replay no longer reuses stale on-policy prior context
- the warm-start bridge stayed exact at `0.9875`
- invalid action rate stayed at `0.0`
- regression coverage and scaffold tests stayed green

## What Failed

- no learned checkpoint matched or beat the teacher
- best learned fell below the teacher immediately, to `0.9625`
- final retained behavior collapsed much further, to `0.7000`
- replay-feature prior context did not recover the old split-base stability story

## Preservation Or Improvement

This was not preservation and not improvement.

It was a correctness follow-up plus a strongly negative short-run result.

## Recommended Next Move

- Keep this correctness fix.
- Treat both stale-context replay and replay-feature-context replay as non-promotable.
- The next high-EV branch should separate replay supervision from rollout-time teacher policy shaping:
  - replay CE should likely target raw student logits before teacher prior / fallback logic,
  - while teacher-as-base stabilization remains a rollout-time policy mechanism.
