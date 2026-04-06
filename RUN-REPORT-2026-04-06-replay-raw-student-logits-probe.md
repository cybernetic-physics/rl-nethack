## Purpose

Test the next highest-EV replay hypothesis after the replay-prior audits:

- replay CE should not supervise the rollout-time teacher-shaped policy logits,
- it should supervise the student's raw mask-aware logits before teacher prior / fallback is applied.

This is a smaller and more principled change than a new improver family. It asks whether replay should train the student head directly while leaving teacher-as-base shaping as a rollout-time stabilizer only.

## Comparable Baseline

- trusted teacher artifact:
  - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- trusted held-out score:
  - `0.9875`
- original split-base constrained branch:
  - `appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a`
- original split-base result:
  - warm-start `0.9875`
  - best learned `0.9875` at `512`
  - retained late `0.9875` at `4352`
  - final retained `0.975` at `4608`
- replay-prior audit branch with prior cleared:
  - `appo_v4_distill_ensemble_l3pure_splitgate_f055_fixreplayraw_4k_a`
- cleared-prior result:
  - warm-start `0.9875`
  - best learned `0.9875` at `256`
  - retained late `0.8625` at `4096`
  - final `0.8375` at `4608`
- replay-prior context follow-up using replay features as prior context:
  - `appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawctx_4k_a`
- replay-feature-context result:
  - warm-start `0.9875`
  - best learned `0.9625` at `512`
  - final `0.7000` at `4608`

## Hypothesis

Replay CE should be supervising the student policy head, not the acted policy after teacher fallback/blend.

Concretely:

- on-policy acting can remain teacher-shaped,
- but replay CE should target raw student logits before rollout-time teacher prior is applied.

If that is the correct separation, then the split-base branch should recover relative to the two replay-prior audit branches.

## Code Paths Touched

- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## Change

Changed `_forward_replay_action_logits(...)` so replay CE now:

- runs replay features through `forward_head` and `forward_core`,
- bypasses patched `forward_tail`,
- reads raw student logits directly from the actor decoder plus action-parameterization layer,
- applies the replay action mask afterward,
- and leaves `_teacher_prior_raw_obs` untouched.

Regression coverage now verifies that:

- replay forward does not call the teacher-prior `forward_tail` path,
- stale prior context is left untouched,
- and both shared-weight and separate-weight actor layouts are supported.

## Validation

Targeted replay-path checks:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k "forward_replay_action_logits or scheduled_teacher_replay_coef or teacher_policy"
```

Result:

- `4 passed`

Full scaffold suite:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `85 passed`

## Exact Commands Run

Short gate with no hyperparameter changes, only the replay-target change:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_4k_a \
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
path='train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_4k_a/checkpoint_p0/checkpoint_000000016_4096.pth'
res=evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl','appo',appo_experiment='appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_4k_a',appo_train_dir='train_dir/rl',appo_checkpoint_path=path,summary_only=True)
print(res['summary'])
PY
```

## Artifacts

- branch:
  - [appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_4k_a)
- improver report:
  - [improver_report.json](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_4k_a/improver_report.json)

## Benchmark Regime

- Level 0 logic change plus Level 2 short online gate
- same trusted held-out trace split
- same observation version `v4`
- same split-base constrained setup except replay CE now targets raw student logits before teacher shaping

## Primary Results

- warm-start:
  - `0.9875`
- best learned checkpoint:
  - `0.9875` at `768`
- retained late checkpoint:
  - `0.9125` at `4096`
- final checkpoint:
  - `0.9125` at `4608`

## Supporting Metrics

- invalid action rate:
  - `0.0` at warm-start, best learned, retained late, and final
- best learned action counts:
  - `north=23`
  - `east=27`
  - `south=19`
  - `west=11`
- retained late / final action counts:
  - `north=24`
  - `east=31`
  - `south=18`
  - `west=7`
- teacher policy config:
  - exact fallback prior `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
  - `fallback_confidence=0.55`
  - `blend_coef=0.0`
  - `disagreement_margin=0.0`

## Interpretation

This is the first replay-path change after the replay-prior audits that clearly improves the corrected branch family.

Compared with the two replay-prior audit branches:

- it restores a teacher tie on the best learned checkpoint,
- and it substantially recovers late stability from `0.8375` / `0.7000` up to `0.9125`.

That is useful evidence that replay CE should indeed target raw student logits rather than the rollout-time teacher-shaped policy output.

But it is still not a teacher-beating improver:

- the best learned checkpoint only ties the teacher,
- the tie still happens early,
- and late checkpoints remain below both the trusted teacher and the original confounded split-base result.

So this is a real stabilizer correction, not a promotable win.

## What Held Up

- replay-to-raw-student-logits is healthier than replay-through-teacher-prior variants
- the warm-start bridge stayed exact at `0.9875`
- invalid action rate stayed at `0.0`
- the split-base branch recovered meaningful late stability relative to the replay-prior audit branches

## What Failed

- no learned checkpoint beat the trusted `0.9875` teacher
- the best checkpoint still looks like early preservation, not genuine learned improvement
- late checkpoints are still materially below the teacher baseline

## Preservation Or Improvement

This is preservation-oriented stabilization, not a real improvement branch.

It is better than the corrected replay-prior variants, but it does not earn promotion to a larger run.

## Recommended Next Move

- Keep this replay-target correction.
- Treat replay supervision and rollout-time teacher shaping as separate modules going forward.
- The next high-EV branch should build on this corrected replay target while adding a real improvement mechanism, most likely one of:
  - prioritized replay keyed to disagreement / weak slices once better replay data exists,
  - selective relabeling tied to trusted benchmark regressions,
  - or a true residual teacher override rather than confidence fallback alone.
