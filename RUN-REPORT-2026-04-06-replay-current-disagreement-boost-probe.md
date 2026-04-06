## Purpose

Test whether scheduled replay should focus more aggressively on the rows the current student still gets wrong.

The corrected replay path now supervises raw student logits directly. This probe asks whether replay CE should further upweight rows where the current student argmax disagrees with the replay teacher action, while keeping rollout-time teacher fallback unchanged.

## Comparable Baseline

- trusted teacher artifact:
  - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- trusted held-out split:
  - `/tmp/x100_v4_heldout_traces.jsonl`
- observation version:
  - `v4`
- corrected split-base replay baseline:
  - `appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_4k_a`
- corrected replay baseline result:
  - warm-start `0.9875`
  - best learned `0.9875` at `768`
  - retained late `0.9125` at `4352`
  - final `0.9125` at `4608`

## Hypothesis

Replay CE may still be too diffuse even after targeting raw student logits.

If replay rows where the current student still disagrees with the replay teacher are upweighted, then the branch may:

- preserve the exact teacher tie longer,
- resist the small `east <-> south` held-out drift more effectively,
- and improve retained late checkpoints without changing rollout-time teacher fallback.

## Code Paths Touched

- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/improver_report.py](/home/luc/rl-nethack/rl/improver_report.py)
- [rl/evaluate.py](/home/luc/rl-nethack/rl/evaluate.py)
- [cli.py](/home/luc/rl-nethack/cli.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## Change

Added `teacher_replay_current_disagreement_boost`.

Replay CE now:

- computes per-row CE loss with `reduction="none"`,
- compares the current student replay argmax against the replay teacher action,
- upweights disagreement rows by a configurable multiplicative factor,
- reports `teacher_replay_current_disagreement_fraction`,
- and records the new replay-source setting in `improver_report.json`.

The probe used:

- `teacher_replay_current_disagreement_boost=2.0`

Regression coverage verifies:

- CLI / config / trainer plumbing for the new setting,
- report serialization,
- and per-row disagreement weighting behavior.

## Validation

Targeted checks:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k 'weight_replay_losses_by_current_disagreement or trainer_scaffold_includes_teacher_reg_args or build_appo_config_preserves_teacher_reg_settings or trainer_scaffold_includes_trace_eval_args or improver_report_links_teacher_and_best_trace_metadata or cli_rl_train_appo_forwards_teacher_prior_controls or forward_replay_action_logits'
```

Result:

- `7 passed`

Full scaffold suite:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `86 passed`

## Exact Commands Run

Short gate:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_dis2_4k_a \
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
  --teacher-replay-current-disagreement-boost 2.0 \
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
path='train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_dis2_4k_a/checkpoint_p0/checkpoint_000000017_4352.pth'
res=evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl','appo',appo_experiment='appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_dis2_4k_a',appo_train_dir='train_dir/rl',appo_checkpoint_path=path,summary_only=True)
print(res['summary'])
PY
```

## Artifacts

- branch:
  - [appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_dis2_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_dis2_4k_a)
- improver report:
  - [improver_report.json](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_dis2_4k_a/improver_report.json)

## Benchmark Regime

- Level 0 replay-logic change plus Level 2 short online gate
- trusted deterministic held-out trace benchmark
- same split-base teacher-fallback setup as the corrected replay baseline
- only replay weighting changed

## Primary Results

- warm-start:
  - `0.9875`
- best learned checkpoint:
  - `0.9875` at `256`
- retained late checkpoint:
  - `0.8875` at `4352`
- final checkpoint:
  - `0.8875` at `4608`

## Supporting Metrics

- invalid action rate:
  - `0.0` at warm-start, best learned, retained late, and final
- warm-start / best learned action counts:
  - `north=23`
  - `east=27`
  - `south=19`
  - `west=11`
- retained late / final action counts:
  - `north=31`
  - `east=27`
  - `south=11`
  - `west=11`
- teacher policy config:
  - exact fallback prior `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
  - `fallback_confidence=0.55`
  - `blend_coef=0.0`
  - `disagreement_margin=0.0`
- replay config:
  - replay trace input `/tmp/x100_v4_train_traces.jsonl`
  - `source_mode=uniform`
  - `priority_power=1.0`
  - `current_disagreement_boost=2.0`

## Interpretation

This result is negative.

Compared with the corrected replay baseline:

- the best learned teacher tie moved earlier, from `768` to `256`,
- retained late performance regressed from `0.9125` to `0.8875`,
- final performance regressed from `0.9125` to `0.8875`,
- and no learned checkpoint beat the trusted `0.9875` teacher.

So current-student disagreement weighting is not the right next replay lever on this tiny replay source. It appears to overconcentrate replay pressure on whatever the current student already disagrees with, without translating that emphasis into better held-out retention.

## What Held Up

- raw-student replay supervision remained stable enough to preserve a teacher tie early
- invalid action rate stayed at `0.0`
- the new weighting path, metrics, and report plumbing validated cleanly

## What Failed

- no learned checkpoint beat the trusted teacher
- late checkpoints were worse than the unweighted corrected replay baseline
- the branch still looks like early preservation, not genuine improvement

## Preservation Or Improvement

This is preservation-only and weaker than the corrected replay baseline. It is not a promotable improver branch.

## Recommended Next Move

- keep `d1213c2` style replay-to-raw-student-logits as the healthier replay baseline
- do not promote current-disagreement replay weighting
- shift the next cycle toward richer targeted teacher data or a real teacher-as-base residual / override mechanism rather than more weighting on the tiny teacher-only replay set
