## Purpose

Test whether replay weighting should focus only on the exact known weak confusion family instead of all current student disagreements.

The corrected replay path already supervises raw student logits directly. Broad current-disagreement weighting was negative. This probe asks whether replay should upweight only exact current `student_action -> replay_teacher_action` confusion pairs for the known `east <-> south` drift family.

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
- broad current-disagreement replay weighting probe:
  - `appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_dis2_4k_a`
- broad current-disagreement result:
  - warm-start `0.9875`
  - best learned `0.9875` at `256`
  - retained late `0.8875` at `4352`
  - final `0.8875` at `4608`

## Hypothesis

The replay weighting problem may not be “disagreement” in general. It may be that only a tiny confusion family is actually relevant.

If replay only upweights exact current confusion pairs:

- `east -> south`
- `south -> east`

then the split-base branch may retain late stability better than both plain replay and broad disagreement weighting, without using held-out states directly.

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

Added `teacher_replay_confusion_pair_boosts`.

Replay CE can now:

- parse exact `student_action->teacher_action=multiplier` specs,
- compute current student replay argmax on each replay row,
- upweight only rows that match configured confusion pairs,
- log `teacher_replay_confusion_pair_fraction`,
- and record the configured pair boosts in `improver_report.json`.

This probe used:

- `teacher_replay_confusion_pair_boosts=east->south=3.0,south->east=3.0`

with:

- no broad current-disagreement weighting,
- no teacher replay action boosts,
- unchanged rollout-time fallback behavior.

## Validation

Targeted checks:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k 'parse_teacher_confusion_pair_boosts or weight_replay_losses_by_confusion_pairs or trainer_scaffold_includes_teacher_reg_args or build_appo_config_respects_value_stability_args or trainer_scaffold_includes_trace_eval_args or improver_report_links_teacher_and_best_trace_metadata or cli_rl_train_appo_forwards_teacher_prior_controls or forward_replay_action_logits'
```

Result:

- `9 passed`

Full scaffold suite:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `88 passed`

## Exact Commands Run

Short gate:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_paires3q_4k_a \
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
path='train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_paires3q_4k_a/checkpoint_p0/checkpoint_000000017_4352.pth'
res=evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl','appo',appo_experiment='appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_paires3q_4k_a',appo_train_dir='train_dir/rl',appo_checkpoint_path=path,summary_only=True)
print(res['summary'])
PY
```

## Artifacts

- branch:
  - [appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_paires3q_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_paires3q_4k_a)
- improver report:
  - [improver_report.json](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayrawstudent_paires3q_4k_a/improver_report.json)

## Benchmark Regime

- Level 0 replay-logic change plus Level 2 short online gate
- trusted deterministic held-out trace benchmark
- same corrected replay-to-raw-student-logits branch as baseline
- only the exact replay confusion-pair weighting changed

## Primary Results

- warm-start:
  - `0.9875`
- best learned checkpoint:
  - `0.975` at `512`
- retained late checkpoint:
  - `0.95` at `4352`
- final checkpoint:
  - `0.9375` at `4608`

## Supporting Metrics

- invalid action rate:
  - `0.0` at warm-start, best learned, retained late, and final
- warm-start action counts:
  - `north=23`
  - `east=27`
  - `south=19`
  - `west=11`
- best learned action counts:
  - `north=24`
  - `east=27`
  - `south=18`
  - `west=11`
- retained late action counts:
  - `north=25`
  - `east=29`
  - `south=15`
  - `west=11`
- final action counts:
  - `north=25`
  - `east=30`
  - `south=15`
  - `west=10`
- replay config:
  - `source_mode=uniform`
  - `priority_power=1.0`
  - `current_disagreement_boost=1.0`
  - `confusion_pair_boosts=east->south=3.0,south->east=3.0`
- teacher policy config:
  - exact fallback prior `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
  - `fallback_confidence=0.55`
  - `blend_coef=0.0`
  - `disagreement_margin=0.0`

## Interpretation

This is still not a teacher-beating improver, so it fails the promotion gate.

But it is informative:

- best learned got worse than the corrected replay baseline, dropping from a teacher tie `0.9875` to `0.975`,
- retained late improved from `0.9125` to `0.95`,
- final improved from `0.9125` to `0.9375`,
- and it clearly beat the broader current-disagreement replay weighting branch late.

So exact confusion-pair replay weighting is a better stabilizer than broad disagreement weighting, but it is not enough to produce a learned checkpoint above the teacher. It shifts the branch toward stronger late preservation at the cost of weaker early teacher matching.

## What Held Up

- exact pair weighting is healthier than broad disagreement replay weighting
- retained late and final checkpoints improved relative to the corrected replay baseline
- invalid action rate stayed `0.0`
- the exact pair selector is now cleanly configurable and reported

## What Failed

- no learned checkpoint beat the trusted `0.9875` teacher
- the best learned checkpoint was worse than the corrected replay baseline
- this is still a stabilizer, not a genuine improver

## Preservation Or Improvement

This is preservation-oriented stabilization with better late retention, not real improvement.

## Recommended Next Move

- do not promote this branch to a medium run
- keep the evidence: exact failure-family replay selectors are healthier than broad disagreement replay weighting
- move the next cycle toward a true improvement mechanism on top of the corrected replay baseline, or toward richer teacher data for the same exact failure family
