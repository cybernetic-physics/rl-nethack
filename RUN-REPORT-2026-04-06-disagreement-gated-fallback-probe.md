## Purpose

Test whether the split-base teacher-as-base branch improves if fallback is triggered by weak student-teacher disagreement rather than by raw low-confidence states.

The concrete question is:

- keep the exact trusted `0.9875` teacher as the rollout fallback base,
- keep the auxiliary distilled teachers as replay / CE supervision,
- remove confidence fallback,
- and only snap back to the teacher when the student disagrees but does not beat the teacher-preferred action by a sufficient probability margin.

## Comparable Baseline

- trusted teacher artifact:
  - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- trusted offline score:
  - `0.9875`
- prior best split-base stabilizer:
  - `appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a`
- prior split-base result:
  - warm-start `0.9875`
  - best learned `0.9875` at `512`
  - retained late `0.9875` at `4352`
  - final retained `0.975` at `4608`
- mask-aware no-fallback baseline:
  - `appo_v4_distill_ensemble_l3pure_maskteacher_4k_a`
  - best learned `0.9875`
  - late retained `0.8875`, `0.8875`

## Hypothesis

The confidence-based split-base gate may still be too blunt.

If fallback only fires on weak disagreements, then:

- the learner should keep more room to make strong teacher-beating deviations,
- while still snapping back on off-support or weakly justified disagreements,
- and this may preserve the late-stability gain without making the branch purely preservation-oriented.

## Code Paths Touched

- [cli.py](/home/luc/rl-nethack/cli.py)
- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
- [rl/evaluate.py](/home/luc/rl-nethack/rl/evaluate.py)
- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## Change

Added a disagreement-aware fallback control:

- `teacher_policy_disagreement_margin`

Semantics:

- when the student and teacher agree on the argmax action, no disagreement gate fires
- when they disagree, the student only keeps its override if its chosen action beats the student probability assigned to the teacher-preferred action by at least the configured margin
- otherwise rollout-time control falls back to the teacher prior

This logic shares one helper between:

- rollout-time actor-path prior application
- learner-side summary metrics

So the recorded fallback/disagreement stats follow the same gate used at action-selection time.

## Validation

Targeted regressions:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k 'teacher_policy_blend_and_fallback_helpers or cli_rl_train_appo_forwards_teacher_prior_controls or trainer_scaffold_includes_teacher_reg_args or build_appo_config_respects_value_stability_args'
```

Result:

- `4 passed`

Full scaffold suite:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `81 passed`

Note:

- one existing checkpoint-monitor test flaked once during the first full-suite run and passed both in isolation and on full-suite rerun
- there was no evidence that the new disagreement gate changed checkpoint-monitor behavior

## Exact Commands Run

Short probe:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_splitgate_dm010_4k_a \
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
  --teacher-policy-disagreement-margin 0.1 \
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
for ckpt in ['checkpoint_000000016_4096.pth', 'checkpoint_000000018_4608.pth']:
    res = evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl', 'appo', appo_experiment='appo_v4_distill_ensemble_l3pure_splitgate_dm010_4k_a', appo_train_dir='train_dir/rl', appo_checkpoint_path=f'train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_dm010_4k_a/checkpoint_p0/{ckpt}', summary_only=True)
    print(ckpt, res['summary'])
PY
```

## Artifacts

- experiment:
  - [appo_v4_distill_ensemble_l3pure_splitgate_dm010_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_dm010_4k_a)

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
  - `0.9875` at `1024`
- retained late checkpoint:
  - `0.7875` at `4096`
- final retained checkpoint:
  - `0.7625` at `4608`

## Supporting Metrics

- invalid action rate:
  - `0.0` at warm-start, best learned, and retained late checkpoints
- action-count summary:
  - warm-start and best learned:
    - `north=23`
    - `east=27`
    - `south=19`
    - `west=11`
  - retained `4096` checkpoint:
    - `north=18`
    - `east=42`
    - `south=14`
    - `west=6`
  - final `4608` checkpoint:
    - `north=17`
    - `east=44`
    - `south=13`
    - `west=6`
- teacher source artifact:
  - rollout fallback base:
    - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
  - replay / CE supervision teachers:
    - `/tmp/x100_v4_distill_textdistil_c020_t2_h512.pt`
    - `/tmp/x100_v4_distill_textdistil_c025_t2_h512.pt`
- teacher-student divergence metric:
  - not yet surfaced in the improver report
- override / fallback rate:
  - not yet surfaced in the improver report
- preservation or real improvement:
  - preservation-only early tie, then strong late regression

## Interpretation

This branch is strong negative evidence against disagreement-only fallback as the next mainline improver.

It tied the teacher briefly, but:

- later than the confidence-based split-base branch
- with far worse retained late behavior
- and without any teacher-beating learned checkpoint

Compared to the prior split-base stabilizer:

- prior split-base branch:
  - best learned `0.9875` at `512`
  - retained late `0.9875` at `4352`
  - final retained `0.975`
- disagreement-gated branch:
  - best learned `0.9875` at `1024`
  - retained late `0.7875`
  - final retained `0.7625`

So removing the confidence floor and relying only on disagreement-margin gating made the constrained-improver branch much worse, not better.

## What Held Up

- the new disagreement-aware gate is implemented cleanly and remains available for later bounded experiments
- split-base teacher configuration still warm-starts exactly at `0.9875`
- the short trusted loop again caught the failure cheaply before any larger run

## What Failed

- disagreement-only gating did not beat the teacher
- disagreement-only gating destroyed the late-stability gain that the confidence-based split-base branch had recovered
- the branch drifted strongly toward `east`, which is exactly the kind of directional collapse the constrained teacher base is supposed to prevent

## Recommended Next Move

Do not promote this branch.

Keep the disagreement-gate code as reusable infrastructure, but treat this result as evidence that:

1. the confidence-based safety anchor was doing important work
2. the next improver should not be a weaker gate
3. the best next branch is a true teacher-plus-residual policy or a disagreement-triggered fallback layered on top of, not instead of, the stronger confidence-based base policy
