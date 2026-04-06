## Purpose

Test the first true logit-space teacher residual on top of the strongest current constrained improver baseline.

The branch keeps:

- the exact trusted `0.9875` teacher as the rollout fallback base,
- the auxiliary distilled teachers as replay / CE supervision,
- the proven confidence fallback at `0.55`,
- and replaces probability blending with a logit-space residual path:
  - `teacher_logits + s * (student_logits - teacher_logits)`
  - with `s = 0.3`

This is the cleanest current approximation to "teacher remains the base policy, learner proposes a small residual delta" without rewriting the actor architecture.

## Comparable Baseline

- trusted teacher artifact:
  - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- trusted offline score:
  - `0.9875`
- strongest current constrained baseline:
  - `appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a`
- split-base confidence-fallback result:
  - warm-start `0.9875`
  - best learned `0.9875` at `512`
  - retained late `0.9875` at `4352`
  - final retained `0.975` at `4608`

## Hypothesis

Probability blending was too lossy, but that does not rule out a real teacher-base residual path.

If the actor-path prior uses teacher logits as the base and only applies a bounded fraction of the student-teacher logit delta, then:

- the teacher remains the default policy geometry,
- the student can still express small corrections,
- late drift may stay bounded better than free-policy updates,
- and the branch may retain enough teacher structure to produce a learned checkpoint above the teacher.

## Code Paths Touched

- [cli.py](/home/luc/rl-nethack/cli.py)
- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
- [rl/evaluate.py](/home/luc/rl-nethack/rl/evaluate.py)
- [rl/improver_report.py](/home/luc/rl-nethack/rl/improver_report.py)
- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## Change

Added a new teacher-policy control:

- `teacher_policy_logit_residual_scale`

Behavior:

- `1.0` keeps the raw student logits
- `0.0` snaps completely to teacher logits
- intermediate values apply a bounded teacher-base residual in logit space

This knob is now wired through config, CLI, trainer launch, evaluation, improver reports, and scaffold tests.

## Validation

Targeted validation:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k "teacher_policy_blend_and_fallback_helpers or cmd_rl_train_appo_forwards_teacher_policy_prior_args or trainer_scaffold_includes_teacher_reg_args or build_appo_config_respects_value_stability_args or improver_report_links_teacher_and_best_trace_metadata"
```

Result:

- `4 passed`

Full scaffold suite:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `81 passed`

## Exact Commands Run

Short probe:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_splitresid_r030_f055_4k_a \
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
  --teacher-policy-logit-residual-scale 0.3 \
  --teacher-policy-fallback-confidence 0.55 \
  --trace-eval-input /tmp/x100_v4_heldout_traces.jsonl \
  --trace-eval-interval-env-steps 128 \
  --trace-eval-top-k 5 \
  --save-every-sec 5 \
  --save-best-every-sec 5 \
  --no-rnn
```

Retained checkpoint evals:

```bash
uv run python - <<'PY'
from rl.trace_eval import evaluate_trace_policy
for ckpt in ['checkpoint_000000016_4096.pth', 'checkpoint_000000018_4608.pth']:
    res = evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl', 'appo', appo_experiment='appo_v4_distill_ensemble_l3pure_splitresid_r030_f055_4k_a', appo_train_dir='train_dir/rl', appo_checkpoint_path=f'train_dir/rl/appo_v4_distill_ensemble_l3pure_splitresid_r030_f055_4k_a/checkpoint_p0/{ckpt}', summary_only=True)
    print(ckpt, res['summary'])
PY
```

## Artifacts

- experiment:
  - [appo_v4_distill_ensemble_l3pure_splitresid_r030_f055_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitresid_r030_f055_4k_a)
- improver report:
  - [improver_report.json](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitresid_r030_f055_4k_a/improver_report.json)

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
  - `0.9875` at `256`
- retained late checkpoint:
  - `0.9625` at `4096`
- final retained checkpoint:
  - `0.9625` at `4608`

## Supporting Metrics

- invalid action rate:
  - `0.0` at warm-start, best learned, late, and final checkpoints
- action-count summary:
  - warm-start and best learned:
    - `north=23`
    - `east=27`
    - `south=19`
    - `west=11`
  - retained late and final:
    - `north=24`
    - `east=25`
    - `south=19`
    - `west=12`
- teacher source artifact:
  - rollout fallback base:
    - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
  - replay / CE supervision teachers:
    - `/tmp/x100_v4_distill_textdistil_c020_t2_h512.pt`
    - `/tmp/x100_v4_distill_textdistil_c025_t2_h512.pt`
- teacher policy config:
  - logit residual scale:
    - `0.3`
  - blend coef:
    - `0.0`
  - fallback confidence:
    - `0.55`
  - disagreement margin:
    - `0.0`
- preservation or real improvement:
  - preservation-only early tie, then late regression

## Interpretation

This is useful but negative evidence.

The logit-residual path is clearly better than the failed probability-blend line:

- blend branch late / final:
  - `0.8125`
  - `0.7875`
- logit-residual branch late / final:
  - `0.9625`
  - `0.9625`

But it is still worse than the plain split-base confidence-fallback stabilizer:

- split-base fallback late / final:
  - `0.9875`
  - `0.975`
- logit-residual branch late / final:
  - `0.9625`
  - `0.9625`

So the logit-residual path is a real stabilizer relative to weak blend-style mixing, but it is not the missing teacher-beating improver and it is not even the strongest stabilizer currently available.

## What Held Up

- teacher-as-base control is still the right search direction
- exact-teacher rollout anchoring still matters more than auxiliary-teacher anchoring
- the new residual knob is reusable infrastructure and is fully benchmark-compatible
- top-level improver reports now capture the teacher-policy configuration needed for constrained-actor comparisons

## What Failed

- no learned checkpoint beat the trusted `0.9875` teacher
- the best learned checkpoint only tied the teacher at the earliest retained point
- late behavior regressed below the stronger split-base fallback baseline
- global actor-path residual interpolation is not yet the right improvement mechanism

## Recommended Next Move

Do not promote this branch to a larger run.

Keep the stronger split-base confidence-fallback baseline as the main constrained line, and move the next short-gate effort to:

- prioritized teacher replay on disagreement / off-support states
- selective teacher relabeling or targeted DAgger on the drift states
- or a more surgical residual / override path that is state-gated rather than globally applied
