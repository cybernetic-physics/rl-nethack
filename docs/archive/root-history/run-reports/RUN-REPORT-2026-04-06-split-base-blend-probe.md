## Purpose

Test the first explicit teacher-base residual variant on top of the strongest current stabilizer.

The branch keeps:

- the exact trusted `0.9875` teacher as rollout fallback base,
- the auxiliary distilled teachers as replay / CE supervision,
- the proven confidence fallback at `0.55`,
- and adds a small teacher-probability blend (`0.15`) so the online actor behaves like a residual around the teacher rather than only a free policy with fallback.

This is the cleanest current approximation to "teacher remains the base policy, learner proposes a small delta" without introducing a larger model rewrite.

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

Confidence fallback alone is a stabilizer but not yet an improver.

If a small teacher-probability blend is added on top of that stabilizer, then:

- the teacher remains the effective base policy,
- the actor only proposes a small residual shift around it,
- late drift may weaken further,
- and the branch may preserve enough teacher structure to let a learned checkpoint finally exceed the teacher.

## Code Paths Touched

- [rl/improver_report.py](/home/luc/rl-nethack/rl/improver_report.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

The online policy logic for blend itself already existed in [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py); this cycle used it as-is.

## Change

Improved improver-report generation so constrained-improver branches now record:

- top-level `best_trace_metadata`
- top-level `final_trace_metadata`
- direct `teacher_policy` config summary
- nested final-checkpoint trace metadata inside `trace_gate`

This makes future short-run reports less dependent on ad hoc postprocessing.

## Validation

Targeted report-path validation:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k 'improver_report_links_teacher_and_best_trace_metadata or build_appo_config_respects_value_stability_args or trainer_scaffold_includes_teacher_reg_args'
```

Result:

- `3 passed`

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
  --experiment appo_v4_distill_ensemble_l3pure_splitblend_b015_f055_4k_a \
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
  --teacher-policy-blend-coef 0.15 \
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
for ckpt in ['checkpoint_000000017_4352.pth', 'checkpoint_000000018_4608.pth']:
    res = evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl', 'appo', appo_experiment='appo_v4_distill_ensemble_l3pure_splitblend_b015_f055_4k_a', appo_train_dir='train_dir/rl', appo_checkpoint_path=f'train_dir/rl/appo_v4_distill_ensemble_l3pure_splitblend_b015_f055_4k_a/checkpoint_p0/{ckpt}', summary_only=True)
    print(ckpt, res['summary'])
PY
```

## Artifacts

- experiment:
  - [appo_v4_distill_ensemble_l3pure_splitblend_b015_f055_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitblend_b015_f055_4k_a)
- improved report:
  - [improver_report.json](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitblend_b015_f055_4k_a/improver_report.json)

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
  - `0.8125` at `4352`
- final retained checkpoint:
  - `0.7875` at `4608`

## Supporting Metrics

- invalid action rate:
  - `0.0` at warm-start, best learned, and retained late checkpoints
- action-count summary:
  - warm-start and best learned:
    - `north=23`
    - `east=27`
    - `south=19`
    - `west=11`
  - retained `4352` checkpoint:
    - `north=26`
    - `east=39`
    - `south=9`
    - `west=6`
  - final `4608` checkpoint:
    - `north=26`
    - `east=41`
    - `south=7`
    - `west=6`
- teacher source artifact:
  - rollout fallback base:
    - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
  - replay / CE supervision teachers:
    - `/tmp/x100_v4_distill_textdistil_c020_t2_h512.pt`
    - `/tmp/x100_v4_distill_textdistil_c025_t2_h512.pt`
- teacher policy config:
  - blend coef:
    - `0.15`
  - fallback confidence:
    - `0.55`
  - disagreement margin:
    - `0.0`
- preservation or real improvement:
  - preservation-only early tie, then strong late regression

## Interpretation

This branch is strong negative evidence against simple probability blending as the missing teacher-base improver.

Compared to the stronger split-base confidence-fallback baseline:

- baseline split-base branch:
  - best learned `0.9875` at `512`
  - retained late `0.9875`
  - final retained `0.975`
- blend branch:
  - best learned `0.9875` at `512`
  - retained late `0.8125`
  - final retained `0.7875`

So the probability-blend version of a teacher-base residual is materially worse than the simpler confidence-fallback stabilizer. It does not improve early, and it destroys the late-stability gain that made the split-base branch worth keeping.

## What Held Up

- the improved improver-report path now records warm-start, best, and final trace metadata directly
- split-base warm-start remains exact at `0.9875`
- the short trusted loop again rejected a plausible branch cheaply

## What Failed

- no teacher-beating learned checkpoint
- no early gain over the current split-base stabilizer
- severe late drift despite both teacher blend and confidence fallback

## Recommended Next Move

Do not promote this branch.

Treat this as evidence that:

1. simple probability blending is not the right residual parameterization
2. the current best constrained branch is still the plain split-base confidence-fallback stabilizer
3. the next mainline branch should either:
   - implement a true additive teacher-plus-residual policy, or
   - leave the actor path alone and improve teacher-data use via prioritized replay / selective relabeling
