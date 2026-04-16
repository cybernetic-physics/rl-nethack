## Purpose

Test whether the strongest current split-base fallback branch can improve by using teacher replay more selectively, without changing the actor-path prior.

The branch keeps:

- the exact trusted `0.9875` teacher as the rollout fallback base,
- the auxiliary distilled teachers as replay / CE supervision,
- the proven confidence fallback at `0.55`,
- and changes only replay sampling:
  - uniform teacher replay
  - plus action-specific replay boosts `east=2.0,south=2.0`

This was chosen after auditing the current best branch’s held-out disagreement pattern.

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

## Audit That Motivated The Change

Before editing code, the replay source and the current best branch were audited.

Two important facts came out:

1. The best split-base branch only drifts on a tiny `east <-> south` confusion pair on the held-out trace set.
2. The current replay dataset `/tmp/x100_v4_train_traces.jsonl` is much weaker than the current config surface suggests:
   - `200` total rows
   - `0` rows with `behavior_action`
   - `0` disagreement rows
   - `0` loop-risk rows
   - `0` failure rows
   - so existing replay modes like `disagreement` and `mixed` mostly collapse to static teacher-action weighting, not true off-support replay

That made a small action-specific replay-weighting probe the highest-EV next step before building more complex selective relabeling.

## Hypothesis

If the current best branch only drifts on held-out `east` and `south` cases, then a replay sampler that oversamples those teacher actions may:

- preserve the strong split-base fallback behavior later into training,
- delay or reduce the observed `east <-> south` drift,
- and potentially let a learned checkpoint finally beat the teacher instead of merely tying it early.

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

Added a new replay control:

- `teacher_replay_action_boosts`

Behavior:

- accepts comma-separated `action=multiplier` values
- applies those multipliers directly to replay sampling weights
- composes with the existing `teacher_replay_source_mode` and `teacher_replay_priority_power`
- is a no-op when unset, preserving all old branches

The new knob is threaded through config, CLI, trainer launch, evaluation compatibility, improver reports, and scaffold tests.

## Validation

Targeted validation:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k "replay_priority_weights_target_requested_rows or cmd_rl_train_appo_forwards_teacher_prior_controls or trainer_scaffold_includes_teacher_reg_args or trainer_scaffold_includes_trace_eval_args or build_appo_config_respects_value_stability_args or improver_report_links_teacher_and_best_trace_metadata"
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

Replay audit:

```bash
uv run python - <<'PY'
import json
from collections import Counter
path='/tmp/x100_v4_train_traces.jsonl'
rows=0
weak=0
dis=0
loop=0
failure=0
actions=Counter()
behavior=0
with open(path) as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        row=json.loads(line)
        rows += 1
        ta = row.get('teacher_action', row.get('action'))
        ba = row.get('behavior_action')
        actions[ta] += 1
        if ba is not None:
            behavior += 1
            dis += int(ba != ta)
        weak += int(ta in {'south','west','search'})
        loop += int((row.get('repeated_state_count',0) or 0) > 0 or (row.get('repeated_action_count',0) or 0) > 0)
        failure += int(bool(row.get('done', False)) and float(row.get('reward',0.0) or 0.0) < 0.0)
print(json.dumps({
    'rows': rows,
    'behavior_action_rows': behavior,
    'disagreement_rows': dis,
    'weak_rows': weak,
    'loop_rows': loop,
    'failure_rows': failure,
    'teacher_actions': actions.most_common(),
}, indent=2))
PY
```

Baseline disagreement audit:

```bash
uv run python - <<'PY'
from rl.trace_eval import trace_disagreement_report
import json
res = trace_disagreement_report(
    '/tmp/x100_v4_heldout_traces.jsonl',
    bc_model_path='/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt',
    appo_experiment='appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a',
    appo_train_dir='train_dir/rl',
    appo_checkpoint_path='train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a/checkpoint_p0/checkpoint_000000018_4608.pth',
    top_k=20,
)
print(json.dumps(res, indent=2))
PY
```

Short probe:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_splitgate_f055_replayes2_4k_a \
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
  --teacher-replay-action-boosts east=2.0,south=2.0 \
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
for ckpt in ['checkpoint_000000017_4352.pth', 'checkpoint_000000018_4608.pth']:
    res = evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl', 'appo', appo_experiment='appo_v4_distill_ensemble_l3pure_splitgate_f055_replayes2_4k_a', appo_train_dir='train_dir/rl', appo_checkpoint_path=f'train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayes2_4k_a/checkpoint_p0/{ckpt}', summary_only=True)
    print(ckpt, res['summary'])
PY
```

## Artifacts

- experiment:
  - [appo_v4_distill_ensemble_l3pure_splitgate_f055_replayes2_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayes2_4k_a)
- improver report:
  - [improver_report.json](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_replayes2_4k_a/improver_report.json)

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
  - `0.9875` at `1280`
- retained late checkpoint:
  - `0.9625` at `4352`
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
    - `north=25`
    - `east=28`
    - `south=16`
    - `west=11`
- teacher source artifact:
  - rollout fallback base:
    - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
  - replay / CE supervision teachers:
    - `/tmp/x100_v4_distill_textdistil_c020_t2_h512.pt`
    - `/tmp/x100_v4_distill_textdistil_c025_t2_h512.pt`
- replay source:
  - trace input:
    - `/tmp/x100_v4_train_traces.jsonl`
  - source mode:
    - `uniform`
  - action boosts:
    - `east=2.0,south=2.0`
- preservation or real improvement:
  - preservation-only later tie, then late regression

## Interpretation

This is useful negative evidence.

The action-weighted replay branch did change the training trajectory:

- the best learned tie moved later:
  - baseline split-base fallback: `512`
  - replay-boost branch: `1280`

But it did not produce a teacher-beating checkpoint, and it made late retention worse:

- baseline split-base fallback late / final:
  - `0.9875`
  - `0.975`
- replay-boost branch late / final:
  - `0.9625`
  - `0.9625`

So static teacher-action replay weighting is not enough. It can move when the branch achieves its best tie, but it does not produce real improvement and does not preserve the stronger late behavior of the plain split-base fallback branch.

The most important conclusion from this cycle is broader:

- current replay prioritization is bottlenecked by data quality, not just weighting
- because the replay set currently has no `behavior_action`, loop, or failure annotations, today’s replay scheduling cannot actually target off-support student states

## What Held Up

- the split-base confidence-fallback branch remains the correct online baseline
- action-specific replay weighting is reusable infrastructure and fully benchmark-compatible
- deterministic trace evaluation again prevented a misleading promotion based on changed training dynamics

## What Failed

- no learned checkpoint beat `0.9875`
- late retention regressed relative to the baseline
- static action reweighting on the small teacher replay set did not solve the `east <-> south` drift family

## Recommended Next Move

Do not promote this branch.

The next high-EV move is no longer another replay-weight heuristic on the same static replay file. It is to improve the replay data itself or the relabeling path so online learning can actually see off-support states:

- selective student-state relabeling / targeted DAgger
- or replay traces that include real student disagreement states, loop states, or failure slices
