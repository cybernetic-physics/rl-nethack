## Purpose

Audit and fix the teacher-regularized APPO loss so it respects the environment action mask, then measure whether that changes the trusted short-run behavior on the current `0.9875` teacher line.

Comparable baseline before this run:

- teacher artifact:
  - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- trusted teacher score:
  - `0.9875`
- comparable short online branch:
  - `appo_v4_distill_ensemble_l3pure_probe_warmstart_b`
- baseline warm-start:
  - `0.9875`
- baseline best learned checkpoint:
  - `0.975` at `256` env steps
- baseline later checkpoints:
  - `0.925` at `1280`
  - `0.9125` at `1536`

## Hypothesis

The current teacher-regularized APPO branch is leaking invalid action preferences into the constraint.

More concretely:

- teacher CE/KL currently uses raw BC teacher logits
- student teacher-loss logits are also unmasked
- but this repo’s low-level control is mask-sensitive
- so the online constraint can be misaligned with the real allowed-action policy surface

If we make teacher CE/KL mask-aware using the allowed-action mask already embedded in the observation features, the short branch should preserve the teacher better.

## Code Paths Touched

- [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)
- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## Code Change

Added an explicit action-mask slice helper to the feature encoder and used it inside teacher regularization.

The key fix is:

- extract the allowed-action mask from `raw_obs`
- mask both teacher logits and student logits before:
  - teacher CE / KL
  - teacher argmax selection
  - teacher/student agreement

Also added diagnostics for:

- `teacher_invalid_preference_fraction`
- `student_invalid_preference_fraction`

## Validation

Targeted regression:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k 'action_mask_slice_extracts_allowed_actions_prefix or teacher_reg_masks_invalid_teacher_and_student_preferences or scheduled_teacher_replay_coef_decays_linearly'
```

Result:

- `3 passed`

Full scaffold suite:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `78 passed`

## Exact Commands Run

### 1. Comparable short probe

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_probe_maskteacher_a \
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
  --train-for-env-steps 1024 \
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
  --trace-eval-input /tmp/x100_v4_heldout_traces.jsonl \
  --trace-eval-interval-env-steps 128 \
  --trace-eval-top-k 5 \
  --save-every-sec 5 \
  --save-best-every-sec 5 \
  --no-rnn
```

### 2. Longer cheap gate

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_maskteacher_4k_a \
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
  --trace-eval-input /tmp/x100_v4_heldout_traces.jsonl \
  --trace-eval-interval-env-steps 128 \
  --trace-eval-top-k 5 \
  --save-every-sec 5 \
  --save-best-every-sec 5 \
  --no-rnn
```

### 3. Late checkpoint evaluation for the 4k branch

```bash
uv run python - <<'PY'
from rl.trace_eval import evaluate_trace_policy
res = evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl', 'appo', appo_experiment='appo_v4_distill_ensemble_l3pure_maskteacher_4k_a', appo_train_dir='train_dir/rl', appo_checkpoint_path='train_dir/rl/appo_v4_distill_ensemble_l3pure_maskteacher_4k_a/checkpoint_p0/checkpoint_000000017_4352.pth', summary_only=True)
print(res['summary'])
PY
```

```bash
uv run python - <<'PY'
from rl.trace_eval import evaluate_trace_policy
res = evaluate_trace_policy('/tmp/x100_v4_heldout_traces.jsonl', 'appo', appo_experiment='appo_v4_distill_ensemble_l3pure_maskteacher_4k_a', appo_train_dir='train_dir/rl', appo_checkpoint_path='train_dir/rl/appo_v4_distill_ensemble_l3pure_maskteacher_4k_a/checkpoint_p0/checkpoint_000000018_4608.pth', summary_only=True)
print(res['summary'])
PY
```

## Artifacts

Short probe:

- [appo_v4_distill_ensemble_l3pure_probe_maskteacher_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_probe_maskteacher_a)

Longer cheap gate:

- [appo_v4_distill_ensemble_l3pure_maskteacher_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_maskteacher_4k_a)

## Primary Results

### 1. The mask-aware fix materially improved the short learned checkpoint

For [appo_v4_distill_ensemble_l3pure_probe_maskteacher_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_probe_maskteacher_a):

- warm-start:
  - `0.9875`
- best learned checkpoint:
  - `0.9875`
  - at `1536` env steps

This is a real improvement over the comparable pre-fix short branch:

- old best learned checkpoint:
  - `0.975` at `256` env steps

So the bug fix closed the short learned gap from `0.975` back up to the full teacher level.

### 2. The branch is still preservation, not improvement

For [appo_v4_distill_ensemble_l3pure_maskteacher_4k_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_maskteacher_4k_a):

- warm-start:
  - `0.9875`
- best learned checkpoint:
  - `0.9875`
  - at `256` env steps
- late retained checkpoints:
  - `4352` env steps -> `0.8875`
  - `4608` env steps -> `0.8875`

So the fix delayed and reduced drift, but it did not produce a learned checkpoint above the teacher.

## Interpretation

What held up:

- mask details really do matter in this repo
- the previous teacher-reg path was not as trustworthy as it looked
- making teacher CE/KL mask-aware materially improved short-run preservation

What did not hold up:

- the fix did not produce a teacher-beating improver
- the branch still drifts later in training
- this is not yet a valid medium-promotion win under the repo’s rules

The most accurate conclusion is:

- this is a **stabilizer fix**
- not yet an **improver breakthrough**

That still matters, because it changes the search landscape:

- before the fix, the branch degraded almost immediately
- after the fix, the same branch can hold the teacher exactly through the short gate

## Recommended Next Move

Do not promote this branch to a medium or large run yet.

The next highest-EV move is:

1. keep the mask-aware teacher loss as the new default
2. build on top of it with a more explicitly constrained improver
3. prioritize:
   - teacher-as-base residual or gated override
   - stronger replay prioritization on disagreement / weak slices
   - selective relabeling on off-support states

This fix reduced one real source of drift. It did not remove the main bottleneck.
