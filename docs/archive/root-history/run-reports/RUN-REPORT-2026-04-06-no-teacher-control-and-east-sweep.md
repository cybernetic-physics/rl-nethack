# Purpose

Validate whether teacher regularization is actually helping the current `0.9875` teacher bridge, then test small offline teacher tweaks aimed at the remaining east-action errors.

# Code Path Touched

- [cli.py](/home/luc/rl-nethack/cli.py)
- [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

# Benchmark Regime

- Trusted metric: deterministic held-out trace match on `/tmp/x100_v4_heldout_traces.jsonl`
- Active strong teacher baseline:
  - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
  - `match_rate = 0.9875`

# Hypotheses

1. The previous no-teacher APPO control was invalid because `cli.py` dropped explicit zero-valued teacher coefficients.
2. If teacher CE/replay are helping, a valid no-teacher control should underperform the earlier `0.975` short learned checkpoints.
3. If the remaining misses are concentrated on `east` decisions, small offline teacher tweaks around that class may improve the teacher before more online RL.

# What Changed

## CLI fix

- `cmd_rl_train_appo` now forwards zero-valued RL control knobs instead of silently dropping them.
- This specifically fixes cases like:
  - `--teacher-loss-coef 0.0`
  - `--teacher-replay-coef 0.0`
  - `--actor-loss-scale 0.0`

## BC trainer extension

- Added `action_weight_boosts` support to [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py).
- This applies per-example weighting to the supervised CE term, while keeping distillation KL unchanged.
- Added parser / wrapper plumbing and regression tests.

# Validation

- `uv run pytest -q tests/test_rl_scaffold.py`
- Result: `70 passed`

# Exact Commands Run

## Dry-run validation for the no-teacher control

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_probe_noteacher_clean_a \
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
  --teacher-loss-coef 0.0 \
  --teacher-loss-final-coef 0.0 \
  --teacher-replay-coef 0.0 \
  --teacher-replay-final-coef 0.0 \
  --trace-eval-input /tmp/x100_v4_heldout_traces.jsonl \
  --trace-eval-interval-env-steps 128 \
  --trace-eval-top-k 5 \
  --save-every-sec 5 \
  --save-best-every-sec 5 \
  --no-rnn \
  --dry-run
```

Key dry-run result:

- `teacher_loss_coef = 0.0`
- `teacher_replay_coef = 0.0`

## Corrected no-teacher APPO probe

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-appo \
  --experiment appo_v4_distill_ensemble_l3pure_probe_noteacher_clean_a \
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
  --teacher-loss-coef 0.0 \
  --teacher-loss-final-coef 0.0 \
  --teacher-replay-coef 0.0 \
  --teacher-replay-final-coef 0.0 \
  --trace-eval-input /tmp/x100_v4_heldout_traces.jsonl \
  --trace-eval-interval-env-steps 128 \
  --trace-eval-top-k 5 \
  --save-every-sec 5 \
  --save-best-every-sec 5 \
  --no-rnn
```

## Offline teacher sweeps

```bash
uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_train_traces.jsonl \
  --output /tmp/x100_v4_distill_ensemble_l3_sup01_h1024.pt \
  --epochs 80 \
  --lr 0.0005 \
  --hidden-size 1024 \
  --num-layers 3 \
  --observation-version v4 \
  --distill-teacher-bc-paths /tmp/x100_v4_distill_textdistil_c020_t2_h512.pt /tmp/x100_v4_distill_textdistil_c025_t2_h512.pt \
  --distill-loss-coef 1.0 \
  --distill-temperature 2.0 \
  --supervised-loss-coef 0.1 \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

```bash
uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_train_traces.jsonl \
  --output /tmp/x100_v4_distill_ensemble_l3_sup01_east2_h1024.pt \
  --epochs 80 \
  --lr 0.0005 \
  --hidden-size 1024 \
  --num-layers 3 \
  --observation-version v4 \
  --distill-teacher-bc-paths /tmp/x100_v4_distill_textdistil_c020_t2_h512.pt /tmp/x100_v4_distill_textdistil_c025_t2_h512.pt \
  --distill-loss-coef 1.0 \
  --distill-temperature 2.0 \
  --supervised-loss-coef 0.1 \
  --action-weight-boosts east=2.0 \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

```bash
uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_train_traces.jsonl \
  --output /tmp/x100_v4_distill_ensemble_l3_pure_h1536.pt \
  --epochs 80 \
  --lr 0.0005 \
  --hidden-size 1536 \
  --num-layers 3 \
  --observation-version v4 \
  --distill-teacher-bc-paths /tmp/x100_v4_distill_textdistil_c020_t2_h512.pt /tmp/x100_v4_distill_textdistil_c025_t2_h512.pt \
  --distill-loss-coef 1.0 \
  --distill-temperature 2.0 \
  --supervised-loss-coef 0.0 \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

# Results

## Corrected no-teacher APPO control

- experiment: [appo_v4_distill_ensemble_l3pure_probe_noteacher_clean_a](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_probe_noteacher_clean_a)
- warmstart trace match: `0.9875`
- best learned checkpoint:
  - [checkpoint_000000006_1536.pth](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_probe_noteacher_clean_a/checkpoint_p0/checkpoint_000000006_1536.pth)
  - `match_rate = 0.975`
- next retained checkpoint:
  - [checkpoint_000000004_1024.pth](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_probe_noteacher_clean_a/checkpoint_p0/checkpoint_000000004_1024.pth)
  - `match_rate = 0.9625`

Interpretation:

- the corrected no-teacher control ties the earlier best learned `0.975`
- so current teacher CE/replay are not what is rescuing the short run
- the improver itself remains the bottleneck

## Disagreement pattern

Strong teacher `0.9875`:

- one mismatch total
- `seed 202, step 0`
- `east -> south`

Best no-teacher APPO checkpoint `0.975`:

- keeps the teacher’s `east -> south` miss
- adds one extra mismatch:
  - `seed 201, step 6`
  - `east -> west`

Interpretation:

- residual error is concentrated on `east` decisions
- APPO drift is not broad; it is adding a second east-family failure

## Offline teacher sweeps

- `/tmp/x100_v4_distill_ensemble_l3_sup01_h1024.pt`
  - held-out trace match: `0.975`
- `/tmp/x100_v4_distill_ensemble_l3_sup01_east2_h1024.pt`
  - held-out trace match: `0.975`
- `/tmp/x100_v4_distill_ensemble_l3_pure_h1536.pt`
  - held-out trace match: `0.9625`

Interpretation:

- adding a small supervised term hurt the current pure-distill teacher
- east-weighting the supervised term did not fix the remaining east miss
- increasing cheap-student width to `1536` was worse than the current `1024` teacher

# What Held Up

- the zero-forwarding CLI fix worked and is now test-covered
- the no-teacher control is finally scientifically valid
- the BC trainer now supports targeted supervised action weighting for future teacher experiments

# What Did Not Hold Up

- the hypothesis that teacher CE/replay were materially propping up the `0.975` short learned checkpoint
- the hypothesis that a small east-weighted supervised term would improve the `0.9875` distilled teacher
- the hypothesis that simply increasing cheap-student width would improve held-out trace match

# Recommendation

Do not promote any new run from this loop.

Current best belief:

1. The APPO-family improver is still the main bottleneck.
2. Current teacher CE/replay settings are not the decisive reason short runs reach `0.975`.
3. The strongest baseline remains:
   - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
   - `match_rate = 0.9875`

Next best move:

- stop spending short-run budget on minor PPO topology or teacher-loss toggles
- either:
  - build a genuinely more behavior-constrained online improver, or
  - find a stronger offline teacher move than the small supervised/action-weight sweep tested here
