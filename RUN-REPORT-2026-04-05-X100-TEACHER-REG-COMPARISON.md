# X100 Run Comparison: Teacher-Reg Only vs Teacher-Reg + Episodic Bonus

Date: `2026-04-05`

## Scope

This report compares the last two long APPO validation runs on the repaired non-RNN path:

1. `appo_teacher_reg_001_x100`
2. `appo_teacher_bonus_001_001_x100`

Both runs used:

- BC warm start from [output/tracefix_v2_explore_bc.pt](/home/luc/rl-nethack/output/tracefix_v2_explore_bc.pt)
- frozen BC teacher regularization with `teacher_loss_coef = 0.01`
- `explore` only
- `v2` observations
- non-RNN actor-critic
- GPU `2`

The only intentional difference was the episodic exploration bonus:

- `appo_teacher_reg_001_x100`: no episodic bonus
- `appo_teacher_bonus_001_001_x100`: `state_hash` episodic bonus with scale `0.01`

This comparison is grounded in the current repo state after commit `df881ee` (`Implement teacher-reg APPO and episodic bonus`).

## Why This Comparison Matters

The current repo has a known objective-alignment problem:

- live training reward is not the trusted model-selection metric
- deterministic trace match against the repaired teacher trace set is the trusted metric

So these runs were evaluated primarily by:

- checkpoint ranking on [data/tracefix_v2_explore_traces.jsonl](/home/luc/rl-nethack/data/tracefix_v2_explore_traces.jsonl)

This report intentionally does **not** treat rising APPO reward as success by itself.

## Run 1: Teacher-Reg Only

Experiment:

- `appo_teacher_reg_001_x100`

Command:

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-appo \
  --experiment appo_teacher_reg_001_x100 \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --recurrence 16 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 1024000 \
  --enabled-skills explore \
  --observation-version v2 \
  --bc-init-path output/tracefix_v2_explore_bc.pt \
  --teacher-bc-path output/tracefix_v2_explore_bc.pt \
  --teacher-loss-coef 0.01 \
  --teacher-loss-type ce \
  --no-rnn
```

Key ranked checkpoints:

- [checkpoint_000000019_19456.pth](/home/luc/rl-nethack/train_dir/rl/appo_teacher_reg_001_x100/checkpoint_p0/checkpoint_000000019_19456.pth): `0.6085`
- [checkpoint_000000043_44032.pth](/home/luc/rl-nethack/train_dir/rl/appo_teacher_reg_001_x100/checkpoint_p0/checkpoint_000000043_44032.pth): `0.6202`
- [checkpoint_000000068_69632.pth](/home/luc/rl-nethack/train_dir/rl/appo_teacher_reg_001_x100/checkpoint_p0/checkpoint_000000068_69632.pth): `0.6279`
- [checkpoint_000000092_94208.pth](/home/luc/rl-nethack/train_dir/rl/appo_teacher_reg_001_x100/checkpoint_p0/checkpoint_000000092_94208.pth): `0.6318`
- [checkpoint_000000116_118784.pth](/home/luc/rl-nethack/train_dir/rl/appo_teacher_reg_001_x100/checkpoint_p0/checkpoint_000000116_118784.pth): `0.6085`

Best trusted checkpoint:

- [checkpoint_000000092_94208.pth](/home/luc/rl-nethack/train_dir/rl/appo_teacher_reg_001_x100/checkpoint_p0/checkpoint_000000092_94208.pth)
- deterministic trace match: `0.6318`

Action counts at best checkpoint:

- `north: 55`
- `east: 67`
- `south: 46`
- `west: 89`
- `search: 1`

Observed trajectory:

- early phase was modestly better than warm start
- mid-run kept improving on the trusted trace benchmark
- after about `94k` frames, live reward continued to rise
- but the trusted trace metric regressed by the next checkpoint

Conclusion for Run 1:

- teacher regularization at `0.01` is real and useful
- scale helped
- this run matched the best repaired APPO line seen so far
- but it still did not beat the BC teacher at `0.6395`

## Run 2: Teacher-Reg + State-Hash Bonus 0.01

Experiment:

- `appo_teacher_bonus_001_001_x100`

Command:

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-appo \
  --experiment appo_teacher_bonus_001_001_x100 \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --recurrence 16 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 1024000 \
  --enabled-skills explore \
  --observation-version v2 \
  --bc-init-path output/tracefix_v2_explore_bc.pt \
  --teacher-bc-path output/tracefix_v2_explore_bc.pt \
  --teacher-loss-coef 0.01 \
  --teacher-loss-type ce \
  --episodic-explore-bonus-enabled \
  --episodic-explore-bonus-scale 0.01 \
  --episodic-explore-bonus-mode state_hash \
  --no-rnn
```

Key ranked checkpoints:

- [checkpoint_000000019_19456.pth](/home/luc/rl-nethack/train_dir/rl/appo_teacher_bonus_001_001_x100/checkpoint_p0/checkpoint_000000019_19456.pth): `0.5891`
- [checkpoint_000000042_43008.pth](/home/luc/rl-nethack/train_dir/rl/appo_teacher_bonus_001_001_x100/checkpoint_p0/checkpoint_000000042_43008.pth): `0.6085`

Best trusted checkpoint:

- [checkpoint_000000042_43008.pth](/home/luc/rl-nethack/train_dir/rl/appo_teacher_bonus_001_001_x100/checkpoint_p0/checkpoint_000000042_43008.pth)
- deterministic trace match: `0.6085`

Action counts at best checkpoint:

- `north: 73`
- `east: 64`
- `south: 36`
- `west: 83`
- `wait: 2`

Observed trajectory:

- the run started substantially worse than the no-bonus line
- it recovered from `0.5891` to `0.6085`
- but it stayed well behind the no-bonus teacher-reg run at comparable frame counts
- it never entered the same quality regime as the no-bonus run

Conclusion for Run 2:

- the `state_hash` bonus with scale `0.01` is stable
- but it is not helping under the current objective and representation
- at x100 scale it is still weaker than teacher-regularized APPO without the bonus

## Head-to-Head Summary

Trusted baseline:

- BC teacher on [data/tracefix_v2_explore_traces.jsonl](/home/luc/rl-nethack/data/tracefix_v2_explore_traces.jsonl): `0.6395`

Best results:

- teacher-reg only x100: `0.6318`
- teacher-reg + state-hash bonus `0.01` x100: `0.6085`

Gap to BC:

- teacher-reg only x100 gap to BC: `0.0077`
- teacher-reg + bonus x100 gap to BC: `0.0310`

At comparable medium checkpoints:

- teacher-reg only at `44,032` frames: `0.6202`
- teacher-reg + bonus at `43,008` frames: `0.6085`

So the bonus version is not only failing to improve the best score; it is also trailing the no-bonus version throughout the useful part of training.

## What Worked

### 1. The repaired fast loop worked

These runs were practical because the harness now supports:

- deterministic trace ranking
- best-trace checkpoint materialization
- no-RNN BC-compatible APPO warm starts
- trustworthy comparison against the teacher trace set

Without that, the higher APPO reward in the later parts of these runs would have looked like progress.

### 2. Teacher regularization is a real lever

The no-bonus run showed that:

- small teacher loss does not just preserve the warm start
- it can improve the policy under scale
- it can close most of the gap to BC

That is the strongest positive result from these two runs.

### 3. The episodic bonus code path is operationally correct

The episodic bonus did not crash training.
It integrated cleanly into the env reward path and remained easy to sweep and disable.

So this is not an implementation failure.
It is a training-signal failure.

## What Did Not Work

### 1. The state-hash bonus did not improve the real objective

The main intended reason for Part 2 was:

- improve frontier-seeking behavior
- reduce shallow repetition
- help the learner under sparse gradients

That did not happen on the trusted trace benchmark.

### 2. Training reward remained a misleading indicator

Both lines showed again that:

- live APPO reward can rise sharply
- while trusted trace behavior plateaus or regresses

That is still the core objective-alignment issue in this repo.

### 3. Directional bias remains

Even the best teacher-reg checkpoint still over-predicts `west`.
The episodic bonus did not solve that.

This suggests the problem is not just “insufficient novelty reward.”
It is likely a combination of:

- imperfect reward proxy
- directional representation weakness
- policy optimization drift away from the teacher target

## Interpretation

The two-run comparison strongly suggests:

- teacher regularization is helping because it directly addresses the teacher-drift problem
- the current episodic bonus is not helping because it is just another proxy layered on top of an already misaligned objective

In other words:

- Part 1 improved the alignment of the optimization target
- Part 2 improved neither alignment nor representation enough to matter

That is why the no-bonus line beats the bonus line despite both having the same teacher constraint.

## Practical Takeaways

1. Keep teacher-regularized APPO as the current best RL training line.
2. Leave episodic bonus support in the codebase, but disabled by default.
3. Do not spend more sweep budget on the current `state_hash` bonus design.
4. Use [checkpoint_000000092_94208.pth](/home/luc/rl-nethack/train_dir/rl/appo_teacher_reg_001_x100/checkpoint_p0/checkpoint_000000092_94208.pth) as the current best teacher-reg APPO checkpoint from these runs.

## Recommended Next Steps

The next high-value work should not be “try a few more novelty scales.”

Instead:

1. add stronger teacher-guided improvement beyond simple CE regularization
   - example: DAgger-style student-state relabeling loops
   - example: behavior-regularized RL objectives

2. improve directional observation quality before more reward tinkering
   - the current persistent `west` skew is more consistent with representation issues than with missing novelty

3. continue selecting models by deterministic trace match, never by training reward alone

4. if intrinsic reward is revisited, it should probably be a more structured frontier-progress signal rather than the current generic state-hash count bonus

## Bottom Line

From the last two x100 runs:

- teacher-regularized APPO is the better path
- the `state_hash` episodic bonus at `0.01` does not help
- the best current RL checkpoint is close to BC but still below it
- the remaining gap looks scientific, not infrastructural

Best current numbers:

- BC teacher: `0.6395`
- best teacher-reg x100 APPO: `0.6318`
- best teacher-reg + bonus x100 APPO: `0.6085`
