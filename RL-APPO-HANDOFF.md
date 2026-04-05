# RL APPO Handoff

This document is a grounded handoff for the current RL workstream in this
repo. It is written against the code and experiment state in the working tree
as of 2026-04-05.


## Executive Summary

The repo now has a real learned RL backend.

That is specifically true in the narrow technical sense that:

- there is a real multi-turn NetHack environment for RL,
- there is a learned actor-critic policy,
- training runs through Sample Factory APPO,
- rollouts, recurrence, value learning, PPO-style updates, and checkpointing
  all work,
- and we have completed both smoke runs and a medium-sized run end to end.

However, the current learned policy is not yet good.

The first medium `explore` experiment completed successfully, but it did not
beat the existing heuristic controller. In fact it performed substantially
worse than `task_greedy`, and in some cases worse than `wall_avoidance`.

So the project is now in an important transition state:

- infrastructure gap for real RL is closed,
- evaluation gap is partially closed,
- policy quality gap is still very open.


## What We Had Before The APPO Work

Before the recent RL work, the repo already contained:

1. supervised fine-tuning for a forward model in [train.py](/home/luc/rl-nethack/train.py),
2. data-generation rollouts for training forward-model examples,
3. a closed-loop task harness with hand-shaped rewards in
   [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py) and
   [src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py),
4. a counterfactual one-step branching tool in
   [scripts/generate_counterfactual_data.py](/home/luc/rl-nethack/scripts/generate_counterfactual_data.py),
5. local policy-data generation with vLLM on GPUs `0,1`.

Important point: that older stack was useful, but it was not yet learned RL.
It was forward-model SFT plus evaluation/control scaffolding.


## What We Added

The new RL stack lives under [rl/](/home/luc/rl-nethack/rl).

Core files:

- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
- [rl/options.py](/home/luc/rl-nethack/rl/options.py)
- [rl/scheduler.py](/home/luc/rl-nethack/rl/scheduler.py)
- [rl/env_adapter.py](/home/luc/rl-nethack/rl/env_adapter.py)
- [rl/rewards.py](/home/luc/rl-nethack/rl/rewards.py)
- [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)
- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/bootstrap.py](/home/luc/rl-nethack/rl/bootstrap.py)

CLI entrypoint:

- [cli.py](/home/luc/rl-nethack/cli.py)
  via `rl-train-appo`

Supporting docs added earlier in the workstream:

- [CURRENT-RL-SYSTEM.md](/home/luc/rl-nethack/CURRENT-RL-SYSTEM.md)
- [APPO-OPTIONS-OVERHAUL.md](/home/luc/rl-nethack/APPO-OPTIONS-OVERHAUL.md)
- [MAESTROMOTIF-INTEGRATION.md](/home/luc/rl-nethack/MAESTROMOTIF-INTEGRATION.md)
- [RL-HARNESS-TASKS.md](/home/luc/rl-nethack/RL-HARNESS-TASKS.md)


## Current RL Architecture

### 1. Environment

The actual APPO environment is [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py).

It wraps a `SkillEnvAdapter`, which itself wraps a real `nle.env.NLE`
instance.

The environment:

- runs real NetHack episodes,
- tracks memory with `MemoryTracker`,
- tracks an active skill,
- computes shaped task reward,
- emits a compact observation vector,
- exposes a discrete action space for Sample Factory.

Current observation/action spaces:

- observation:
  - vector of shape `(106,)`
- action space:
  - `Discrete(13)`

The action set currently includes:

- `north`
- `south`
- `east`
- `west`
- `wait`
- `search`
- `pickup`
- `up`
- `down`
- `kick`
- `eat`
- `drink`
- `drop`

That action set is intentionally broad, but it is already one of the current
problems for `explore` training because invalid or useless actions are not
robustly masked during learning.


### 2. Skills / Options

Current skills are defined in [rl/options.py](/home/luc/rl-nethack/rl/options.py):

- `explore`
- `survive`
- `combat`
- `descend`
- `resource`

Each skill currently has:

- directive text,
- `can_start`,
- `should_stop`,
- `allowed_actions`.

This is the first real step toward an options hierarchy, but it is still
lightweight:

- initiation/termination logic is heuristic,
- the policy is not yet explicitly conditioned on skill in a rich way beyond
  the encoded active-skill features,
- the scheduler is still rule-based.


### 3. Scheduler

The scheduler is in [rl/scheduler.py](/home/luc/rl-nethack/rl/scheduler.py).

Right now it is purely rule-based:

- low HP -> `survive`
- visible monsters -> `combat`
- useful item cue -> `resource`
- stairs nearby -> `descend`
- otherwise -> `explore`

This is useful for bootstrapping but is not yet a learned hierarchical policy.


### 4. Reward Source

Reward wiring is in [rl/rewards.py](/home/luc/rl-nethack/rl/rewards.py), but
the actual task shaping logic comes from
[src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py).

So APPO is currently optimizing:

- hand-shaped task reward,
- optionally mixed with env reward,
- but in practice the recent runs used intrinsic hand-shaped reward only.

This is important:

- we do have learned RL now,
- but we do not yet have learned reward models,
- and we do not yet have preference-trained reward heads.


### 5. Model / Learner

The learner backend is Sample Factory APPO.

The model Sample Factory builds for the current env is:

- MLP encoder
- GRU core
- linear policy head
- linear value head

This is created automatically from the env observation/action space and config.

The current APPO stack therefore includes:

- rollout workers,
- asynchronous collection,
- PPO-style clipping,
- value loss,
- recurrence,
- checkpointing,
- GPU training.


## Dependency / Packaging State

There was an upstream dependency conflict:

- `sample-factory` expects `gymnasium < 1.0`
- `nle==1.2.0` expects `gymnasium == 1.0.0`

To avoid breaking `uv lock`, the repo now handles Sample Factory as a runtime
bootstrap dependency instead of a normal locked dependency.

That logic is in [rl/bootstrap.py](/home/luc/rl-nethack/rl/bootstrap.py).

Operationally this means:

- normal `uv run python cli.py rl-train-appo ...` works,
- the backend is installed into the project venv on demand,
- no special manual `--no-sync` path is required anymore.


## What Was Validated Before The Medium Run

We validated several layers before scaling up:

1. dry-run plan generation,
2. automatic backend bootstrap,
3. tiny serial APPO smoke training,
4. checkpoint writing,
5. manual checkpoint loading and deterministic evaluation through
   Sample Factory’s model API.

Smoke artifacts still present on disk:

- [train_dir/rl/appo_test_run/config.json](/home/luc/rl-nethack/train_dir/rl/appo_test_run/config.json)
- [train_dir/rl/appo_test_run/checkpoint_p0/checkpoint_000000006_48.pth](/home/luc/rl-nethack/train_dir/rl/appo_test_run/checkpoint_p0/checkpoint_000000006_48.pth)
- [train_dir/rl/appo_explore_smoke/config.json](/home/luc/rl-nethack/train_dir/rl/appo_explore_smoke/config.json)
- [train_dir/rl/appo_explore_smoke/checkpoint_p0/checkpoint_000000006_48.pth](/home/luc/rl-nethack/train_dir/rl/appo_explore_smoke/checkpoint_p0/checkpoint_000000006_48.pth)


## Important Bug Found During The Medium Experiment

Before the medium run, I found a real configuration bug:

- `--enabled-skills`
- `--scheduler`
- `--active_skill_bootstrap`

were being parsed at the CLI/config layer but were not fully propagated into
the Sample Factory env construction path.

That meant a command such as:

```bash
uv run python cli.py rl-train-appo --enabled-skills explore
```

could say “explore-only” at the trainer layer without truly producing an
`explore`-only env in the actual APPO process tree.

I fixed this locally in:

- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)

These changes are currently in the working tree but were not committed as part
of this handoff commit.


## Medium Experiment: What Was Run

The medium run was deliberately constrained to the `explore` skill so the
learning problem and evaluation target were clean.

Command used:

```bash
uv run python cli.py rl-train-appo \
  --experiment appo_explore_medium \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --recurrence 16 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 20000 \
  --enabled-skills explore
```

Resolved effective setup:

- backend: Sample Factory APPO
- device: GPU
- total parallel envs: `4 * 8 = 32`
- rollout length: `32`
- recurrence: `16`
- target env steps: `20000`
- reward source: `hand_shaped`
- scheduler: `rule_based`
- enabled skills: `explore`

Artifacts:

- [train_dir/rl/appo_explore_medium/config.json](/home/luc/rl-nethack/train_dir/rl/appo_explore_medium/config.json)
- [train_dir/rl/appo_explore_medium/sf_log.txt](/home/luc/rl-nethack/train_dir/rl/appo_explore_medium/sf_log.txt)
- [train_dir/rl/appo_explore_medium/checkpoint_p0/checkpoint_000000020_20480.pth](/home/luc/rl-nethack/train_dir/rl/appo_explore_medium/checkpoint_p0/checkpoint_000000020_20480.pth)
- [train_dir/rl/appo_explore_medium/checkpoint_p0/checkpoint_000000021_21504.pth](/home/luc/rl-nethack/train_dir/rl/appo_explore_medium/checkpoint_p0/checkpoint_000000021_21504.pth)

Observed training stats:

- total collected frames: `21504`
- final reported FPS: about `182.3`
- run completed successfully with status `0`


## Baseline Comparison Setup

To judge whether APPO actually learned anything useful, I compared it against
the existing closed-loop controllers on the same task.

Baselines:

- `task_greedy`
- `wall_avoidance`

Evaluation command used for baselines:

```bash
uv run python cli.py task-evaluate --task explore --seeds 42,43,44 --max-steps 50
```

Measured baseline results over seeds `42,43,44`, horizon `50`:

### `task_greedy`

- avg task reward: `37.17`
- avg env reward: `2.67`
- avg unique tiles: `74.33`
- avg rooms: `1.67`
- survival rate: `100%`

### `wall_avoidance`

- avg task reward: `-3.72`
- avg env reward: `0.00`
- avg unique tiles: `48.00`
- avg rooms: `1.00`
- survival rate: `100%`

This gives a clear target band:

- `task_greedy` is the strong local baseline,
- `wall_avoidance` is the weak baseline,
- APPO should eventually beat `wall_avoidance` and approach or exceed
  `task_greedy`.


## Medium Experiment Results

I loaded the final APPO checkpoint manually and evaluated it on the same
`explore` task over seeds `42,43,44` with horizon `50`.

Unmasked deterministic policy results:

- avg reward: `-32.867`
- avg unique tiles: `32.667`
- avg rooms: `1.0`

Episode-level behavior:

- seed `42`
  - reward `-30.7`
  - unique tiles `42`
  - top actions dominated by `drink`
- seed `43`
  - reward `-33.35`
  - unique tiles `36`
  - top action `west`
- seed `44`
  - reward `-34.55`
  - unique tiles `20`
  - top action `west`

This is a failure as a policy result.

It is far below `task_greedy` and below `wall_avoidance`.


## Action-Masking Probe

Because the unmasked policy was selecting clearly bad actions for `explore`,
I ran a second probe:

- take the learned logits,
- apply the env’s `allowed_actions` mask at inference time,
- then choose argmax.

Masked deterministic policy results:

- avg reward: `-34.45`
- avg unique tiles: `41.333`
- avg rooms: `1.0`

Behavior after masking:

- seed `42`: repeated `east`
- seed `43`: repeated `west`
- seed `44`: repeated `east`

So masking did remove obviously absurd actions like `drink`, but it did not
solve the deeper problem. The policy still collapsed to directional repetition.


## Interpretation Of The Results

The good news:

- the APPO backend is real and works,
- training on GPU works,
- checkpoint/eval loop works,
- experiment configs are reproducible,
- the stack is fast enough to iterate.

The bad news:

- the learned policy is not yet good,
- it does not beat the heuristic controller,
- it is not even consistently beating a weak baseline,
- action selection is poorly aligned with the skill intent,
- exploration still collapses into repetitive directional policies.

My current read is that this is not surprising. The current APPO setup is still
missing several things a serious policy learner needs.


## Why The Policy Is Currently Weak

The main issues, in order:

### 1. No robust action masking during training

The env exposes `allowed_actions`, and the feature vector includes an action
mask, but the Sample Factory policy is not actually constrained by that mask
during sampling or loss computation.

That means the policy can learn on a space that includes:

- `drink`
- `drop`
- `eat`
- `kick`

even in plain `explore` contexts where those actions are nonsense.

This is the most obvious immediate defect.


### 2. Exploration action space is too broad

Even with masking, the action set for `explore` may still be wider than it
should be.

For `explore`, the action space probably wants to be closer to:

- `north`
- `south`
- `east`
- `west`
- `wait`
- `search`
- `pickup`
- maybe `down`
- maybe `up`

Everything else adds distraction before the learner has basic competence.


### 3. Observation encoder is too weak

The current `(106,)` vector is compact and serviceable for plumbing, but it is
too lossy for a hard long-horizon game.

It currently captures:

- coarse scalar state,
- local adjacent tile types,
- current skill one-hot,
- action mask bits.

What it does not capture well:

- wider local map structure,
- longer memory context,
- richer threat geometry,
- frontier layout,
- inventory state in a useful way,
- recent trajectory beyond very indirect proxies.


### 4. Reward is still hand-shaped and brittle

The APPO agent is optimizing the same shaped reward stack the heuristic harness
uses. That is useful for bootstrapping, but it also means:

- reward quality is limited by hand-designed heuristics,
- we can easily induce loop-seeking or fixation,
- the reward is not learned from actual preferences or better trajectory
  comparisons.


### 5. The current scheduler is still not a learned hierarchy

Even though the repo now has “skills”, it does not yet have:

- learned option initiation,
- learned termination,
- learned high-level selection,
- robust hierarchical credit assignment.

So the policy is still much flatter than the final intended system.


### 6. No SFT-to-APPO bridge

You asked to “use the SFT model as base”.

That is not currently possible in this repo.

Why:

- the SFT model is a LoRA-tuned language forward model in
  [train.py](/home/luc/rl-nethack/train.py),
- the APPO learner is a separate actor-critic network created by Sample
  Factory,
- there is no shared architecture or weight mapping,
- and there is no current behavior-cloning or distillation path from the LM
  model into the APPO policy.

So right now the only honest answer is:

- APPO can be compared to the old stack behaviorally,
- but it cannot yet be initialized from the SFT adapter weights.


## What We Learned

This experiment was still valuable.

It tells us:

1. real RL is now operational,
2. evaluation through checkpoints is operational,
3. the current RL design is trainable but underconstrained,
4. the next bottleneck is not infra,
5. the next bottleneck is policy design and control correctness.

That is a real result.


## What Should Happen Next

### Immediate fixes

1. Implement real action masking in the APPO env/model path.
2. Reduce the effective action space for `explore`.
3. Add a first-class APPO evaluation command to compare checkpoints against
   `task_greedy` automatically.
4. Commit the local config-propagation fix in
   [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py) and
   [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py).

### Next training improvements

5. Improve the observation encoder with better local map and memory features.
6. Train on shorter horizons first and benchmark learning curves explicitly.
7. Add behavior cloning or imitation warm start from task-harness rollouts.
8. Add a policy-eval script that reports:
   - task reward
   - unique tiles
   - rooms
   - action histogram
   - repeated action rate

### Next architectural step

9. Add a BC / distillation path if we truly want “SFT as base”.
10. Move toward learned reward models and a real options hierarchy.


## Current Working Tree State At Time Of This Handoff

At the time this document was written, the working tree also contains
uncommitted changes outside this file:

- [CURRENT-RL-SYSTEM.md](/home/luc/rl-nethack/CURRENT-RL-SYSTEM.md)
- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- `train_dir/` experiment artifacts

Important:

- this handoff commit only records this document,
- it does not commit the local code fixes or artifacts,
- so the next person should inspect and either commit or refine those changes
  deliberately.


## Bottom Line

The repo has crossed an important threshold:

- it now has a real APPO RL backend,
- it can train and checkpoint learned policies,
- and it can run medium experiments on GPU.

But the first meaningful `explore` experiment did not uplift over the existing
controller.

So the next phase should not be “scale the same run harder”.

It should be:

1. fix action masking,
2. tighten the action space,
3. improve policy inputs,
4. add proper APPO evaluation,
5. then rerun the `explore` benchmark until APPO can beat `wall_avoidance`
   consistently and start approaching `task_greedy`.
