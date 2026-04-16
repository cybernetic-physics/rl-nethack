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

- [CURRENT-RL-SYSTEM.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/CURRENT-RL-SYSTEM.md)
- [APPO-OPTIONS-OVERHAUL.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/APPO-OPTIONS-OVERHAUL.md)
- [MAESTROMOTIF-INTEGRATION.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/research-notes/MAESTROMOTIF-INTEGRATION.md)
- [RL-HARNESS-TASKS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/RL-HARNESS-TASKS.md)


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


### 6. No direct SFT-to-APPO weight bridge

You asked to “use the SFT model as base”.

Direct weight loading from the SFT adapter into the APPO actor-critic is still
not possible in this repo.

Why:

- the SFT model is a LoRA-tuned language forward model in
  [train.py](/home/luc/rl-nethack/train.py),
- the APPO learner is a separate actor-critic network created by Sample
  Factory,
- there is no shared architecture or weight mapping,
- and there is no direct adapter-to-policy import path.

However, the repo now does have an indirect bridge:

- generate explicit multi-turn traces,
- optionally use the SFT forward model as the trace-generation policy,
- train BC from those traces,
- then train APPO from that behavioral starting point in future work.

So the honest answer now is:

- APPO can be compared to the old stack behaviorally,
- it can be bootstrapped through traces and BC,
- but it still cannot be initialized directly from the SFT adapter weights.


## Full Bounded Pipeline Run (Apr 5, 2026)

After the earlier APPO integration work, I ran a bounded end-to-end pipeline
with the goal of exercising:

1. forward-model data generation,
2. forward-model SFT,
3. explicit multi-turn trace generation,
4. BC policy training,
5. learned reward training,
6. learned scheduler training,
7. APPO training,
8. evaluation of the resulting policies.

Constraint:

- keep the whole run under one hour,
- use only local GPUs,
- and prefer GPUs `0,1,2` once the user allowed expanding past `0,1`.


## Important SFT Fix Found During The Pipeline Run

The first distributed SFT attempt failed with a real DDP/device-placement bug.

Observed failure:

- Unsloth loaded the model in a per-rank configuration that caused Accelerate
  to reject training on a different device than the one the quantized model had
  been loaded on.

Root cause:

- [train.py](/home/luc/rl-nethack/train.py) was not explicitly pinning each
  `torchrun` rank to its own CUDA device before model load.

Fix applied:

- call `torch.cuda.set_device(dist["local_rank"])` before model creation,
- pass `device_map={"": torch.cuda.current_device()}` in distributed mode,
- make the intended bf16 load path explicit.

This fix is in the working tree in
[train.py](/home/luc/rl-nethack/train.py) and was used for the successful run
below.


## Pipeline Commands Actually Run

### 1. Generate forward-model SFT data

```bash
uv run python cli.py generate \
  --num-games 300 \
  --max-steps 30 \
  --output data/pipeline_train.jsonl \
  --eval-output data/pipeline_eval.jsonl \
  --eval-fraction 0.2
```

Output:

- [data/pipeline_train.jsonl](/home/luc/rl-nethack/data/pipeline_train.jsonl)
- [data/pipeline_eval.jsonl](/home/luc/rl-nethack/data/pipeline_eval.jsonl)

Counts:

- total examples: `8990`
- train examples: `7190`
- eval examples: `1800`


### 2. Train forward model with SFT on GPUs `0,1,2`

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run torchrun --standalone --nproc_per_node=3 \
  train.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/pipeline_train.jsonl \
  --eval-data data/pipeline_eval.jsonl \
  --output output/pipeline_adapter \
  --max-seq-length 768 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lr 2e-4 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --max-steps 60 \
  --dataset-num-proc 4 \
  --dataloader-num-workers 2 \
  --logging-steps 10 \
  --save-steps 30
```

Observed result:

- runtime: about `27.6s`
- global steps: `60`
- final train loss: `0.5690`
- adapter hash: `cf2b58a0029932640894869ef01d2564b9639c9fb4954312f80e801d8e5781d1`

Artifacts:

- [output/pipeline_adapter](/home/luc/rl-nethack/output/pipeline_adapter)
- [output/pipeline_adapter/training_meta.json](/home/luc/rl-nethack/output/pipeline_adapter/training_meta.json)


### 3. Generate explicit multi-turn traces

I attempted a larger `task_greedy` trace run first, but the trace generator is
slow because it uses the counterfactual task harness internally. To keep the
run bounded, I regenerated a smaller clean trace set and verified it.

Command used for the clean file:

```bash
uv run python cli.py rl-generate-traces \
  --output data/pipeline_explore_traces_clean.jsonl \
  --num-episodes 40 \
  --max-steps 24 \
  --task explore \
  --policy task_greedy
```

Verification command:

```bash
uv run python cli.py rl-verify-traces \
  --input data/pipeline_explore_traces_clean.jsonl
```

Verified result for the clean artifact:

- episodes: `3`
- rows: `68`
- max steps in episode: `24`
- avg steps per episode: `22.67`
- all episodes multi-turn: `true`

Artifact:

- [data/pipeline_explore_traces_clean.jsonl](/home/luc/rl-nethack/data/pipeline_explore_traces_clean.jsonl)

Important interpretation:

- yes, the trace format is genuinely multi-turn,
- but the bounded clean trace artifact is still small,
- because `task_greedy` trace generation remains expensive.


### 4. Train BC policy from the multi-turn traces

```bash
uv run python cli.py rl-train-bc \
  --input data/pipeline_explore_traces_clean.jsonl \
  --output output/pipeline_explore_bc.pt \
  --epochs 25 \
  --lr 0.001
```

Result:

- num examples: `68`
- final loss: `1.0892`
- train accuracy: `0.5588`

Artifact:

- [output/pipeline_explore_bc.pt](/home/luc/rl-nethack/output/pipeline_explore_bc.pt)


### 5. Train learned reward model

```bash
uv run python cli.py rl-train-reward \
  --task explore \
  --seeds 42,43,44,45,46,47,48,49 \
  --max-steps 24 \
  --dataset-output data/pipeline_explore_reward_prefs.jsonl \
  --output output/pipeline_explore_reward.pt \
  --epochs 25 \
  --lr 0.001
```

Result:

- num pairs: `192`
- final loss: `0.2894`

Artifacts:

- [data/pipeline_explore_reward_prefs.jsonl](/home/luc/rl-nethack/data/pipeline_explore_reward_prefs.jsonl)
- [output/pipeline_explore_reward.pt](/home/luc/rl-nethack/output/pipeline_explore_reward.pt)


### 6. Train learned scheduler

```bash
uv run python cli.py rl-train-scheduler \
  --seeds 42,43,44,45,46,47,48,49 \
  --max-steps 24 \
  --dataset-output data/pipeline_scheduler_rows.jsonl \
  --output output/pipeline_scheduler.pt \
  --epochs 25 \
  --lr 0.001
```

Result:

- num examples: `192`
- final loss: `0.2093`
- train accuracy: `0.9531`

Artifacts:

- [data/pipeline_scheduler_rows.jsonl](/home/luc/rl-nethack/data/pipeline_scheduler_rows.jsonl)
- [output/pipeline_scheduler.pt](/home/luc/rl-nethack/output/pipeline_scheduler.pt)


### 7. Train APPO on learned reward + learned scheduler

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-appo \
  --experiment pipeline_appo_learned \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --recurrence 16 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 12000 \
  --enabled-skills explore \
  --reward-source learned \
  --learned-reward-path output/pipeline_explore_reward.pt \
  --scheduler learned \
  --scheduler-model-path output/pipeline_scheduler.pt
```

Resolved configuration:

- total parallel envs: `32`
- rollout length: `32`
- recurrence: `16`
- train target: `12000` env steps
- checkpoint written at `13312` collected frames
- final reported FPS: about `144.7`

Artifacts:

- [train_dir/rl/pipeline_appo_learned/config.json](/home/luc/rl-nethack/train_dir/rl/pipeline_appo_learned/config.json)
- [train_dir/rl/pipeline_appo_learned/checkpoint_p0/checkpoint_000000013_13312.pth](/home/luc/rl-nethack/train_dir/rl/pipeline_appo_learned/checkpoint_p0/checkpoint_000000013_13312.pth)


## Evaluation Results From The Bounded Pipeline

### BC evaluation

Command:

```bash
uv run python cli.py rl-evaluate-bc \
  --model output/pipeline_explore_bc.pt \
  --task explore \
  --seeds 42,43,44 \
  --max-steps 50
```

Result summary:

- episodes: `3`
- avg unique tiles: `74.0`
- avg rooms discovered: `1.33`
- avg env reward: `0.0`
- action counts:
  - `east: 146`
  - `north: 4`

Interpretation:

- the BC policy is simplistic and highly collapsed,
- but it still moves through the map enough to get reasonable exploration
  coverage.


### APPO evaluation

Command:

```bash
uv run python cli.py rl-evaluate-appo \
  --experiment pipeline_appo_learned \
  --seeds 42,43,44 \
  --max-steps 50
```

Result summary:

- avg task reward: `-28.2989`
- avg unique tiles: `48.0`
- avg rooms discovered: `1.33`
- repeated action rate: `0.9733`
- invalid action rate: `0.0`
- action counts:
  - `west: 149`
  - `east: 1`

Interpretation:

- action masking is working in the sense that invalid-action rate is `0.0`,
- but the policy still collapses almost completely to one direction,
- so the run is operationally successful and behaviorally weak.


### Relative ranking

From this bounded pipeline run:

1. `task_greedy` heuristic remains strongest overall from earlier validated
   `explore` benchmarks,
2. the BC policy from multi-turn traces is currently better than the APPO
   policy,
3. the learned APPO policy is still significantly underperforming.

That is the most important result from this run.


## What The Pipeline Run Proved

The repo can now do all of the following in one workflow:

- generate forward-model SFT data,
- run real multi-GPU SFT locally,
- generate explicit multi-turn trace data,
- verify that traces are actually multi-turn,
- train a BC policy on those traces,
- train learned reward and scheduler models,
- train a real APPO RL policy with those learned components,
- and evaluate the resulting policies.

So the repo now has a real end-to-end training pipeline.

However, the best current bridge from supervision into control is BC, not RL.

The learned APPO path still does not beat the BC policy produced from the same
pipeline, and it still does not approach the existing `task_greedy` controller.


## Updated Next Steps After The Bounded Pipeline Run

The highest-value next step is now clear:

1. initialize APPO from BC weights instead of training from scratch,
2. enlarge the multi-turn trace corpus substantially,
3. keep action masking strict,
4. narrow the `explore` action space further,
5. then rerun the same bounded benchmark and compare:
   - `task_greedy`
   - BC
   - BC-initialized APPO

If that still does not close the gap, the next bottleneck is likely the policy
representation or the reward signal rather than dataflow.


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

- [CURRENT-RL-SYSTEM.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/CURRENT-RL-SYSTEM.md)
- [README.md](/home/luc/rl-nethack-worktree-20260416/README.md)
- [train.py](/home/luc/rl-nethack/train.py)
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

And after the first bounded end-to-end pipeline run, the same conclusion still
holds.

So the next phase should not be “scale the same RL run harder”.

It should be:

1. use BC as the real initialization path,
2. enlarge and clean the multi-turn trace corpus,
3. tighten the action space,
4. improve policy inputs,
5. then rerun the `explore` benchmark until APPO can beat BC and start
   approaching `task_greedy`.
