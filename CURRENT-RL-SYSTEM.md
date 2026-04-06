# Current RL System in This Repo

This document explains what the repo currently does, grounded in the code as it
exists today.

The short version:

- this repo now has a **real APPO backend** for low-level RL training,
- it now also has an **offline short-horizon world-model path** that can
  augment policy features,
- it also has:
  - supervised fine-tuning for a forward model,
  - gameplay rollout code for collecting training data,
  - a task-directed closed-loop control and evaluation harness,
  - a counterfactual rollout generator for "what if I took action X?" data,
  - a custom NetHack skill environment wired into Sample Factory APPO.

If you came in expecting a standard RL codebase with rollout workers,
advantages, value heads, clipped objectives, and policy updates: that now
**does exist in early form**, but only for the new APPO path. The rest of the
repo still contains older non-RL and pre-RL systems.


## 1. What Is Actually Trained Today

There are now **two distinct training paths** in the repo.

The repo also now has a third learned component that sits between imitation and
RL:

- a **short-horizon latent world model**

So the practical stack is now:

1. forward-model SFT
2. BC / behavior-reg teacher training
3. world-model training and feature augmentation
4. teacher-regularized APPO

### Path A: forward-model supervised training

This is in [train.py](/home/luc/rl-nethack/train.py).

That script uses:

- `trl.SFTTrainer`
- Unsloth LoRA
- ShareGPT-style JSONL conversations

It trains a language model to do **next-step outcome prediction**.

More precisely:

- input:
  - system prompt
  - user prompt containing state + action
- target:
  - assistant text describing the resulting delta

So the main learned model today is a **forward model**, not a policy network.

The trainer is configured in [train.py](/home/luc/rl-nethack/train.py):

- `per_device_train_batch_size = --batch-size`
- `gradient_accumulation_steps = --gradient-accumulation-steps`
- `num_train_epochs = --epochs`
- `max_steps = --max-steps`

There is no RL loss here. No returns, no GAE, no policy ratio, no clipped
objective, no reward model training, no rollout buffer.

### Path B: APPO low-level RL training

This is in:

- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)

This path uses:

- Sample Factory
- APPO
- a custom Gymnasium env wrapping NLE
- hand-shaped task rewards projected into a skill-conditioned RL env

The APPO path does have:

- rollout workers
- rollout length
- recurrence
- actor-critic training
- value loss
- GAE
- PPO-style clipping

So the repo now has **real learned RL**, but only in this new RL subtree.

### Path C: short-horizon world-model training

This is in:

- [rl/world_model.py](/home/luc/rl-nethack/rl/world_model.py)
- [rl/world_model_dataset.py](/home/luc/rl-nethack/rl/world_model_dataset.py)
- [rl/train_world_model.py](/home/luc/rl-nethack/rl/train_world_model.py)
- [rl/world_model_eval.py](/home/luc/rl-nethack/rl/world_model_eval.py)
- [rl/world_model_features.py](/home/luc/rl-nethack/rl/world_model_features.py)

This path trains a small MLP world model on trace rows. It is **not** a
Dreamer-style end-to-end RL replacement. It is a short-horizon predictive model
used to improve feature quality.

The world model currently learns from trace tuples:

- current feature vector
- current action
- current task / skill
- future feature vector after `K` steps
- cumulative reward over that short horizon
- done-within-horizon flag

The current model has:

- an encoder producing a latent state
- a transition model conditioned on action and skill
- heads for:
  - future features
  - reward
  - done
  - current-feature reconstruction
  - current-action classification

That last pair matters: they were added specifically because a pure predictive
latent was too weak as a policy feature.


## 2. What "Rollout" Means In This Repo

The word "rollout" appears in a few places, but it does **not** mean the same
thing as "policy rollouts used by an RL learner" in a PPO codebase.

In this repo today, a "rollout" usually means:

- run a NetHack episode for some number of steps,
- collect transitions,
- maybe query an LLM for the next action,
- write training/eval examples,
- or score task behavior.

There are now four main rollout/training systems.

With the world-model path added, there are effectively **five** systems:

1. SFT data generation
2. counterfactual branching
3. task harness / task-greedy control
4. BC / behavior-reg / teacher traces
5. APPO RL

The world model itself trains on trace datasets produced by those systems. It
does not collect its own environment rollouts.


## 3. System A: LLM / Policy Data Generation

This is in
[scripts/generate_training_data.py](/home/luc/rl-nethack/scripts/generate_training_data.py).

### What it does

It runs NetHack games, chooses actions with either:

- a remote API model,
- a local OpenAI-compatible server such as vLLM,
- or in-process batched vLLM,

and writes supervised training pairs for the forward model.

Each pair is:

- prompt: enriched state + chosen action
- target: predicted delta after that action

### What counts as a rollout here

In this script, **one game = one rollout**.

This is represented by the `GameRollout` dataclass in
[scripts/generate_training_data.py](/home/luc/rl-nethack/scripts/generate_training_data.py).

Each rollout contains:

- one NLE env
- one current observation
- one `MemoryTracker`
- action history
- generated pairs
- cumulative reward
- step count

### Are these multi-turn rollouts?

Yes.

This path is definitely multi-turn.

Each game runs for up to `--max-steps` turns, or until termination/truncation.
So if you run:

```bash
uv run python scripts/generate_training_data.py --num-games 100 --max-steps 50
```

you are asking for up to:

- `100` rollouts
- each rollout up to `50` turns long

### What is "group size" here?

There is no PPO-style "group size".

The closest things are:

1. `--workers`
   - number of concurrent game rollouts
   - in the threaded path, this is the size of the `ThreadPoolExecutor`
   - in the in-process batched vLLM path, this is also the number of active
     game rollouts advanced together per turn batch

2. `--vllm-max-num-seqs`
   - maximum number of sequences the in-process vLLM engine may batch
   - this is an inference batching setting, not an RL grouping setting

So:

- `workers` = concurrent episodes / rollouts
- not "group size" in the GRPO sense

### Important implication

This script does **not** learn from the rollouts.
It only **collects supervised data** from them.


## 4. System B: Counterfactual Rollout Generator

This is in
[scripts/generate_counterfactual_data.py](/home/luc/rl-nethack/scripts/generate_counterfactual_data.py).

### What it does

It scans a running game for "interesting moments" such as:

- combat
- threat
- survival
- descent
- pickup

Then, at those moments, it forks the env and tries multiple actions from the
same state.

### What counts as a rollout here

There are two levels:

1. outer game rollout
   - one live NetHack game per seed
2. counterfactual micro-rollouts
   - one-step branches from the same state

### Are these multi-turn rollouts?

The outer game is multi-turn.

The counterfactual branches are **one-step only**.

The branch action list is currently:

- `north`
- `south`
- `east`
- `west`
- `northeast`
- `northwest`
- `southeast`
- `southwest`
- `wait`

So at each interesting state, it can generate up to **9 one-step branches**.

### What is the group size here?

If by group size you mean "how many alternatives are compared from one state",
the answer is:

- up to `9` counterfactual actions per interesting moment

But again, this is not GRPO or grouped preference optimization. It is just
branching one env state into multiple one-step futures.

### What is it used for?

Right now it is a data-generation / analysis tool, not a training loop.


## 5. System C: Task-Directed Closed-Loop Harness

This is in
[src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py) and
[src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py).

This is the closest thing in the repo to an RL control harness.

### What it does

It defines shaped task rewards for:

- `explore`
- `survive`
- `combat`

## 6. System D: BC / Teacher Policies

This is in:

- [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py)
- [rl/bc_model.py](/home/luc/rl-nethack/rl/bc_model.py)
- [rl/evaluate_bc.py](/home/luc/rl-nethack/rl/evaluate_bc.py)
- [rl/train_behavior_reg.py](/home/luc/rl-nethack/rl/train_behavior_reg.py)

This path is still the strongest aligned policy path in the repo.

What it does:

- train a policy from trace rows with action masking
- optionally use stronger offline behavior-regularized training
- evaluate either:
  - live in NLE, or
  - deterministically on trace datasets

The trusted metric is still deterministic trace match, not live reward.

That matters for the world-model work too, because the world model is judged by
whether it helps the teacher/policy match traces better, not by whether its raw
prediction loss is low.


## 7. System E: World-Model Feature Augmentation

This is the new bridge between offline training and RL.

### What it does

The world model can now be used in three modes:

- `replace`
  - replace the original feature vector with the latent only
- `concat`
  - append the latent to the original feature vector
- `concat_aux`
  - append:
    - the latent
    - the action-logit auxiliary head
    to the original feature vector

`concat_aux` is the only mode that currently looks promising.

### How it helps

The world model helps the repo in a narrower way than a full model-based RL
system.

It helps by giving the policy a feature space that tries to encode:

- short-horizon future structure
- action-relevant latent structure
- some current-state reconstruction pressure

So in practice the world model is being used as:

- **representation learning**
- **feature augmentation**

not as:

- a standalone planner
- a replacement RL algorithm

### Why we added reconstruction and action heads

The first world-model version only predicted short-horizon future outcomes.
That produced latents that were too lossy for policy learning.

Symptoms:

- BC on latent-only traces collapsed badly
- simple latent concatenation also underperformed

So the current world model was changed to also predict:

- current features
- current action

This makes the latent carry more policy-relevant information.

### What is the current status?

On older `v2` traces, world-model features are still weak.

On the stronger `v4` teacher traces, `concat_aux` is finally good enough to be
useful:

- offline BC on held-out transformed traces reached about `0.9375` trace match

That is strong enough to say the feature path works.

But online RL still drifts once training starts. So the world model currently:

- helps **representation**
- does **not** yet solve objective drift in APPO


## 8. How The Live RL Env Uses The World Model

The live RL env path is:

- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)
- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)

The env now supports:

- `world_model_path`
- `world_model_feature_mode`

So APPO can train directly on:

- the base observation encoding, or
- the augmented observation produced by the world model

This was the important integration step. Before that, the offline transformed
trace path and the live env path were not actually using the same feature space.

That mismatch is now fixed.


## 9. What The World Model Does Not Do Yet

The repo does **not** yet have:

- imagination rollouts inside RL
- Dreamer-style latent planning
- MuZero-style search
- skill-level world-model planning

So if you are reading the code and asking "is this model-based RL now?",
the honest answer is:

- **not really**

It is still mainly an imitation + teacher-regularized APPO repo, with a new
world-model-assisted feature path.


## 10. Current Practical Interpretation

Today, the repo has:

- a real APPO backend
- a strong offline teacher path
- a trusted deterministic trace metric
- a working world-model feature augmentation path

The main open problem is still:

- online RL drifts away from the teacher too quickly

So the world model currently helps us by improving the policy input space, but
the repo is still bottlenecked by:

- RL objective alignment
- value/reward scaling at longer horizons
- preserving the strong offline teacher during online updates

That is why the world model is useful now, but not yet the main control
mechanism.
- `descend`
- `resource`

Then it runs real episodes and either:

- uses the old heuristic `wall_avoidance` policy, or
- uses `task_greedy`, a one-step counterfactual controller

### Are these multi-turn rollouts?

Yes.

`run_task_episode(...)` in
[src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py) runs a full
episode up to `max_steps`.

So here:

- one seed = one episode rollout
- each rollout is multi-turn

### How does the controller work?

For `task_greedy`, at each time step:

1. build a candidate action list
2. fork the current env once per candidate action
3. step each branch one turn
4. compute shaped reward for that one-step outcome
5. choose the best action
6. execute that action in the real episode

So the controller is:

- closed-loop
- online
- task-conditioned
- **one-step greedy**

It is **not learned RL yet**.

### What is the action group size here?

The action group is dynamic and comes from `_candidate_actions(...)` in
[src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py).

Base candidates:

- `north`
- `south`
- `east`
- `west`
- `wait`
- `search`

Sometimes also:

- `pickup`
- `up`
- `down`

So the task harness usually compares **6 to 9 candidate actions per state**.

That is the closest thing in the current repo to "group size".

### Important limitation

This controller does not do:

- multi-step search
- value backup
- learned reward modeling
- policy gradient
- actor-critic training

It only does **one-step counterfactual selection**.


## 6. System D: Sample Factory APPO Backend

This is in:

- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)
- [rl/env_adapter.py](/home/luc/rl-nethack/rl/env_adapter.py)
- [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)

### What it does

It trains a low-level policy with APPO against a custom NetHack skill env.

The APPO env currently:

- wraps NLE
- tracks memory via `MemoryTracker`
- tracks an active skill
- computes a compact vector observation
- uses the repo’s hand-shaped task rewards as the main training signal

### Are these multi-turn rollouts?

Yes.

This is the first true RL rollout system in the repo.

Rollouts are controlled by Sample Factory parameters such as:

- `num_workers`
- `num_envs_per_worker`
- `rollout`
- `recurrence`

These are now real RL rollout knobs, not just data-generation concurrency.

### What is learned?

Sample Factory builds and trains an actor-critic model.

In the current smoke-tested setup it used:

- observation space:
  - `Dict('obs': Box(-10.0, 10.0, (106,), float32))`
- action space:
  - `Discrete(13)`
- model:
  - MLP encoder
  - GRU core
  - linear policy head
  - linear value head

### What is the policy optimizing?

Right now:

- intrinsic/task reward from the repo’s hand-shaped task reward functions
- optionally mixed with env reward through:
  - `intrinsic_reward_weight`
  - `extrinsic_reward_weight`

So the APPO path is learned RL, but the reward source is still the old
hand-shaped task logic, not a learned reward model.

### What is "group size" here?

Still no GRPO-style group size.

The important RL knobs are now:

- `num_workers`
- `num_envs_per_worker`
- `rollout`
- `recurrence`
- `batch_size`
- `num_batches_per_epoch`
- `num_epochs`

Those are the real APPO dataflow parameters.


## 7. Task Rewards: What The Harness And APPO Env Optimize

Task rewards are defined in
[src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py).

These are repo-defined shaped rewards, not raw NLE score.

Examples:

- `explore`
  - rewards new tiles, new rooms, stairs seen, useful items seen
  - penalizes repeated states, repeated actions, revisits, invalid actions
- `survive`
  - penalizes death and HP loss
  - rewards HP gain
  - penalizes repeated bad looping
- `combat`
  - rewards kill-like outcomes and safe threat resolution
- `descend`
  - rewards stairs and depth gain
- `resource`
  - rewards useful pickup / gains

This is important:

- the harness is already task-conditioned,
- the APPO env also currently consumes these rewards,
- but the reward is still **hand-shaped**, not learned from feedback.


## 8. Evaluation: What We Measure Today

There are currently two very different evaluation paths.

### Open-loop forward-model evaluation

In [src/evaluator.py](/home/luc/rl-nethack/src/evaluator.py).

This measures how well a model predicts the next-step delta text.

Metrics include:

- exact-match rate
- position accuracy
- HP accuracy
- gold accuracy
- depth accuracy
- survived accuracy

This is not RL evaluation. It is next-step prediction evaluation.

### Closed-loop task evaluation

Via `cli.py task-evaluate`, backed by
[src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py).

This measures trajectory behavior:

- total task reward
- total env reward
- unique tiles
- rooms discovered
- final HP
- final depth
- final gold
- repeated state rate
- repeated action rate
- action counts

This is the current behavior-level evaluation layer.

### APPO smoke validation

The new APPO backend has been smoke-tested through a real run launched from
`cli.py rl-train-appo`.

That run successfully:

- registered the custom env,
- initialized the actor-critic,
- collected experience,
- trained,
- and wrote checkpoints/config under `train_dir/rl/...`

This is backend validation, not yet a meaningful benchmark of policy quality.


## 9. What "Batch Size" Means In This Repo

There are several unrelated batch-like knobs.

### Training batch size

In [train.py](/home/luc/rl-nethack/train.py):

- `--batch-size`
  - per-device SFT batch size
- `--gradient-accumulation-steps`
  - accumulation before optimizer step

Global batch is explicitly computed as:

`batch_size * gradient_accumulation_steps * world_size`

This is **training batch size for supervised fine-tuning**, not RL rollout
batch size.

### Rollout concurrency

In [scripts/generate_training_data.py](/home/luc/rl-nethack/scripts/generate_training_data.py):

- `--workers`
  - number of simultaneous games / rollouts

This is data collection concurrency, not learner batch size.

### Inference batch size

In the `vllm-batch` backend:

- active games are batched together each turn
- `--vllm-max-num-seqs` caps inference concurrency inside vLLM

This is inference throughput tuning, not RL grouping.

### APPO learner batch size

In [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py) and
[rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py):

- `--batch-size`
  - APPO minibatch size
- `--num-batches-per-epoch`
  - how many minibatches are collected before each training iteration
- `--ppo-epochs`
  - how many passes over that dataset

This is now a real RL batch concept in the repo.


## 10. What Is Missing If You Expect A Mature RL Stack

The repo now **does** have APPO.

But it still does **not yet** have:

- GRPO
- replay buffer / rollout buffer
- reward model training
- learned option / skill policy conditioning in the model itself
- learned high-level policy over skills
- learned reward models from preferences
- robust action masking inside the APPO policy
- a richer observation encoder than the current compact 106-dim vector
- serious multi-GPU / high-throughput APPO configs for this box
- real benchmark results against `task_greedy`

What it does have is:

- episode rollouts for data collection
- one-step counterfactual branching
- hand-shaped task reward functions
- trajectory evaluation
- a real APPO actor-critic training path

So the current stack is best described as:

1. collect trajectories,
2. train a forward model with SFT,
3. train a low-level APPO policy on hand-shaped task reward,
4. evaluate closed-loop task behavior with both a hand-shaped controller and a learned RL backend,
5. prepare for hierarchical / reward-model / scheduler work.


## 11. Practical Answers To Your Specific Questions

### "How many rollouts are there?"

Depends on which subsystem you mean.

- `generate_training_data.py`
  - `num_games` rollouts
  - each rollout is one game
- `task-evaluate`
  - one rollout per seed
- `generate_counterfactual_data.py`
  - one outer rollout per game
  - plus up to 9 one-step branches per interesting state
- `rl-train-appo`
  - `num_workers * num_envs_per_worker` live RL rollouts in parallel

### "Are these multi-turn rollouts?"

- forward-data generation games: **yes**
- task-harness episodes: **yes**
- counterfactual branches: **no**, one-step only
- APPO env rollouts: **yes**
- SFT training itself: **not rollouts at all**

### "What is the group size?"

There is no single RL group-size concept in the current codebase.

Closest answers:

- task harness candidate action group:
  - usually **6 to 9 actions**
- counterfactual branch group:
  - up to **9 actions**
- rollout concurrency for data generation:
  - `--workers`
- APPO parallel rollouts:
  - `num_workers * num_envs_per_worker`
- SFT global batch:
  - `batch_size * grad_accum * world_size`
- APPO learner batch:
  - `batch_size`, `num_batches_per_epoch`, `num_epochs`

### "Is there any learned policy right now?"

Yes.

There are now two learned model families:

- the forward prediction LM from [train.py](/home/luc/rl-nethack/train.py)
- the APPO low-level policy/value model from the new `rl/` stack

The current task controller in [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)
is still algorithmic.


## 12. Recommended Mental Model For This Repo

If you want the correct mental model, think of the repo as:

- **today**
  - hybrid project with:
    - forward-model SFT,
    - a hand-shaped control harness,
    - and a first real APPO RL backend
- **not yet**
  - full hierarchical skill RL system with learned rewards and scheduler

The most important files for understanding that are:

- [train.py](/home/luc/rl-nethack/train.py)
- [scripts/generate_training_data.py](/home/luc/rl-nethack/scripts/generate_training_data.py)
- [scripts/generate_counterfactual_data.py](/home/luc/rl-nethack/scripts/generate_counterfactual_data.py)
- [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)
- [src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py)
- [src/evaluator.py](/home/luc/rl-nethack/src/evaluator.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)


## 13. If We Wanted To Mature This RL System Next

The current code naturally points to this progression:

1. keep the current task rewards and task evaluation,
2. benchmark APPO against `task_greedy`,
3. improve observations / model architecture,
4. use counterfactual rollouts to label better/worse action outcomes,
5. train a reward or value model per task,
6. move from flat task-conditioned APPO toward full options / scheduler hierarchy

That would be the first point where terms like:

- rollout length
- number of environments
- minibatches
- policy epochs
- value loss
- group size

become central in the usual RL sense.
