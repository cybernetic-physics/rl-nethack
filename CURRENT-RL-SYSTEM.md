# Current RL System in This Repo

This document explains what the repo currently does, grounded in the code as it
exists today.

The short version:

- this repo does **not** currently have a learned PPO / GRPO / APPO-style RL
  training loop,
- it **does** have:
  - supervised fine-tuning for a forward model,
  - gameplay rollout code for collecting training data,
  - a task-directed closed-loop control and evaluation harness,
  - a counterfactual rollout generator for "what if I took action X?" data.

If you came in expecting a standard RL codebase with rollout workers,
advantages, value heads, clipped objectives, and policy updates: that is **not**
 what is implemented here yet.


## 1. What Is Actually Trained Today

The only actual model training path in the repo right now is in
[train.py](/home/luc/rl-nethack/train.py).

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


## 2. What "Rollout" Means In This Repo

The word "rollout" appears in a few places, but it does **not** mean the same
thing as "policy rollouts used by an RL learner" in a PPO codebase.

In this repo today, a "rollout" usually means:

- run a NetHack episode for some number of steps,
- collect transitions,
- maybe query an LLM for the next action,
- write training/eval examples,
- or score task behavior.

There are three main rollout systems.


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


## 6. Task Rewards: What The Harness Optimizes

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
- but the reward is currently **hand-shaped**, not learned from feedback.


## 7. Evaluation: What We Measure Today

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


## 8. What "Batch Size" Means In This Repo

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


## 9. What Is Missing If You Expect "Real RL"

The repo currently does **not** have:

- PPO
- GRPO
- APPO
- actor-critic training
- replay buffer / rollout buffer
- advantage estimation
- value function training
- reward model training
- option / skill policy training
- learned high-level policy over skills

What it does have is:

- episode rollouts for data collection
- one-step counterfactual branching
- hand-shaped task reward functions
- trajectory evaluation

So the current stack is best described as:

1. collect trajectories,
2. train a forward model with SFT,
3. evaluate closed-loop task behavior with a hand-shaped controller,
4. prepare for future RL / planning work.


## 10. Practical Answers To Your Specific Questions

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

### "Are these multi-turn rollouts?"

- forward-data generation games: **yes**
- task-harness episodes: **yes**
- counterfactual branches: **no**, one-step only
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
- SFT global batch:
  - `batch_size * grad_accum * world_size`

### "Is there any learned policy right now?"

Not in the main path.

The main trained model is a **forward prediction LM**.
The current task controller is **algorithmic**, not learned.


## 11. Recommended Mental Model For This Repo

If you want the correct mental model, think of the repo as:

- **today**
  - forward-model training project with control/eval scaffolding
- **not yet**
  - full RL agent training system

The most important files for understanding that are:

- [train.py](/home/luc/rl-nethack/train.py)
- [scripts/generate_training_data.py](/home/luc/rl-nethack/scripts/generate_training_data.py)
- [scripts/generate_counterfactual_data.py](/home/luc/rl-nethack/scripts/generate_counterfactual_data.py)
- [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)
- [src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py)
- [src/evaluator.py](/home/luc/rl-nethack/src/evaluator.py)


## 12. If We Wanted To Make This A Real RL System Next

The current code naturally points to this progression:

1. keep the current task rewards and task evaluation,
2. use counterfactual rollouts to label better/worse action outcomes,
3. train a reward or value model per task,
4. replace one-step brute-force action selection with a learned scorer,
5. then add real policy optimization if needed.

That would be the first point where terms like:

- rollout length
- number of environments
- minibatches
- policy epochs
- value loss
- group size

become central in the usual RL sense.

