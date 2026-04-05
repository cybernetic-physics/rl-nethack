# APPO + Options Overhaul Plan

This document explains how this repo should be overhauled from its current
state into a real multi-step RL system for NetHack.

It is grounded in:

- the current code in this repo,
- NetHack-specific RL systems used by others,
- the hierarchical / AI-feedback lessons from MaestroMotif and Motif.


## 1. Current State

The repo currently has three useful but separate pieces:

1. supervised forward-model training in [train.py](/home/luc/rl-nethack/train.py)
2. multi-turn rollout/data generation in
   [scripts/generate_training_data.py](/home/luc/rl-nethack/scripts/generate_training_data.py)
3. task-shaped closed-loop control/evaluation in
   [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)

That means the repo can:

- collect episodes,
- score behavior,
- train predictive models,
- run one-step counterfactual control.

But it still does **not** have a real learned long-horizon RL stack.


## 2. Why APPO, Not GRPO, As The Core Learner

The repos that actually work for long-horizon NetHack use:

- Sample Factory APPO
- PPO-family asynchronous on-policy RL
- skill transfer / hierarchical options on top

This is true in:

- `ngoodger/nle-language-wrapper`
- `facebookresearch/motif`
- `mklissa/maestromotif`
- `ucl-dark/skillhack`

GRPO is attractive for language-model post-training, but it is not the best
core low-level learner for this problem.

Why:

- NetHack has long credit assignment horizons
- the state is partially observed and memory-conditioned
- throughput matters a lot
- the action space is environment-native and discrete
- prior successful systems use APPO, not GRPO, for low-level control

Recommendation:

- use **APPO as the main low-level RL learner**
- use **options / skills** for temporal abstraction
- reserve **GRPO** for later high-level skill selection or textual planning if
  we still want it


## 3. Target Architecture

The target system should have three layers.

### Layer A: low-level skill policy

Input:

- current observation
- memory summary
- active skill id

Output:

- primitive action distribution
- value estimate

Training:

- APPO
- one shared backbone, skill-conditioned

### Layer B: option / skill layer

Each skill should define:

- initiation condition
- termination condition
- action constraints or preferred subset
- reward source
- evaluation metrics

### Layer C: high-level skill scheduler

Input:

- current structured state
- memory
- active skill
- task specification

Output:

- next skill

Start with:

- hand-coded / rules

Later:

- learned scheduler
- or LLM/code policy
- or GRPO over high-level skill choices


## 4. Skill Set For This Repo

Start from the repo’s existing tasks:

- `explore`
- `survive`
- `combat`
- `descend`
- `resource`

Promote them into actual options.

Immediate additions:

- add `ascend`

Later splits:

- split `resource` into:
  - `pickup`
  - `consume`
  - `merchant`
  - `worship`


## 5. New RL Package Layout

The new code should live in a dedicated top-level package:

- [rl/](/home/luc/rl-nethack/rl)

Recommended modules:

- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
  - dataclass configs for environment, training, model, options
- [rl/options.py](/home/luc/rl-nethack/rl/options.py)
  - option specs and registry
- [rl/scheduler.py](/home/luc/rl-nethack/rl/scheduler.py)
  - high-level skill selection interface
- [rl/env_adapter.py](/home/luc/rl-nethack/rl/env_adapter.py)
  - wraps NLE + memory + current skill state
- [rl/rewards.py](/home/luc/rl-nethack/rl/rewards.py)
  - adapters from repo task rewards to skill rewards
- [rl/policy.py](/home/luc/rl-nethack/rl/policy.py)
  - policy interface and baseline implementations
- [rl/model.py](/home/luc/rl-nethack/rl/model.py)
  - model interface and skill-conditioning spec
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
  - APPO training bootstrap / integration
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
  - runnable training entrypoint


## 6. Design Rules For Easy Refactoring

The code should be easy to change quickly.

That means:

### Rule 1: separate interfaces from implementations

We should be able to change:

- the policy network,
- the scheduler,
- the reward source,
- the APPO backend,

without rewriting the environment or options.

### Rule 2: keep option logic declarative

The option definition should be simple data + pure functions.

Avoid burying skill logic deep inside trainers.

### Rule 3: the environment adapter should own episode state

One place should track:

- memory
- active skill
- steps-in-skill
- loop/revisit counters
- episode metrics

### Rule 4: reward computation should be swappable

Start with current hand-shaped rewards from
[src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py),
but make it easy to replace them with:

- learned reward models,
- preference models,
- mixed intrinsic/extrinsic reward

### Rule 5: training backend should be thin

The APPO runner should consume:

- env adapter
- policy/model factory
- config

It should not contain skill definitions inline.


## 7. How Others Structure This

### Sample Factory / APPO systems

NetHack systems based on Sample Factory use:

- many rollout workers
- many envs per worker
- asynchronous learning
- recurrent or transformer policy
- huge environment step budgets

This is the right backbone for long tasks.

### Motif

Motif shows a good separation:

1. annotate trajectory pairs with LLM feedback
2. train reward model
3. train RL agent with APPO using that reward

That means our reward-model path should be separate from the RL trainer.

### MaestroMotif

MaestroMotif adds:

- initiation / termination functions
- training-time policy over skills
- skill-conditioned policy

That should directly shape how we design the `options` and `scheduler`
modules.

### SkillHack

SkillHack shows that:

- long sparse tasks become tractable when skills are learned and transferred

That is exactly why the option layer should be first-class in this repo.


## 8. Phase Plan

### Phase 1: scaffold and baseline

Goal:

- get a clean APPO + options skeleton into the repo
- no clever reward learning yet

Work:

1. add `rl/` package
2. add option registry
3. add env adapter
4. add scheduler interface
5. add trainer bootstrap
6. wire current hand-shaped rewards into the RL env

Deliverable:

- a dry-run / config-printable APPO training entrypoint

### Phase 2: learned low-level policies

Goal:

- replace the current one-step brute-force controller with a learned policy

Work:

1. add skill-conditioned policy network
2. integrate recurrent memory or compact memory summary
3. train primitive-action skill policy with APPO
4. benchmark against `task_greedy`

Deliverable:

- learned policy beats `wall_avoidance`
- learned policy approaches or beats one-step `task_greedy` on some tasks

### Phase 3: learned reward models

Goal:

- stop hand-tuning skill rewards forever

Work:

1. build preference dataset generator
2. train Bradley-Terry reward model per skill
3. add reward-model inference path to RL env
4. keep hand-shaped rewards as fallback / baseline

Deliverable:

- reward-model-driven training for `explore` and `survive`

### Phase 4: high-level scheduling

Goal:

- compose skills over long tasks

Work:

1. rule-based scheduler
2. evaluate multi-skill tasks
3. try learned or LLM-generated scheduler
4. optionally use GRPO at this layer

Deliverable:

- multi-skill task execution, not just one skill per episode


## 9. What We Should Not Do

### 1. Do not replace everything with one GRPO loop

That would overfit the project to an LM-training abstraction instead of the
actual NetHack control problem.

### 2. Do not remove the current task harness

Keep [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py) as:

- oracle baseline
- regression testbed
- reward-debugging tool

### 3. Do not hardcode skills inside the trainer

The trainer should be generic.

### 4. Do not couple policy code to one reward definition

Reward models will change.


## 10. Immediate Next Steps

1. Add the new [rl/](/home/luc/rl-nethack/rl) package.
2. Add option definitions for the current task set.
3. Add a rule-based scheduler.
4. Add an env adapter that exposes active skill state.
5. Add an APPO trainer entrypoint stub that can be swapped to Sample Factory.
6. Validate the interfaces with dry-run config and smoke tests.


## 11. Current Backend Status

This repo now contains a real Sample Factory APPO backend integration in:

- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)

The backend has been smoke-tested with a tiny serial run that:

- registered the custom NetHack skill env,
- created the actor-critic,
- collected rollouts,
- trained,
- and wrote a checkpoint.

Example command:

```bash
uv run --no-sync python cli.py rl-train-appo \
  --serial-mode \
  --num-workers 1 \
  --num-envs-per-worker 1 \
  --rollout-length 8 \
  --recurrence 8 \
  --batch-size 8 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 32 \
  --experiment appo_smoke2
```

Why `--no-sync` is currently needed:

- upstream `sample-factory` depends on `gymnasium<1.0`
- current `nle==1.2.0` depends on `gymnasium==1.0.0`
- the runtime stack works in the project `.venv`, but the dependency metadata
  is inconsistent

So the code backend is real, but packaging is currently constrained by upstream
dependency metadata rather than the repo architecture.


## 12. Bottom Line

The right overhaul is:

- APPO for low-level RL
- options for temporal abstraction
- shared skill-conditioned policy
- hand-shaped rewards first
- learned reward models next
- scheduler on top
- GRPO only later, if used, at the high-level skill-selection layer

That is the most defensible path from the codebase we have now to a real
long-horizon RL system.
