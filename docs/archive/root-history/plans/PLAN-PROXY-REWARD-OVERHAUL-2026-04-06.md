# Proxy Reward Overhaul Plan 2026-04-06

## Purpose

This document defines the next concrete repo shift:

- stop centering the RL stack around the current hand-shaped scalar reward
- replace that center of gravity with teacher-derived, short-horizon proxy targets
- keep the current hand-shaped reward as a baseline and fallback, not the main research bet

This plan is grounded in:

- the current code in [src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py), [rl/rewards.py](/home/luc/rl-nethack/rl/rewards.py), [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py), and [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
- the current project diagnosis in [wagmi.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/wagmi.md)
- the current repo shape plan in [REPO-MORPH-PLAN-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/REPO-MORPH-PLAN-2026-04-06.md)
- the NetHack and NetHack-RL literature, especially:
  - [references/papers/nle_neurips2020.pdf](/home/luc/rl-nethack/references/papers/nle_neurips2020.pdf)
  - [references/papers/nethack_offline_pretrain_arxiv2023.pdf](/home/luc/rl-nethack/references/papers/nethack_offline_pretrain_arxiv2023.pdf)
  - [references/papers/skillhack_arxiv2022.pdf](/home/luc/rl-nethack/references/papers/skillhack_arxiv2022.pdf)
  - [references/papers/motif_iclr2024.pdf](/home/luc/rl-nethack/references/papers/motif_iclr2024.pdf)
  - [references/papers/maestromotif_iclr2025.pdf](/home/luc/rl-nethack/references/papers/maestromotif_iclr2025.pdf)

## Research Summary

### What the current reward is actually doing

The current online reward for the `explore` skill is mostly:

- new tiles
- new rooms
- stairs seen
- useful items seen

with penalties for:

- repeated state
- revisited tile
- repeated action
- some invalid-like actions
- death

That makes it a good debug reward. It does not make it a good NetHack competence objective.

### What real NetHack play values

The literature and game documentation point to a different structure:

- survive first
- progress safely
- use memory and domain knowledge
- search only in contextually justified places
- descend with the right timing
- avoid farming shallow local novelty when it hurts long-term progress

Important implications:

- local novelty is not enough
- score is not enough
- generic exploration bonuses are not enough
- `search` is only correct in narrow structural contexts
- short-horizon teacher competence is more learnable than full-game sparse return

### Why this matters for the repo

The repo is currently optimizing:

- hand-shaped intrinsic skill reward
- plus PPO value targets

while we actually trust:

- deterministic held-out teacher trace match

That mismatch is now the main bottleneck.

## Core Thesis

The next proxy should not be:

- another hand-tuned scalar
- another generic novelty bonus
- another purely local one-step reward

The next proxy should be:

- **teacher-derived**
- **short-horizon**
- **decomposed into interpretable heads**
- **trained from the existing trace data first**

## What We Should Build

## Phase 1: Teacher-Derived Proxy Dataset

### Goal

Turn the existing trace rows into a short-horizon proxy-learning dataset.

### Why the current traces are enough

Current trace rows already contain:

- feature vectors
- chosen action
- allowed actions
- planner traces
- obs hashes
- reward
- done flag
- memory counts
- episode id and step

This is enough for a first proxy layer without a new rollout stack.

### New targets to add

For each row, compute labels over a future window `K` where `K` starts at `8` and later expands to `16`.

Add:

1. `k_step_progress`
   - weighted future gain in:
     - explored tiles
     - rooms
     - stairs reach / stairs stand
     - descent

2. `k_step_survival`
   - penalty for:
     - death
     - severe HP loss
     - dangerous adjacent-hostile increase

3. `k_step_loop_risk`
   - future repeated-state and repeated-position pressure

4. `k_step_resource_value`
   - useful-item exposure / pickup / gold gain when relevant

5. `search_context_label`
   - whether the state looks structurally appropriate for `search`
   - start with deterministic heuristics, not human labeling

6. `teacher_margin`
   - margin between chosen teacher action and rejected alternatives from `planner_trace`

### Code changes

Add:

- [rl/proxy_dataset.py](/home/luc/rl-nethack/rl/proxy_dataset.py)
  - build `K`-step proxy rows from trace JSONL
  - compute future aggregates and labels

- [rl/proxy_labels.py](/home/luc/rl-nethack/rl/proxy_labels.py)
  - label helpers:
    - progress
    - survival
    - loop risk
    - search context

Modify:

- [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
  - optionally emit a little more state metadata if needed:
    - standing-on-stairs
    - adjacent tile summary
    - local geometry marker if not already inferable

### Debug loop

Before training any model:

1. build a tiny proxy dataset shard
2. dump label histograms
3. print top positive and top negative examples for each head
4. manually inspect `search_context_label` examples
5. reject the label design if it mostly marks nonsense states

### Promotion gate

Do not move to Phase 2 until:

- label distributions look sane
- `search_context_label` has clear positive and negative examples
- `k_step_progress` is not trivially identical to current scalar reward

## Phase 2: Proxy Model

### Goal

Train a multi-head proxy model from trace features.

### Model shape

Add:

- [rl/proxy_model.py](/home/luc/rl-nethack/rl/proxy_model.py)

Structure:

- shared MLP trunk over current features
- heads for:
  - progress regression
  - survival regression or classification
  - loop-risk classification
  - search-context classification
  - teacher-action margin or pairwise preference

Optional later:

- let the model consume world-model latents as an additional branch

### Code changes

Add:

- [rl/train_proxy_model.py](/home/luc/rl-nethack/rl/train_proxy_model.py)
- [rl/proxy_eval.py](/home/luc/rl-nethack/rl/proxy_eval.py)
- [rl/proxy_report.py](/home/luc/rl-nethack/rl/proxy_report.py)

Modify:

- [cli.py](/home/luc/rl-nethack/cli.py)
  - add:
    - `rl-build-proxy-dataset`
    - `rl-train-proxy`
    - `rl-evaluate-proxy`

### Debug loop

1. train on tiny shard
2. verify loss decreases and heads are non-degenerate
3. run held-out eval
4. inspect:
   - calibration of survival head
   - top states predicted as good search states
   - top states predicted as high loop risk
5. compare proxy outputs against `planner_trace` ordering where available

### Promotion gate

Do not wire into the env until:

- held-out proxy metrics beat trivial baselines
- search-context head is visibly better than naive always/no-search baselines
- progress head correlates with future teacher progress better than current scalar reward does

## Phase 3: Reward-System Refactor

### Goal

Make proxy rewards first-class and decomposed.

### Code changes

Refactor:

- [rl/rewards.py](/home/luc/rl-nethack/rl/rewards.py)

into three layers:

1. `HandShapedSkillReward`
   - current baseline

2. `TeacherProxySkillReward`
   - learned multi-head proxy
   - returns:
     - total
     - per-head components

3. `MixedSkillReward`
   - configurable blend:
     - hand-shaped
     - learned proxy
     - env reward

Add:

- [rl/proxy_reward.py](/home/luc/rl-nethack/rl/proxy_reward.py)
  - inference wrapper and mixing helpers

Modify:

- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)
  - include proxy components in `info["debug"]`
  - make reward source selection explicit and inspectable

### Debug loop

1. smoke test env reset/step with proxy reward enabled
2. verify reward components are finite
3. run `N=20` deterministic teacher states through env reward code
4. compare:
   - hand-shaped total
   - proxy total
   - teacher action chosen
5. check whether proxy changes make `search` states look special rather than uniformly bad

### Promotion gate

Do not start RL with proxy reward until:

- env debug output is stable
- proxy reward differs meaningfully from hand-shaped reward
- reward decomposition on curated states looks intelligible

## Phase 4: Offline Policy Validation Before RL

### Goal

Use the proxy offline before trusting it online.

### Code changes

Add:

- [rl/proxy_rerank.py](/home/luc/rl-nethack/rl/proxy_rerank.py)
  - score candidate actions using proxy heads

Modify:

- [rl/trace_eval.py](/home/luc/rl-nethack/rl/trace_eval.py)
  - optional proxy-based diagnostics

- [rl/teacher_report.py](/home/luc/rl-nethack/rl/teacher_report.py)
  - include proxy agreement summaries when available

### Validation tasks

1. action reranking on held-out trace rows
2. search-specific shard evaluation
3. stairs/progression shard evaluation
4. weak-action slices:
   - `south`
   - `west`
   - `search`

### Promotion gate

Do not promote proxy reward to RL until at least one of these is true:

- proxy reranking improves held-out teacher agreement
- proxy improves `search` shard materially
- proxy improves progression/stairs shard materially

If none of these happen, the proxy is not ready.

## Phase 5: Short RL Integration

### Goal

Use the proxy conservatively inside the online learner.

### First integration mode

Do **not** replace the full reward immediately.

Start with:

- hand-shaped reward as base
- learned proxy as auxiliary reward term
- proxy heads also available as critic features later

### Code changes

Modify:

- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
  - add:
    - `proxy_reward_path`
    - `proxy_reward_weight`
    - `proxy_reward_mode`

- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [cli.py](/home/luc/rl-nethack/cli.py)
  - plumb proxy args

- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)
  - use mixed reward source

### Debug loop

Run in this order:

1. `128`-step smoke
2. `1024`-step short run
3. `5k`-step medium run

At each stage:

- rank checkpoints by held-out trace match
- dump reward component summaries
- compare action distributions to teacher
- inspect whether `search` usage changes in the right direction

### Hard stop condition

If proxy-weighted RL:

- improves live reward
- but degrades held-out trace match

then stop and go back to the proxy itself.
Do not scale a misleading proxy.

## Phase 6: World Model Coupling

### Goal

Use the world model where it is already helping: representation.

### What to do

Keep:

- [rl/world_model.py](/home/luc/rl-nethack/rl/world_model.py)
- [rl/world_model_features.py](/home/luc/rl-nethack/rl/world_model_features.py)

but use the world model as:

- an input branch to the proxy model
- an auxiliary target source
- a possible source of `loop_risk` / `frontier_gain` features

### What not to do yet

Do not:

- switch to full imagined-rollout control
- replace teacher-derived proxy with free-running model rollouts

## Concrete Debugging Discipline

Every implementation phase should use the same loop:

1. tiny dataset / smoke test
2. unit tests in [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)
3. explicit artifact inspection
4. held-out deterministic comparison
5. only then longer training

## New tests to add

Add regression tests for:

- proxy dataset label generation
- search-context labeling
- masked action handling in proxy scoring
- proxy reward decomposition
- mixed reward env stepping
- CLI plumbing for proxy commands

## Immediate Next Implementation Order

1. Add `rl/proxy_labels.py`
2. Add `rl/proxy_dataset.py`
3. Add `rl/proxy_model.py`
4. Add `rl/train_proxy_model.py`
5. Add `rl/proxy_eval.py`
6. Add CLI commands
7. Add proxy reward source in `rl/rewards.py`
8. Add env integration in `rl/sf_env.py`
9. Run offline validation
10. Only then run short RL

## First Concrete Experiment

The first experiment should be deliberately small:

1. build an `explore` proxy dataset from `/tmp/x100_v4_train_wm_aux.jsonl`
2. train a proxy with heads:
   - progress
   - survival
   - loop risk
   - search context
3. evaluate on `/tmp/x100_v4_heldout_wm_aux.jsonl`
4. inspect top `search_context_label` states manually
5. run action reranking on the held-out trace set

If that does not improve the offline picture, do not wire it into RL yet.

## Why This Plan Is Better Than More Hand-Tuning

This plan is better because it:

- uses the teacher and traces we already trust
- respects the actual structure of NetHack
- keeps the world model in the support role where it already helps
- gives us multiple offline gates before expensive RL
- makes the reward path more inspectable instead of less

## Success Criteria

This overhaul is working if:

1. proxy heads are interpretable on held-out traces
2. `search` is no longer treated as uniformly bad or uniformly absent
3. offline reranking improves teacher agreement on at least one hard slice
4. short RL runs no longer rely entirely on the old hand-shaped scalar
5. the next large run is promoted because the offline and short-run evidence improved, not because we need another guess

## References

- NLE paper: https://arxiv.org/abs/2006.13760
- NetHack Challenge report: https://nethackchallenge.com/report.html
- NetHack Guidebook: https://www.nethack.org/docs/nh360/nethack-360-Guidebook.pdf
- NetHack wiki standard strategy: https://nethackwiki.com/wiki/Standard_strategy
- NetHack wiki search: https://nethackwiki.com/wiki/Search
- NetHack wiki ascension: https://nethackwiki.com/wiki/Ascension
- AutoAscend: https://github.com/maciej-sypetkowski/autoascend
