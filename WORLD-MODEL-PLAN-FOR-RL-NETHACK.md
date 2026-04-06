# World Model Plan for RL NetHack

## Purpose

This document answers a practical question:

How should we use world models to improve the current RL stack in this repository?

It is intentionally narrower than the earlier research notes. The goal here is not to survey world models in general. The goal is to decide what to build next in this codebase.

This plan is grounded in:

- the current repo state
- the current failure mode of the RL harness
- the longer-horizon NetHack requirements we already identified
- the relevant literature reviewed in:
  - [RESEARCH-NOTES-WORLD-MODELS-NETHACK.md](/home/luc/rl-nethack/RESEARCH-NOTES-WORLD-MODELS-NETHACK.md)
  - [RESEARCH-NOTES-NETHACK-HORIZONS.md](/home/luc/rl-nethack/RESEARCH-NOTES-NETHACK-HORIZONS.md)

## Current Repo State

The current strongest pipeline is:

1. offline teacher / behavior-regularized policy on `v4` features
2. teacher-constrained APPO online improvement
3. deterministic held-out trace match as the trusted metric

This is already much better than the earlier broken RL path.

But the main failure mode remains:

- the policy can tie the teacher
- then online RL drifts away from it
- reward improves while the trusted trace metric does not

This tells us something important:

- our next world-model work should **not** try to replace the whole control stack
- it should improve:
  - representation
  - short-horizon planning
  - alignment of policy updates

## What We Should Not Build First

Do **not** start with:

- a full Dreamer-style end-to-end replacement of APPO
- a pure one-step reconstruction model trained only to predict raw next observations
- a long free-running imagination loop over weak latents

Why:

- NetHack is too partially observable and too long-horizon
- our current issue is not “we have no model”
- our current issue is “the online learner drifts after correct initialization”

So the first world-model contribution should be conservative and local.

## What We Should Build First

We should build a **skill-conditioned latent world model** with a **short-horizon objective**.

The model should take:

- current observation / map context
- memory summary or short history
- current skill id
- optionally current action or short action prefix

And predict:

1. next latent state after `K` steps
2. cumulative skill reward / teacher-relevant return proxy over `K` steps
3. loop / revisit risk
4. frontier gain / exploration gain
5. skill termination flag

This is the right compromise between:

- the generic world-model literature
- the skill-planning literature
- and the current structure of our repo

## Why This Is The Right First World Model

### 1. It fits the current RL problem

The current learner already has a decent teacher and decent local policy features.

The gap is:

- preserving teacher-aligned behavior
- improving beyond it without drifting

A latent predictive model helps by making the policy representation carry more future-relevant structure.

### 2. It matches NetHack better than one-step prediction

NetHack is long-horizon and partially observable.

One-step next-observation prediction is too myopic.

A `K`-step or skill-level target is more aligned with:

- exploration
- descent
- search
- combat setup

### 3. It matches our codebase

This repo already has:

- skills
- teacher traces
- BC / behavior-reg
- teacher-constrained APPO
- deterministic trace evaluation

So we do not need a brand-new algorithmic stack.

We need a new shared latent module that can plug into the existing stack.

## Proposed Architecture

## Stage 1: Latent Encoder

Add a learned encoder that maps the current structured state into a latent vector `z_t`.

Inputs should include:

- current local observation
- directional / frontier features
- recent action history
- repeated-state counters
- skill id
- short memory summary

This encoder should become a replacement candidate for the hand-built `v4` feature path.

### Output

- latent state `z_t`

### Immediate use

- BC / behavior-reg feature extractor
- APPO policy input

## Stage 2: Short-Horizon Predictive Heads

On top of the encoder, add heads that predict:

1. `z_{t+K}` or a short-horizon latent summary
2. cumulative shaped reward over the next `K` steps
3. frontier / exploration gain over the next `K` steps
4. repeated-state / loop risk over the next `K` steps
5. skill termination

For the first iteration, start with:

- `K = 8` or `K = 16`

That is long enough to matter and short enough to learn reliably from traces.

## Stage 3: Policy Integration

Use the world model in three ways, in this order:

### A. Encoder pretraining for BC / behavior-reg

Train the encoder + predictive heads offline.

Then:

- freeze or partially freeze the encoder
- retrain BC / behavior-reg
- evaluate on held-out trace match

This is the cheapest first experiment.

### B. Auxiliary loss during APPO

Add a predictive auxiliary loss during RL:

- predict short-horizon latent target
- predict exploration gain / loop risk

The purpose is not to control directly.

The purpose is to make the policy representation remain predictive and structured, reducing drift.

### C. Short-horizon action / option ranking

For a small candidate action set:

- encode current state
- imagine a short horizon
- score candidates by predicted teacher-relevant quantities

This can later replace or augment the current weak scalar proxy used in some teacher / task-harness logic.

## Data Sources

The world model should be trained in stages:

### Stage A: current repo traces

Use:

- teacher traces
- behavior-reg teacher traces
- DAgger traces when useful

This keeps the first prototype small and debuggable.

### Stage B: Dungeons and Data / AutoAscend

Once the pipeline works locally:

- use Dungeons and Data for large-scale latent pretraining
- especially for long-horizon representation learning

This is where world models become especially attractive for this repo.

The Dungeons and Data paper is one of the strongest reasons to do this:

- huge scale
- competent AutoAscend behavior
- long-horizon trajectories
- much better coverage than our small local traces

## How This Changes The Repo

## New files to add

- [rl/world_model.py](/home/luc/rl-nethack/rl/world_model.py)
  - encoder and predictive heads

- [rl/train_world_model.py](/home/luc/rl-nethack/rl/train_world_model.py)
  - offline trainer on trace data

- [rl/world_model_dataset.py](/home/luc/rl-nethack/rl/world_model_dataset.py)
  - converts trace rows into `K`-step training tuples

- [rl/world_model_eval.py](/home/luc/rl-nethack/rl/world_model_eval.py)
  - evaluates predictive accuracy and latent usefulness

- [rl/world_model_features.py](/home/luc/rl-nethack/rl/world_model_features.py)
  - optional feature-construction helper if we want to keep the current encoder and world model cleanly separated

## Existing files to modify

- [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)
  - add a learned latent-backed observation mode, likely `v5`

- [rl/train_behavior_reg.py](/home/luc/rl-nethack/rl/train_behavior_reg.py)
  - allow loading a pretrained encoder / latent backbone

- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
  - add flags for:
    - world-model checkpoint
    - freeze / finetune encoder
    - auxiliary predictive loss toggles

- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
  - wire auxiliary loss and latent encoder into the APPO path

- [cli.py](/home/luc/rl-nethack/cli.py)
  - add:
    - `rl-train-world-model`
    - `rl-evaluate-world-model`
    - optional `--world-model-path`
    - optional `--world-model-aux-loss`

- [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
  - add support for `K`-step targets and short-horizon labels

## Fast Debug Loop for World Models

We should not debug this with long RL runs first.

Use the same fast-loop discipline we already learned:

### Step 1

Train world model on a tiny fixed trace set.

Check:

- does it predict `K`-step targets at all?
- do losses go down?

### Step 2

Plug encoder into BC only.

Check:

- does held-out trace match improve over `v4`?

This is the first real gate.

### Step 3

Plug encoder into behavior-reg.

Check:

- does weak-action shard performance improve further?

### Step 4

Only then add APPO auxiliary losses.

Check:

- does the first learned checkpoint hold teacher alignment better than current teacher-reg APPO?

### Step 5

Only then try short-horizon imagined ranking.

## Success Criteria

The first world-model milestone is **not**:

- solve NetHack
- beat AutoAscend
- replace APPO

The first milestone is:

- learned latent encoder improves held-out trace match over hand-built `v4` features

The second milestone is:

- auxiliary world-model loss reduces early online drift relative to the current APPO baseline

The third milestone is:

- short-horizon imagined ranking improves teacher-aligned action choice on held-out traces or focused shards

## Ordered Plan

1. Build `K`-step world-model dataset from existing teacher traces.
2. Implement latent encoder + predictive heads.
3. Train offline on current local traces.
4. Evaluate whether the encoder helps BC on held-out traces.
5. Scale the same world-model training to Dungeons and Data / AutoAscend.
6. Re-run BC / behavior-reg with the pretrained latent encoder.
7. Add world-model auxiliary losses to APPO.
8. Compare early teacher-reg APPO drift with and without the world-model auxiliary loss.
9. Add short-horizon imagined action / option ranking.
10. Only after that consider more ambitious model-based control.

## Bottom Line

World models can help this repo, but the best path is:

- **world model for representation and short-horizon skill prediction**
- then **world model as auxiliary loss**
- then **world model for short-horizon ranking**

not:

- full Dreamer replacement
- raw one-step reconstruction
- long imagination before we have a trusted latent state

For this project, the right first world model is a **skill-conditioned short-horizon latent predictive model trained from teacher and AutoAscend traces**.
