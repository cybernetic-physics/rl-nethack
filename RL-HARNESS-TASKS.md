# RL Harness Task Definitions and Evaluation

This document defines how this repo should identify tasks, shape rewards, and
evaluate progress for a NetHack RL or planning harness.

The goal is not to jump directly to "train PPO on raw env reward." NetHack is
too sparse, too long-horizon, and too compositional for that to be the default
plan here.

Instead, this repo should:

1. define a small set of explicit tasks,
2. shape rewards around those tasks,
3. evaluate trajectory outcomes for each task family,
4. train forward / value / planner components against those task signals,
5. only then consider policy RL on top.


## Why This Document Exists

The default NLE reward is score delta. In practice, that is usually too sparse
to drive useful early learning. On this repo's current local check, a random
50-step rollout produced zero reward on every step.

This is consistent with the literature:

- The NLE paper presents NetHack as a benchmark for exploration, planning,
  skill acquisition, and generalization rather than a simple dense-reward game.
  Source: https://arxiv.org/abs/2006.13760
- The NetHack Challenge report explicitly says score and ascension are not well
  aligned, and that many agents learned to camp for score instead of making
  real progress. Source: https://nethackchallenge.com/report.html
- SkillHack argues that predefined skills are a useful inductive bias in
  NetHack-like sparse-reward settings. Source: https://arxiv.org/abs/2207.11584
- HiHack / AutoAscend analysis shows that effective NetHack behavior is already
  naturally decomposed into explicit strategy routines such as exploration,
  combat, safe eating, and item handling. Source:
  https://papers.nips.cc/paper_files/paper/2023/file/764ba7236fb63743014fafbd87dd4f0e-Paper-Conference.pdf

Inference from those sources:

- raw score is not the right sole objective for this repo,
- skill / task decomposition is appropriate,
- evaluation must measure meaningful game progress, not only local prediction
  accuracy or score.


## What We Can Observe in NLE

This repo uses NLE `1.2.0`. The environment exposes enough state to define
useful tasks and evaluations:

- `chars`, `glyphs`, `colors`, `specials`, `screen_descriptions`
- `blstats` including HP, gold, score, depth, XP, hunger, time
- inventory observations: `inv_glyphs`, `inv_strs`, `inv_letters`,
  `inv_oclasses`
- episode info: `end_status`, `is_ascended`

This means we can define task rewards over:

- survival and damage
- exploration and novelty
- dungeon progression
- inventory interaction
- combat state transitions
- repeated / degenerate behavior


## Design Principles

### 1. Do not define "the reward" too early

Keep separate task rewards:

- `r_explore`
- `r_survive`
- `r_combat`
- `r_descend`
- `r_resource`

Only combine them into a weighted scalar later if needed.

Why:

- easier debugging
- easier ablations
- easier targeted scenario generation
- less risk of hiding regressions behind one mixed score


### 2. Evaluate trajectories, not just one-step labels

This repo already measures one-step delta prediction. That is necessary but not
sufficient.

Every serious evaluation should also report trajectory metrics:

- alive at horizon
- cumulative shaped reward
- unique tiles explored
- rooms discovered
- stairs discovered / reached / descended
- gold gained
- net HP delta
- deaths
- repeated-action rate
- repeated-state rate
- invalid or nonsensical action rate


### 3. Prefer task-conditioned evaluation over raw aggregate score

The same agent can be:

- good at exploration,
- bad at combat,
- decent at not dying,
- terrible at making real progress.

We should measure those separately.


### 4. Build tasks that are derivable from env state

If a task cannot be recognized from NLE observations and metadata, it should
not be a first-class benchmark in this repo.

Use only tasks whose reward and success signals can be computed from:

- `blstats`
- visible map and memory
- messages
- inventory
- terminal info


## Task Families

These are the task families this repo should support first.

### 1. Exploration

Definition:

- make safe map progress,
- reveal unseen tiles,
- discover rooms and important landmarks,
- reduce aimless looping.

Suggested shaped reward:

- `+1.0 * newly_seen_tiles`
- `+5.0 * newly_detected_room`
- `+10.0 * first_time_stairs_seen`
- `+2.0 * first_time_useful_item_seen`
- `-0.25 * repeated_state`
- `-0.10 * revisiting_recent_tile`
- `-0.50 * invalid_or_nonsensical_action`

Success metrics:

- unique tiles explored by step horizon
- rooms found
- stairs found
- average revisit ratio

Use cases:

- early-game control
- memory validation
- anti-looping diagnostics


### 2. Survival

Definition:

- remain alive,
- avoid unnecessary damage,
- stabilize when low HP.

Suggested shaped reward:

- `-100.0` on death
- `-1.0 * HP_loss`
- `+0.5 * HP_gain`
- extra penalty for taking damage below a low-HP threshold
- bonus for increasing distance from immediate threats when low HP

Success metrics:

- alive at horizon
- death rate
- min HP over episode
- HP recovered from low-HP states

Use cases:

- debugging reckless policies
- testing escape behavior
- value-model calibration


### 3. Combat

Definition:

- resolve nearby hostile encounters efficiently,
- trade HP for advantage well,
- avoid dying in tactical situations.

Suggested shaped reward:

- `+15.0` for confirmed kill event or enemy disappearance after attack sequence
- `-1.5 * HP_loss`
- `-100.0` on death
- `+5.0` for ending combat state with no adjacent hostiles
- `-0.5` for obvious no-op under threat

Success metrics:

- combat win rate
- combat death rate
- net HP delta in combat episodes
- kills per combat encounter

Use cases:

- one-step counterfactual ranking
- tactical planning
- testing whether the model understands local danger


### 4. Descent / Progression

Definition:

- make real dungeon progress instead of camping,
- reach and use stairs,
- improve strategic state over time.

Suggested shaped reward:

- `+25.0` for first reaching visible stairs
- `+100.0` for descending a floor
- `+0.1 * score_delta` as a weak auxiliary term only
- `+0.2 * gold_delta`
- `+1.0 * XP_gain`
- penalty for long local camping without exploration or progress

Success metrics:

- floors descended
- depth reached
- time to first descent
- proportion of runs with real progression

Use cases:

- challenge-style evaluation
- anti-camping objective
- long-horizon planning


### 5. Resource Interaction

Definition:

- pick up useful items,
- avoid nonsense inventory behavior,
- consume resources only in sensible contexts.

Suggested shaped reward:

- `+2.0` for valid pickup of useful visible item
- `+1.0` for collecting gold
- `+3.0` for sensible emergency consumption
- `-1.0` for invalid pickup/open/kick/eat/drink/drop attempts
- `-2.0` for harmful or obviously context-free inventory actions

Success metrics:

- pickup precision
- gold collected
- useful item collection rate
- nonsensical inventory action rate

Use cases:

- current local policy data cleanup
- inventory-aware planning


## Proposed Scenario Suites

This repo should evaluate not only full random games, but also targeted
scenario suites.

### Suite A: Exploration Scenarios

- empty room with multiple exits
- corridor with branching junction
- frontier with one promising unseen branch
- visible stairs within short distance

Questions:

- does the agent move toward novelty?
- does it avoid useless waiting/search loops?


### Suite B: Survival Scenarios

- low HP with nearby safe corridor
- low HP with nearby monster
- hunger pressure with and without food

Questions:

- does the agent de-risk when fragile?
- does it avoid suicidal behavior?


### Suite C: Combat Scenarios

- adjacent weak enemy
- enemy at short distance in corridor
- multiple nearby threats

Questions:

- does the agent fight, retreat, or stall appropriately?
- does predicted value line up with actual HP trade?


### Suite D: Progression Scenarios

- stairs visible but not adjacent
- stairs adjacent
- room mostly explored, one progress route open

Questions:

- does the agent choose progression over score-camping?


### Suite E: Resource Scenarios

- item underfoot
- nearby gold
- bad context for pickup / open / kick

Questions:

- does the agent distinguish valid resource interaction from noise?


## How To Identify New Tasks

When adding a task, require all of the following:

1. The task corresponds to a recognizable game competency.
2. The task reward can be computed from env observations and repo memory.
3. The task has at least one trajectory-level success metric.
4. The task can be stress-tested in a targeted scenario suite.
5. The task is useful either for:
   - early-game control,
   - planner/value learning,
   - curriculum shaping,
   - or final-game progress.

If a proposed task fails one of those checks, do not add it yet.


## Recommended Evaluation Stack

This repo should use three evaluation layers.

### Layer 1: Open-loop Prediction

Current signals:

- exact-match delta accuracy
- per-field accuracy

Keep these, but do not treat them as the main success criterion.


### Layer 2: Closed-loop Task Evaluation

For each task family, run fixed-horizon episodes and report:

- mean shaped return
- success rate
- failure modes
- trajectory metrics relevant to the task

This should become the main model-selection layer for planner / value work.


### Layer 3: Golden Closed-loop Replay

Use the existing golden harness to verify:

- prompt formatting
- parser consistency
- one-step alignment
- no silent train/eval mismatch

Do not trust any RL or planning result if this layer is broken.


## How This Repo Should Train From These Tasks

### Stage 1: Supervised transition learning

Train the forward model to predict state deltas.

Goal:

- reliable one-step transitions
- especially on targeted scenario data


### Stage 2: Supervised reward / value learning

Label transitions or short rollouts with task rewards and returns.

Train:

- task-specific reward predictors, or
- task-conditioned value heads

Goal:

- estimate which action sequences improve task outcomes


### Stage 3: Planner over forward model

At decision time:

1. enumerate candidate actions,
2. roll each forward via the learned transition model,
3. score by task-conditioned reward / value,
4. choose highest predicted return.

This is the most natural next step for this repo.


### Stage 4: Policy RL, if still needed

Only after the above is working should we do online policy optimization.

If RL is added:

- initialize from SFT / BC
- optimize shaped task rewards, not raw score alone
- keep targeted scenario evals as hard gates


## Immediate Repo Changes Recommended

1. Add `src/task_rewards.py`
   - compute per-step shaped rewards for:
     - exploration
     - survival
     - combat
     - descent
     - resource interaction

2. Add `src/trajectory_eval.py`
   - compute per-episode metrics and success/failure summaries

3. Add targeted scenario generators
   - likely based on existing counterfactual / seed-scanning utilities

4. Extend manifest/evaluation output
   - include trajectory metrics, not only field accuracy

5. Keep SFT scaling ahead of full RL
   - first train better transition / reward models
   - then layer planning or RL on top


## What Not To Do

- Do not use raw NLE score as the only objective.
- Do not rank agents only by local exact-match prediction accuracy.
- Do not start PPO without targeted task evals and a stable golden harness.
- Do not collapse all behavior into one scalar before task-specific debugging.


## References

- The NetHack Learning Environment
  https://arxiv.org/abs/2006.13760
- Hierarchical Kickstarting for Skill Transfer in Reinforcement Learning
  https://arxiv.org/abs/2207.11584
- NetHack Challenge report
  https://nethackchallenge.com/report.html
- NetHack is Hard to Hack
  https://papers.nips.cc/paper_files/paper/2023/file/764ba7236fb63743014fafbd87dd4f0e-Paper-Conference.pdf
