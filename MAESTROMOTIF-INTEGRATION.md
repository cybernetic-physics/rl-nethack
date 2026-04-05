# Integrating MaestroMotif-Style Methods Into This Repo

This document explains how to adapt the main ideas from the ICLR 2025
MaestroMotif paper to the current `rl-nethack` codebase.

It is intentionally grounded in the code as it exists now, not in an imagined
future architecture.

Primary source:

- MaestroMotif ICLR 2025:
  https://proceedings.iclr.cc/paper_files/paper/2025/file/2dc5a0faac8102fd47363795f71126ee-Paper-Conference.pdf

Supporting sources:

- Motif ICLR 2024:
  https://openreview.net/pdf/7fafd8a9cedfff67f51f1129cf8c1a76f436d271.pdf
- SkillHack:
  https://arxiv.org/abs/2207.11584
- Dungeons and Data / NetHack Learning Dataset:
  https://arxiv.org/abs/2211.00539


## 1. The Main Point

MaestroMotif’s core idea is directly relevant here:

- do not use an LLM as the low-level NetHack controller,
- use language to define a small set of reusable skills,
- use AI feedback to learn reward models for those skills offline,
- train low-level skill controllers with RL,
- let a higher-level planner or code policy operate over skills.

That is a much better fit for this repo than trying to force everything into:

- one flat reward,
- one primitive-action policy,
- one monolithic RL loop.


## 2. What The Repo Has Right Now

Current relevant pieces:

- [train.py](/home/luc/rl-nethack/train.py)
  - supervised fine-tuning for a forward model
- [scripts/generate_training_data.py](/home/luc/rl-nethack/scripts/generate_training_data.py)
  - multi-turn gameplay rollouts for forward-model training data
- [scripts/generate_counterfactual_data.py](/home/luc/rl-nethack/scripts/generate_counterfactual_data.py)
  - one-step counterfactual branching from the same state
- [src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py)
  - hand-shaped task rewards
- [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)
  - closed-loop task evaluation and one-step greedy control
- [src/memory_tracker.py](/home/luc/rl-nethack/src/memory_tracker.py)
  - the memory layer we already need for partial observability

The current repo is therefore already good at:

- task definitions,
- rollout collection,
- state memory,
- counterfactual action comparison,
- closed-loop task evaluation.

The main missing pieces are:

- explicit options / skills,
- learned reward models from preferences,
- learned low-level skill controllers,
- high-level skill scheduling,
- joint multi-skill training.


## 3. What MaestroMotif Says We Should Change

The paper suggests one major shift:

- stop thinking of the problem as "optimize one scalar reward over primitive
  actions",
- start thinking of it as "learn a hierarchy of reusable skills".

For this repo, that means the target architecture should become:

1. high-level skill policy
2. low-level skill controllers
3. skill reward models
4. skill initiation / termination logic
5. trajectory evaluation at both skill and task-composition levels

That is a better extension of the current code than flat PPO.


## 4. Mapping MaestroMotif To Our Current Task Set

MaestroMotif’s NetHack skills are:

- Discoverer
- Descender
- Ascender
- Merchant
- Worshipper

Our current task families in
[src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py) are:

- `explore`
- `survive`
- `combat`
- `descend`
- `resource`

These are already close to a usable skill library.

The right move is to **promote them into real options**.

Recommended first skill set for this repo:

- `explore`
  - maps to Discoverer
- `descend`
  - maps to Descender
- `ascend`
  - currently missing, should be added
- `combat`
  - this repo needs it explicitly, even though MaestroMotif bundles some of it
    into other skills
- `survive`
  - our repo should keep this separate because low-HP escape behavior is a
    distinct failure mode
- `resource`
  - pickup / simple item handling

Longer-term, split `resource` into:

- `pickup`
- `merchant`
- `worship`
- `consume`

but not before the simpler set works.


## 5. Promote Tasks Into Real Options

Right now the tasks are reward/eval labels plus a one-step greedy controller.
MaestroMotif says they should become actual options:

- initiation condition
- termination condition
- low-level policy
- reward function

### What to add

Create a new module such as:

- [src/skill_options.py](/home/luc/rl-nethack/src/skill_options.py)

with an interface roughly like:

```python
class SkillOption:
    name: str
    def can_start(state, memory) -> bool: ...
    def should_stop(state, memory, step_in_skill) -> bool: ...
    def reward(obs_before, obs_after, ...) -> float: ...
    def allowed_actions(state, memory) -> list[str]: ...
```

### How that connects to existing code

- move task directives out of
  [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)
  into option specs
- keep reward math in
  [src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py)
- make `task_greedy` skill-aware by selecting an active option, then choosing
  actions only within that option’s action/support set

### Why this matters

Without options, the current controller re-decides the full problem every
single turn.

With options, the system gets:

- temporal abstraction,
- cleaner training signals,
- better compositional planning,
- more realistic evaluation.


## 6. Use LLMs As Judges, Not Actors

This is the single strongest idea in MaestroMotif for this repo.

We already proved that:

- LLMs can cheaply generate action-labeled trajectories,
- but low-level action generation is noisy,
- and hand-tuned reward shaping quickly becomes brittle.

The better role for the LLM is:

- judge which short trace better satisfies a skill description.

That means we should build **offline preference datasets**.

### What to add

Create:

- [scripts/generate_skill_preferences.py](/home/luc/rl-nethack/scripts/generate_skill_preferences.py)

This script should:

1. collect short trajectory snippets from the current harness
2. pair them up for the same skill
3. format a skill-specific judgment prompt
4. ask a local or remote LLM:
   - which snippet is better?
   - or tie / unclear
5. write a preference dataset

### The preference examples should include

For each snippet:

- current state summary
- skill name and natural-language directive
- 3 to 10 step trajectory
- HP / HP delta
- gold / XP / depth deltas
- visible threats
- memory summary from [src/memory_tracker.py](/home/luc/rl-nethack/src/memory_tracker.py)
- termination / death / loop signals

This is important because MaestroMotif reports that reward elicitation improves
when prompts include stats and recent changes, not just raw observations.

### Good initial preference questions

For `explore`:

- "Which snippet better explores safely and reveals new map area without
  getting stuck?"

For `survive`:

- "Which snippet better preserves the agent’s life and avoids reckless damage?"

For `descend`:

- "Which snippet better progresses toward and through stairs while staying
  viable?"

For `combat`:

- "Which snippet handles nearby threats more safely and effectively?"


## 7. Train Skill Reward Models

Once we have preference data, the next step is to fit a reward model per skill,
as in MaestroMotif and Motif.

### What to add

Create:

- [train_skill_reward.py](/home/luc/rl-nethack/train_skill_reward.py)

This should:

1. load pairwise preference data
2. encode each snippet into a feature/input representation
3. train a small reward model `r_phi(skill, snippet)` using Bradley-Terry loss

### Representation options

There are two reasonable starting points.

#### Option A: small structured MLP

Use structured features only:

- task metrics from [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)
- deltas from [src/state_encoder.py](/home/luc/rl-nethack/src/state_encoder.py)
- memory summary features

This is the cheapest and fastest path.

#### Option B: LM-based reward head

Use text snippets formatted similarly to the current forward-model training
format and train a small classifier / scalar head.

This is closer to the paper but heavier.

### Recommended order

Start with Option A first.

Reason:

- easier to debug
- faster training
- fewer moving parts
- enough to replace the current hand-tuned scalar

### How it plugs into current code

Replace the current hand-shaped `compute_task_rewards(...)` usage in
[src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py) with:

- `reward_model.score(skill, transition_or_snippet)`

Keep the hand-shaped reward as:

- a baseline,
- a bootstrapping source,
- a regression oracle.


## 8. Add Count-Normalization And Anti-Fixation

This is one of the best cross-paper lessons.

Motif explicitly uses count-based normalization to avoid fixation on repeated
observations.

We independently rediscovered the same issue in the current task harness:

- the early `task_greedy` controller looped badly,
- we had to add repeated-state / repeated-action penalties.

That means future learned reward models should include explicit anti-fixation
logic.

### What to add

Add support for:

- episodic observation-count normalization
- repeated-state penalties
- repeated-action penalties
- repeated-tile penalties
- reward thresholding for low-confidence outputs

### Where to add it

- support code in [src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py)
- rollout-time tracking in [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)
- later, reward-model postprocessing in the new reward-model inference layer

### Why this matters

Without this, a learned reward model can prefer:

- tiny repetitive gains,
- loops that preserve HP,
- short-term safe stagnation,
- reward hacking on superficial features.


## 9. Move From Step Prediction To Skill-Conditioned Prediction

Right now the main learned model predicts **one-step deltas**.

That is useful, but MaestroMotif suggests the more useful abstraction is:

- current state
- active skill
- short horizon
- predicted skill outcome

### What to add

Extend the forward-model training data so each example can optionally include:

- `skill_id`
- `steps_remaining_in_skill`
- `skill_start_state_summary`
- `skill_termination_flag`
- `cumulative_skill_reward`

### New training target families

1. one-step delta
2. short-horizon rollout summary
3. whether the skill should terminate
4. predicted cumulative reward over the next `k` steps

### Where to modify the code

- [scripts/generate_training_data.py](/home/luc/rl-nethack/scripts/generate_training_data.py)
  - add `skill` metadata and short-horizon summaries
- [train.py](/home/luc/rl-nethack/train.py)
  - eventually support a skill-conditioned dataset variant

### Why this matters

It turns the repo from:

- "predict one action outcome"

into:

- "predict what happens if I stay in this mode of behavior"

That is much closer to planning over skills.


## 10. Train Skills Jointly, Not In Isolation

MaestroMotif reports an important effect:

- training multiple skills in the same episodes creates an emergent curriculum
- training them in isolation is much weaker

This is very relevant here.

### What we should not do

Do not create five completely independent training scripts like:

- one rollout generator for `explore`
- another for `survive`
- another for `combat`

and train them in totally disjoint state distributions.

### What we should do

Create a **training-time policy over skills**:

- start in `explore`
- switch to `combat` when adjacent threat appears
- switch to `survive` when HP is low
- switch to `resource` when standing on useful items
- switch to `descend` when stairs are seen and explore is sufficiently complete

### Where this belongs

Add:

- [src/skill_scheduler.py](/home/luc/rl-nethack/src/skill_scheduler.py)

Start with a hand-coded scheduler first.

Then later:

- allow LLM-generated scheduler code,
- or a learned scheduler.

### How this connects to current code

It should wrap [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py),
not replace it.

The current harness already tracks:

- memory
- repeated states
- action counts
- task reward

That is exactly the information a scheduler needs.


## 11. Use A Shared Backbone Conditioned On Skill ID

MaestroMotif found that separate heads collapsed and that a shared backbone
conditioned on skill identity worked better.

That is very plausible here too.

### Recommendation

When we move to learned skill controllers or value models, do:

- one shared state encoder
- one skill embedding or one-hot
- one shared trunk
- skill-conditioned policy/value heads if needed

Avoid:

- training five completely disconnected models from scratch

### Why

The skills share a lot:

- map reading
- threat recognition
- memory use
- movement mechanics
- item recognition

So a shared representation should be much more sample-efficient.


## 12. Expand Evaluation Beyond Flat Task Reward

Current `task-evaluate` already helps, but it is still flatter than what the
paper suggests.

We should evaluate at three levels.

### Level 1: skill execution quality

For each skill:

- success rate
- cumulative skill reward
- loop rate
- average duration
- termination correctness

Examples:

- `descend`
  - did it reach stairs?
  - did it go down?
- `combat`
  - did it reduce adjacent hostiles?
  - how much HP did it lose?

### Level 2: option quality

When skills become real options:

- initiation precision
- termination precision
- average steps per option
- reward per option execution

### Level 3: composed tasks

Add multi-skill tasks like:

- explore until stairs -> descend
- explore -> pickup -> continue exploration
- explore -> combat -> survive -> descend

This is much closer to the MaestroMotif benchmark style.

### Where to add this

Extend:

- [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)

or add:

- [src/skill_eval.py](/home/luc/rl-nethack/src/skill_eval.py)


## 13. What To Build First

The correct order is not "implement PPO."

The correct order for this repo is:

### Phase 1: structure

1. create explicit option / skill interfaces
2. add initiation / termination logic
3. add scheduler scaffolding

### Phase 2: reward modeling

4. generate offline preference data for `explore` and `survive`
5. train a small reward model per skill
6. replace hand-shaped scalar scoring in `task_greedy`

### Phase 3: learned skill control

7. train a skill-conditioned low-level controller
8. keep the scheduler hand-coded at first
9. evaluate on skill-level and composed tasks

### Phase 4: hierarchy

10. add an LLM-generated or learned high-level policy over skills
11. add multi-skill planning
12. only then consider full end-to-end RL over primitive actions if still
    necessary


## 14. Proposed File-Level Plan

### New files

- [src/skill_options.py](/home/luc/rl-nethack/src/skill_options.py)
  - skill specs
- [src/skill_scheduler.py](/home/luc/rl-nethack/src/skill_scheduler.py)
  - training-time / eval-time high-level policy over skills
- [scripts/generate_skill_preferences.py](/home/luc/rl-nethack/scripts/generate_skill_preferences.py)
  - offline AI-feedback dataset generation
- [train_skill_reward.py](/home/luc/rl-nethack/train_skill_reward.py)
  - Bradley-Terry reward-model fitting
- [src/skill_eval.py](/home/luc/rl-nethack/src/skill_eval.py)
  - richer option/composition metrics

### Existing files to extend

- [src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py)
  - keep as bootstrap reward layer
- [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)
  - evolve from task-harness to option-harness
- [scripts/generate_counterfactual_data.py](/home/luc/rl-nethack/scripts/generate_counterfactual_data.py)
  - natural source of candidate snippets for preference comparisons
- [scripts/generate_training_data.py](/home/luc/rl-nethack/scripts/generate_training_data.py)
  - add skill-conditioned rollouts and richer metadata
- [train.py](/home/luc/rl-nethack/train.py)
  - later support skill-conditioned forward / value training variants


## 15. Risks And Failure Modes

The paper also points to the main risks.

### 1. Bad skill definitions

If the skills are vague or overlapping, the whole hierarchy gets muddy.

Mitigation:

- keep the first skill set small and concrete
- write explicit initiation / termination contracts

### 2. Reward model wireheading

If the reward model learns superficial proxies, the controller will exploit
them.

Mitigation:

- keep the hand-shaped reward baseline
- inspect preference prompts and errors
- use anti-fixation penalties
- run human spot checks

### 3. Brittle scheduler

A bad high-level skill-selection policy can waste all the low-level work.

Mitigation:

- start with rule-based scheduler
- benchmark skill transitions explicitly
- only later let an LLM synthesize selection logic

### 4. Overengineering too early

If we try to jump to the full MaestroMotif stack immediately, we will drown in
infrastructure.

Mitigation:

- implement the stack incrementally
- keep every phase benchmarkable
- never remove the current task harness until its replacement is clearly better


## 16. Bottom Line

The best way to integrate MaestroMotif into this repo is not:

- "replace everything with PPO"

It is:

- promote current tasks into options,
- use LLM feedback to learn reward models offline,
- train skill-conditioned low-level controllers,
- schedule skills jointly in shared episodes,
- plan and evaluate at the skill level.

That is the natural continuation of what this repo already does well.

The current task harness is a useful first control layer, but it is still too
flat. MaestroMotif gives the blueprint for the next version of the project:

- hierarchical,
- memory-conditioned,
- reward-model based,
- skill-compositional.

