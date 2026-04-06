# Repo Morph Plan 2026-04-06

## Thesis

The repo should stop being organized around "how do we tune APPO more?" and instead be organized around:

- build the strongest teacher we can
- measure that teacher with a trusted deterministic benchmark
- improve data and targets before scale
- treat online improvers as replaceable modules

The current evidence supports this:

- the teacher and representation are the strongest parts of the system
- the online improver is the weakest part of the system
- APPO plus teacher extras preserves quality briefly but does not reliably improve it
- behavior-reg, in its current form, is not yet a better improver either

## Desired Repo Shape

The intended shape is:

1. `teacher construction`
2. `teacher evaluation`
3. `teacher-data refinement`
4. `online improver modules`

That means the teacher is the mainline artifact and the online learner is downstream of it, not the other way around.

## New Top-Level Concepts

### 1. Teacher pipeline

The teacher pipeline should own:

- offline BC
- offline behavior-reg
- world-model-augmented teachers
- future learned proxy / short-horizon teacher heads

The output of this pipeline is a teacher checkpoint plus a report on held-out deterministic traces.

### 2. Teacher benchmark

The benchmark pipeline should own:

- held-out trace datasets
- checkpoint ranking
- disagreement reports
- action-slice metrics

This pipeline already mostly exists, but it should be treated as first-class rather than support tooling.

### 3. Teacher-data refinement

This layer should own:

- DAgger / relabeling
- weak-action slices
- disagreement slices
- merged datasets
- future online teacher refresh

### 4. Improver modules

Improvers should be distinct implementations with comparable inputs and outputs.

Examples:

- `appo_teacher_replay`
- `behavior_reg_teacher_init`
- future `demo_actor_critic`
- future `teacher_proxy_actor_critic`

## Codebase Changes

### Phase 1: Clarify ownership in the current tree

Without moving the whole repo at once, the code should begin to separate into clearer categories.

#### Teacher-oriented modules

- [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py)
- [rl/train_behavior_reg.py](/home/luc/rl-nethack/rl/train_behavior_reg.py)
- [rl/train_world_model.py](/home/luc/rl-nethack/rl/train_world_model.py)
- [rl/world_model_features.py](/home/luc/rl-nethack/rl/world_model_features.py)

#### Benchmark-oriented modules

- [rl/trace_eval.py](/home/luc/rl-nethack/rl/trace_eval.py)
- [rl/checkpoint_tools.py](/home/luc/rl-nethack/rl/checkpoint_tools.py)
- [rl/debug_tools.py](/home/luc/rl-nethack/rl/debug_tools.py)

#### Data-refinement modules

- [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
- [rl/dagger.py](/home/luc/rl-nethack/rl/dagger.py)

#### Improver-oriented modules

- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)

### Phase 2: Add explicit teacher artifacts

The repo should gain a canonical way to describe a teacher artifact.

Minimum metadata:

- source dataset
- observation version
- world-model feature mode
- held-out trace score
- weak-action slice score
- disagreement summary path

This should live beside model checkpoints so every teacher is inspectable.

### Phase 3: Add improver manifests

Every online improver run should describe:

- teacher source checkpoint
- replay source
- reward objective
- trace gate
- best learned checkpoint

This makes it possible to compare improvers without reverse-engineering CLI commands.

## Immediate Engineering Plan

### Step 1

Add teacher artifact reporting for:

- BC
- behavior-reg

### Step 2

Add a single helper that emits a teacher report from any offline checkpoint:

- held-out trace score
- disagreement report
- weak-action shard score if provided

### Step 3

Add improver run metadata that points back to the teacher artifact it used.

### Step 4

Keep APPO as one improver branch, but stop treating it as the repo’s center.

## Operational Rules

1. A new teacher must beat or match the current teacher on the trusted benchmark before it becomes a default source for online runs.
2. A new improver must beat the current learned checkpoint baseline before it gets a larger run.
3. World-model changes that do not improve teacher quality should not dominate the roadmap.
4. New online improvers should consume canonical teacher artifacts, not ad hoc checkpoint paths.

## First Concrete Refactor

The first concrete move should be small and high-signal:

- add a `teacher_report` path to offline trainers
- write that report automatically after BC and behavior-reg training
- make the report include trusted trace evaluation when a held-out trace set is supplied

This does not change the learning algorithm, but it changes the repo state in the right direction:

- the teacher becomes the explicit artifact
- evaluation becomes inseparable from training output

## Success Criteria

The repo has moved in the right direction if:

1. offline teachers are easier to compare than online runs
2. every improver run clearly names the teacher artifact it uses
3. it is easy to swap improver families without rethinking the whole repo
4. the main loop becomes:
   - build teacher
   - validate teacher
   - improve teacher data
   - try online improver

## Next Code Task

Implement canonical teacher report generation for:

- [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py)
- [rl/train_behavior_reg.py](/home/luc/rl-nethack/rl/train_behavior_reg.py)
- [cli.py](/home/luc/rl-nethack/cli.py)

and use that as the first real code move toward the new repo shape.
