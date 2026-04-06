# WAGMI

## Purpose

This document is a working decision memo for the repo.

It is trying to answer six concrete questions:

1. Have we beaten the teacher?
2. If not, what exactly is still broken?
3. Which parts of the current system are now trustworthy?
4. Which explanations for failure are no longer plausible?
5. What does the literature suggest we should do next?
6. What exact code changes should follow from that conclusion?

This is not just a run report.
It is the current best theory of the project.

## Reader Guide

This document is long because it serves multiple purposes at once.

If you only need one thing:

- to understand the current diagnosis:
  read `Executive Summary`, `Core Thesis`, `Strongest Current Hypothesis`, and `Final Bottom Line`

- to continue the repo:
  read `Codebase Map`, `Concrete Repo Changes`, `Immediate Experiment Plan`, and `Operational Rules For The Next Phase`

- to sanity-check whether the diagnosis is warranted:
  read `Evidence Table`, `Strongest Positive Evidence`, `Strongest Negative Evidence`, `Counterarguments And Why I Reject Them`, and `If We Are Wrong`

- to understand how we got here:
  read `Project Timeline` and `Experimental Ledger`

## Executive Summary

No, we have not beaten the teacher yet.

The current best trusted result on the strongest world-model branch is still a tie with the offline teacher clone, not a win:

- best large-run checkpoint:
  `train_dir/rl/appo_wm_v4_x10_ckpt10_rs0005_replay001_vloss01_g099_lr1e4/checkpoint_p0/checkpoint_000000044_352.pth`
- held-out deterministic trace match:
  `0.9375`
- best checkpoint happened very early:
  `352` env steps

That is still meaningful progress because older 500-step branches collapsed badly. The current branch no longer collapses immediately. But it is a **teacher-preservation** regime, not yet a **teacher-improvement** regime.

The highest-confidence conclusion now is:

**the repo has crossed from infrastructure-limited to improver-limited.**

Earlier in the project, bad results could reasonably be blamed on:

- broken APPO integration
- broken warm-start transfer
- broken evaluation
- broken checkpoint handling
- weak representation

That is no longer the strongest explanation.

The strongest explanation now is:

- we can build a strong offline teacher
- we can faithfully clone it into the online learner
- we can keep it from collapsing instantly
- but our current online improver does not generate safe positive updates beyond that teacher

That is a much narrower problem, and therefore a much better one.

## Core Thesis

The core thesis of this document is:

**the world-model branch fixed enough of the representation floor that the next bottleneck is the learning rule, not the feature plumbing.**

More concretely:

- the world model helped build a stronger teacher
- the bridge to APPO can now preserve the teacher at step 0
- deterministic trace evaluation can now detect whether online learning helped or hurt
- the current APPO + static teacher replay branch is stable enough to study
- and that branch still does not improve on the teacher

So the next work should focus on:

- how teacher data influences learning over time
- how replay is scheduled and prioritized
- whether APPO is even the right improver class for this phase

Not on:

- more representation churn first
- more blind scale first
- more reward-based interpretation first

## What "Beat The Teacher" Means Here

This repo currently has several different notions of "better":

1. higher live reward during training
2. higher live reward in seeded evaluation
3. better held-out deterministic trace match
4. better longer-horizon gameplay

Historically in this repo, these have diverged.

For now, the only trustworthy objective is:

- **held-out deterministic trace match**

because:

- live seeded evaluation was shown to be nondeterministic
- reward repeatedly improved while teacher alignment got worse
- trace evaluation already caught regressions that live reward hid

So for the current project phase:

- "beat the teacher" means:
  produce a learned checkpoint whose held-out trace match exceeds the current teacher on the same trace set

That is a narrow target, but it is the right target until the live evaluation stack is much more trustworthy.

## Assumptions

This document rests on several assumptions. They should be explicit.

### Assumption 1: the held-out trace benchmark is the current source of truth

Why I accept it:

- it is deterministic
- it is already catching real regressions
- nothing else in the repo currently has the same reliability

### Assumption 2: the current teacher is worth beating

The current teacher is not a perfect NetHack agent. It is a strong local policy object under the current task framing.

Why I accept it:

- it is better than earlier learned policies
- warm-starting from it clearly helps
- preserving it is already nontrivial

### Assumption 3: improving trace match is still the right near-term goal

Why I accept it:

- the repo still lacks a stronger trustworthy long-horizon online benchmark
- we need one reliable target before broadening the scope again

### Assumption 4: the current bottleneck is algorithmic, not just model capacity

Why I accept it:

- offline teacher quality is already high
- step-0 warm-start quality is already high
- deterioration starts after online updates begin

That pattern points much more strongly to update geometry than to insufficient width.

## What We Know vs What We Infer

This distinction matters because too much of this project used to run on fuzzy intuitions.

### Things we know

- deterministic trace evaluation is more trustworthy than live seeded evaluation in this repo
- the current best learned checkpoint on the strongest world-model branch is `0.9375`
- that best checkpoint occurs early, at `352` env steps
- later checkpoints in the current large run have not displaced that result
- world-model augmentation improved offline teacher quality
- the APPO warm-start path can now faithfully clone the teacher at step 0
- teacher replay improves stability compared with branches that lacked it

### Things we infer

- the current bottleneck is the online improver, not the representation
- teacher replay needs to be scheduled and prioritized
- uniform replay is too weak
- APPO may not be the right final improver class

### Things we do not yet know

- whether scheduled replay alone is enough to beat the teacher
- whether a demo-aware critic is the missing ingredient
- whether a behavior-regularized learner will outperform APPO in this repo
- how well the current conclusions transfer from the trace benchmark to longer-horizon live play

## Non-Goals

This document is not trying to:

- claim we are close to solving full-game NetHack
- argue that APPO is useless in general
- argue that world models were a waste
- optimize live reward at the expense of alignment
- replace the whole repo with a totally new architecture overnight

The problem now is narrower:

- find the right online improver once the teacher and representation are already strong

## Project Timeline

This timeline matters because it shows where progress really came from.

### Phase 1: APPO exists at all

What happened:

- a real APPO backend was built
- environment integration, trainer, evaluation, and CLI were added
- end-to-end learned RL became possible

What we learned:

- the repo had real RL
- but policy quality was poor

Main issue then:

- the basic RL infrastructure existed, but the learner was weak and hard to trust

### Phase 2: trusted evaluation

What happened:

- deterministic trace evaluation was added
- checkpoint ranking by held-out trace match was added
- live seeded evaluation was demoted to diagnostic-only

What we learned:

- many earlier "improvements" were not actually trustworthy
- the project needed one reliable metric before anything else

Main issue then:

- objective mismatch became visible

### Phase 3: teacher-regularized APPO

What happened:

- teacher CE / KL was added
- BC warm-start path was repaired
- targeted action weighting was explored

What we learned:

- teacher regularization helps preserve competence
- but static teacher loss alone does not create teacher-beating learning

Main issue then:

- the learner was anchored, but still drifted

### Phase 4: world-model representation branch

What happened:

- a world-model feature path was built
- `v4 + wm_concat_aux` became the strongest offline teacher branch
- weak direction slices improved materially

What we learned:

- representation really was part of the problem
- the world model helped the policy "see" the state better
- but online improvement still plateaued

Main issue then:

- representation was no longer the main blocker

### Phase 5: conservative 500-step APPO with teacher replay

What happened:

- episode horizon increased to `500`
- reward scale was lowered
- learning rate was lowered
- value-loss pressure was lowered
- teacher replay was added
- checkpoint cadence increased

What we learned:

- catastrophic collapse was mostly fixed
- the branch was stable enough for an x10 run
- the best checkpoint still happened very early

Main issue now:

- the learner preserves early teacher quality, but still does not improve on it

## Why The Timeline Matters

The project history is not just narrative; it determines what explanations are still plausible.

For example:

- if we had never fixed the warm-start bridge, then "APPO is bad" would be ambiguous
- if we had never added deterministic trace evaluation, then "teacher replay helped" would be ambiguous
- if we had never built the world-model branch, then "representation is still the main blocker" would remain very plausible

The timeline removes those escape hatches.

That is why this document is confident in a way that earlier notes could not be.

## What Changed The Project The Most

Three changes mattered more than almost everything else.

### 1. Deterministic trace evaluation

Before this, we could not trust the scientific readout.
After this, we could.

This single change made the rest of the project interpretable.

### 2. Warm-start equivalence fixes

Before this, APPO often started from a policy that only looked like the teacher on paper.
After this, the step-0 clone actually matched the teacher.

This removed a large amount of false ambiguity.

### 3. `v4 + wm_concat_aux`

Before this, the policy representation was clearly too weak in the directional slices that mattered.
After this, the offline teacher got much stronger.

This raised the quality ceiling of the repo.

## What Did Not Matter As Much As Expected

There were also several ideas that were worth trying, but turned out not to be the main lever.

### 1. Generic intrinsic bonus tuning

The state-hash bonus and similar experiments were not useless, but they did not address the core alignment/drift problem.

### 2. Generic DAgger without focus

Plain relabel-and-merge loops were too blunt.
They did not target the states where the learner actually needed help.

### 3. Static teacher-loss coefficient sweeps

These helped identify safe ranges, but they did not unlock teacher-beating learning.

### 4. More static APPO budget on unstable branches

Those runs were informative for diagnosis, but not productive for progress.

This matters because it tells us what *not* to overinvest in next.

## Evidence Table

| Branch | Regime | What happened |
|---|---|---|
| early APPO | no strong teacher constraint | policy collapsed or badly underperformed |
| repaired warm-start APPO | faithful step-0 clone | showed that bridge bugs were real and fixable |
| teacher-reg APPO | static teacher CE | preserved teacher better but still drifted |
| `v4 + wm_concat_aux` offline | world-model augmented teacher | offline teacher quality improved materially |
| 500-step APPO + teacher replay | conservative PPO + replay | first branch that stayed near the teacher instead of collapsing |
| current x10 500-step run | same static replay schedule | best checkpoint still at `352` env steps, `0.9375`, no later improvement |

This means:

- feature quality improved
- stability improved
- teacher preservation improved
- actual online improvement has still not happened

## Strongest Positive Evidence

We should not undersell what has genuinely improved.

### Positive fact 1: teacher preservation was hard and is now partially solved

That is a real project milestone.

### Positive fact 2: teacher replay helped weak branches

Even when it did not create a teacher-beating run, it converted catastrophic failure into stable proximity.

### Positive fact 3: world-model augmentation improved offline teacher quality

That means the world-model work was not a distraction.

### Positive fact 4: the project is now disciplined enough to support principled next experiments

This is easy to overlook, but it matters a lot.
We no longer have to guess blindly.

## Strongest Single Positive Sign

If I had to pick one encouraging signal from the entire project so far, it would be:

- the repo can now build a strong offline teacher and carry that policy faithfully into the online learner

That was not true earlier.
It changes the nature of the problem from:

- "can we even start from something good?"

to:

- "how do we improve something good without breaking it?"

That is a much more tractable stage of the project.

## Strongest Negative Evidence

The negative evidence is equally important.

### Negative fact 1: the best checkpoint happens almost immediately

The current best large-run checkpoint still occurs at:

- `352` env steps

That is not what "steady improvement" looks like.
It is what "preserve early, then plateau or drift" looks like.

### Negative fact 2: later training does not displace the early checkpoint

That means the learner is not discovering a better policy later and partially forgetting it.
It is simply not finding better updates than the earliest safe ones.

### Negative fact 3: replay helps stability, but static replay does not create gains

So the general idea is right, but the current implementation shape is too weak.

### Negative fact 4: better representation alone did not unlock improvement

This is why the world-model branch should now be treated as enabling infrastructure rather than the main research bottleneck.

## Strongest Single Negative Sign

If I had to pick one discouraging signal from the current branch, it would be:

- the best checkpoint still happens before the learner has had time to do anything meaningfully "RL-like"

That strongly suggests the online learner is not yet contributing genuine improvement.

## What The Codebase Says

After reviewing the current RL path, the main facts are:

### 1. The representation problem got better

The `v4 + wm_concat_aux` path is real.

- world-model feature augmentation lives in [rl/world_model_features.py](/home/luc/rl-nethack/rl/world_model_features.py)
- offline world-model training/eval lives in:
  - [rl/world_model.py](/home/luc/rl-nethack/rl/world_model.py)
  - [rl/world_model_dataset.py](/home/luc/rl-nethack/rl/world_model_dataset.py)
  - [rl/train_world_model.py](/home/luc/rl-nethack/rl/train_world_model.py)
  - [rl/world_model_eval.py](/home/luc/rl-nethack/rl/world_model_eval.py)

This branch improved offline teacher quality. The world model is helping as representation learning.

More specifically, the current representation stack now has three layers:

1. handcrafted feature encoding in [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)
2. world-model latent and action-logit augmentation in [rl/world_model_features.py](/home/luc/rl-nethack/rl/world_model_features.py)
3. policy consumption through BC/APPO input widths tracked in:
   - [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py)
   - [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
   - [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)

That stack is now coherent enough that representation should no longer be treated as the main blocker.

### 2. The warm-start bridge is no longer the main bug

Earlier in the project, APPO warm-start corrupted the teacher because of:

- hidden-size mismatch
- normalization mismatch
- nonlinearity mismatch
- recurrent bridge mismatch

Those were fixed in the APPO path. The current branch can preserve the teacher clone at step 0.

That changes the interpretation of every later run:

- before: a bad APPO result could mean the bridge was broken
- now: a bad APPO result usually means the online updates are bad

This is one of the most important project-state changes so far.

### 3. The current improver is teacher-constrained, but still static

Teacher regularization and teacher replay are implemented in [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py).

Right now the learner adds:

- teacher CE/KL on current minibatch states
- replay CE on teacher trace rows
- optional parameter anchor

But the regime is still fundamentally static:

- fixed `teacher_loss_coef`
- fixed `teacher_replay_coef`
- fixed reward scale
- fixed PPO/APPO update rule

The code already has fields for scheduled teacher loss:

- `teacher_loss_final_coef`
- `teacher_loss_warmup_env_steps`
- `teacher_loss_decay_env_steps`

But the current successful branch is still basically:

- flat teacher pressure
- flat replay
- conservative PPO

There are three deeper problems with the current implementation:

1. **teacher replay is uniform**
   - replay rows are sampled uniformly from a fixed trace file
   - there is no prioritization toward disagreement, failure, or weak actions

2. **teacher replay is additive, not structural**
   - the main learner is still an on-policy actor-critic update
   - teacher replay is appended as one more loss term
   - replay data is not shaping the full optimization distribution

3. **teacher replay is policy-head-heavy**
   - it constrains the policy head
   - it does not currently give the value path a stronger demonstration-aware target
   - the critic is still mostly learning from online returns

These three issues together are almost certainly why the branch stabilizes but does not improve.

### 4. The trusted evaluation is correct, and it says we are flat

The trusted metric is deterministic trace match:

- [rl/trace_eval.py](/home/luc/rl-nethack/rl/trace_eval.py)
- trace-gated checkpoint selection in [rl/checkpoint_tools.py](/home/luc/rl-nethack/rl/checkpoint_tools.py)

This is the right source of truth. It is telling us:

- the current branch no longer collapses immediately
- but it also does not learn past the teacher
- the best checkpoint is still an early checkpoint

That pattern is now consistent enough to believe.

We should treat this as a solved methodological question:

- the trace metric is not a convenience
- it is the current scientific foundation of the project

Any next-step plan that does not center that metric is probably wrong.

## Codebase Map

This is the shortest practical map of where the important logic now lives.

### Policy / teacher / replay

- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
  - current-state teacher CE/KL
  - teacher replay CE
  - parameter anchor
  - learner patching

### APPO config and launch

- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [cli.py](/home/luc/rl-nethack/cli.py)

### Trusted evaluation

- [rl/trace_eval.py](/home/luc/rl-nethack/rl/trace_eval.py)
- [rl/checkpoint_tools.py](/home/luc/rl-nethack/rl/checkpoint_tools.py)
- [rl/evaluate.py](/home/luc/rl-nethack/rl/evaluate.py)

### Offline teacher training

- [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py)
- [rl/train_behavior_reg.py](/home/luc/rl-nethack/rl/train_behavior_reg.py)

### Trace generation and DAgger

- [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
- [rl/dagger.py](/home/luc/rl-nethack/rl/dagger.py)

### World model

- [rl/world_model.py](/home/luc/rl-nethack/rl/world_model.py)
- [rl/world_model_dataset.py](/home/luc/rl-nethack/rl/world_model_dataset.py)
- [rl/world_model_features.py](/home/luc/rl-nethack/rl/world_model_features.py)
- [rl/train_world_model.py](/home/luc/rl-nethack/rl/train_world_model.py)
- [rl/world_model_eval.py](/home/luc/rl-nethack/rl/world_model_eval.py)

This map matters because the next algorithmic changes should mostly touch:

- `teacher_reg.py`
- `traces.py`
- `dagger.py`
- a possible new improver file

## Deeper Code Review Findings

A second pass through the implementation yields a few important clarifications.

### 1. The current "behavior-regularized" trainer is not yet the next online improver

The file [rl/train_behavior_reg.py](/home/luc/rl-nethack/rl/train_behavior_reg.py) is useful, but its name can mislead us if we are not careful.

What it actually does today:

- trains an offline MLP policy
- uses imitation loss to teacher actions
- adds a KL term toward the global action prior
- optionally reweights classes / action classes
- selects the best offline epoch by held-out trace match if a held-out file is provided

What it does **not** do today:

- online improvement
- advantage-weighted updates
- value-learning from demonstrations
- actor-critic fine-tuning
- offline-to-online policy optimization in the AWAC sense

So this file is valuable as:

- an offline teacher builder
- a weak-action repair tool

But not yet as:

- the true next behavior-regularized online improver

This distinction matters because it means we should not overclaim what the current repo already has.

### 2. The current DAgger path is still generic in how it mixes data

The file [rl/dagger.py](/home/luc/rl-nethack/rl/dagger.py) is already real and useful, but it currently has a fairly blunt data-mixing story.

What it actually does today:

- generate relabeled student traces
- merge them with base traces according to:
  - `base_only`
  - `uniform_merge`
  - `weighted_recent`
- retrain a BC model
- optionally stop a schedule on held-out regression

What it does **not** yet do:

- prioritize disagreement states
- prioritize weak teacher actions
- prioritize loop/failure states
- prioritize recent student mistakes

So the current DAgger tooling is closer to:

- "generic iterative relabeling"

than to:

- "surgical relabeling where the student currently fails"

That explains why earlier generic DAgger passes did not change the trajectory enough.

### 3. The checkpoint monitor is useful but intentionally coarse

The file [rl/checkpoint_tools.py](/home/luc/rl-nethack/rl/checkpoint_tools.py) gives us one of the highest-value pieces of infrastructure in the repo: automatic trace-gated checkpoint tracking.

But we should be precise about what it does and does not guarantee.

What it does:

- poll checkpoint files
- evaluate checkpoints once they pass the configured env-step interval
- record `best_trace_match.json`
- materialize `best_trace_match.pth`

What it does not do:

- evaluate every gradient update
- give us a fully dense view of policy quality over time
- guarantee that the exact best possible micro-checkpoint was observed

So when the document says:

- "best checkpoint is at 352 env steps"

that means:

- "best among the saved/evaluated checkpoints under the current cadence"

That is still scientifically useful, but it is worth stating explicitly.

### 4. The current teacher replay path is policy-focused, not trajectory-focused

The replay tensors loaded in [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py) currently contain:

- `feature_vector`
- teacher action label
- allowed action mask

That means the replay signal is fundamentally:

- supervised action imitation on isolated rows

It is not yet:

- sequence-aware
- return-aware
- value-aware
- advantage-aware

This matters because it reinforces the earlier diagnosis:

- replay helps keep the policy near the teacher
- replay does not yet tell the learner how to improve over trajectories

### 5. The world-model augmentation path is feature-only in the online learner

The file [rl/world_model_features.py](/home/luc/rl-nethack/rl/world_model_features.py) currently augments observations with:

- latent features
- optionally world-model action logits

This is useful and already working.

But in the online learner, that augmentation is still mostly serving as:

- better input features

It is not yet serving as:

- an auxiliary prediction loss
- a latent consistency target
- a rollout/planning module

So the current world-model branch should be understood as:

- representation improvement

not:

- full model-based RL

### 6. The current online reward is still mostly local intrinsic reward

The reward construction in [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py) and [rl/rewards.py](/home/luc/rl-nethack/rl/rewards.py) is important enough to state explicitly.

The environment currently computes:

- `total_reward = extrinsic_weight * env_reward + intrinsic_weight * (skill_reward + episodic_bonus)`

And the current configuration family has largely favored:

- `extrinsic_weight = 0.0`
- `intrinsic_weight = 1.0`

So the learner is not mostly optimizing true long-horizon environment return.
It is mostly optimizing:

- local shaped skill reward

This does not mean the reward path is useless.
It does mean we should not expect that reward alone to pull the learner beyond the teacher on the trusted benchmark.

### 7. The current 500-step branch is an intentional local regime, not the repo’s full horizon capability

From [rl/config.py](/home/luc/rl-nethack/rl/config.py):

- repo default `max_episode_steps` is now `5000`

But the branch we are currently analyzing has deliberately been running:

- `env_max_episode_steps = 500`

That distinction matters because it separates:

- what the codebase can support

from:

- what the current scientific loop is actually testing

The present branch is intentionally local. It is not yet claiming to solve full-horizon NetHack.

### 8. Checkpoint cadence is part of the measurement instrument

The current project state depends heavily on [rl/checkpoint_tools.py](/home/luc/rl-nethack/rl/checkpoint_tools.py).

Because the monitor only evaluates saved checkpoints, dense saving is not just an operational convenience.
It is part of the scientific instrument.

Without frequent checkpoints, the repo would not have discovered that the current best branch peaks extremely early.

That means:

- save cadence belongs in the experimental method

not just in training hygiene.

## Ownership Of The Next Changes

If someone were to take this repo over tomorrow, the next phase breaks down cleanly by file responsibility.

### Schedule and replay logic

- primary file:
  [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)

This file owns:

- teacher CE/KL
- replay CE
- replay sampling
- summary reporting of teacher-related losses

### Config surface

- primary files:
  [rl/config.py](/home/luc/rl-nethack/rl/config.py)
  [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
  [cli.py](/home/luc/rl-nethack/cli.py)

These files own:

- new schedule knobs
- replay focus modes
- experiment presets

### Trace enrichment

- primary file:
  [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)

This file should own:

- disagreement markers
- weak-action markers
- loop/failure markers
- replay-priority metadata

### Relabeling / focused DAgger

- primary file:
  [rl/dagger.py](/home/luc/rl-nethack/rl/dagger.py)

This file should own:

- disagreement-focused data aggregation
- failure-focused aggregation
- held-out-gated schedule logic

### New improver branch

- likely new file:
  `rl/train_awac.py`
  or
  `rl/train_demo_actor_critic.py`

That branch should be kept intentionally separate from the current APPO path so results remain interpretable.

## What We Actually Learned

### The good news

We are no longer blocked on plumbing.

The following are now true:

- the RL harness is real
- deterministic trace evaluation is real
- BC warm-start is real
- teacher regularization is real
- teacher replay is real
- world-model feature augmentation is real
- 500-step episodes do not automatically explode anymore

That matters. We are now failing in an interesting way, not a broken way.

### The bad news

The online RL improver is still not actually improving.

The current pattern is:

1. build a strong offline policy
2. initialize APPO from it
3. preserve it briefly
4. plateau or drift

So the bottleneck is no longer:

- feature wiring
- evaluation determinism
- checkpoint loading
- warm-start corruption

The bottleneck is:

**the online improvement rule is too weak or too poorly constrained to produce better-than-teacher behavior.**

## What The Current Repo Already Proves

This is an important section because it separates "we think" from "we have already demonstrated."

The current repo already proves:

1. We can build an offline teacher that is meaningfully strong under the current benchmark.
2. We can transfer that teacher into APPO without corrupting it at step 0.
3. We can run a 500-step online regime without immediate total collapse.
4. We can automatically rank checkpoints by the trusted metric.
5. We can improve offline weak-action behavior through targeted reweighting.
6. We can use world-model augmentation to improve offline teacher quality.

That is a lot.

What the current repo does **not** yet prove:

1. that APPO can beat the teacher under the current setup
2. that generic DAgger can fix the online plateau
3. that teacher replay alone is enough if left static
4. that the current trace benchmark is sufficient for final claims about long-horizon gameplay

## A More Precise Statement Of The Problem

The current problem is not:

- "the agent does not know how to act"

The teacher already knows how to act reasonably well.

The current problem is not:

- "the model cannot represent the right policy"

The offline teacher already gives evidence against that.

The current problem is:

- "the learner does not know how to move from a good demonstrated policy to a better policy without first moving to a worse one"

That is exactly the kind of problem where demonstration-regularized improvement methods exist.

## Failure Taxonomy

It is useful to separate the different kinds of failure we have already passed through.

### Failure type A: harness failure

Examples:

- broken APPO warm start
- nondeterministic live evaluation used as if it were a real benchmark
- wrong checkpoint selection
- parser/argument bugs

Status:

- mostly fixed

### Failure type B: representation failure

Examples:

- directional confusion
- weak frontier awareness
- poor weak-action recall on `south` / `west`

Status:

- materially improved by `v4` and the world-model branch
- not fully solved, but no longer the dominant bottleneck

### Failure type C: optimization instability

Examples:

- catastrophic collapse after a few updates
- huge value-loss spikes with immediate policy degradation

Status:

- improved by lower LR, lower reward scale, lower value loss, denser checkpoints
- still present in some regimes, but not the dominant failure of the current best branch

### Failure type D: objective / improvement failure

Examples:

- best checkpoint occurs almost immediately
- later RL updates do not beat the teacher
- static replay preserves but does not improve

Status:

- this is now the main bottleneck

This breakdown matters because it shows why "just debug more" is no longer enough.
We now need a different training logic, not just fewer bugs.

## Failure Sequencing

Another useful way to think about the project is by failure order.

Earlier failures were upstream failures:

- bad data
- bad features
- bad bridge
- bad evaluation

Current failures are downstream failures:

- bad online update geometry
- bad control of when teacher influence decays
- bad use of teacher data inside the learner

This sequencing matters because upstream fixes are prerequisite to downstream diagnosis.
We now have enough upstream fixes that downstream diagnosis is worth trusting.

## Counterarguments And Why I Reject Them

### Counterargument 1: maybe we just need more scale

Why I reject it:

- the best checkpoint happens extremely early
- later checkpoints do not improve on it
- that is not what "undertrained but improving" looks like

### Counterargument 2: maybe we just need a bigger model

Why I reject it:

- the offline teacher is already strong
- the clone is already strong
- deterioration starts after optimization, not before

That pattern does not point first to capacity.

### Counterargument 3: maybe the trace metric is too narrow

Why I only partly accept it:

- yes, it is narrow
- but it is still the best metric we currently have
- and the repo is not yet trustworthy enough on long-horizon live evaluation to replace it

So even if it is incomplete, it is still necessary.

### Counterargument 4: maybe the world model is still too weak

Why I reject it as the main explanation:

- the offline teacher improved
- the online branch still plateaued
- representation improved, but the improver did not

### Counterargument 5: maybe APPO is fine and we just tuned the wrong scalar

Why I reject it:

- we already explored many scalar-only adjustments
- the branch changed from collapsing to stable, but not from stable to improving
- that suggests a structural issue, not a missing lucky coefficient

## What Would Be A Bad Next Move

The next bad moves are easy to list now.

### Bad move 1: another large static replay run with slightly different scalar weights

Reason:

- that is unlikely to change the qualitative behavior we are already seeing

### Bad move 2: abandoning the teacher and hoping longer RL horizons solve it

Reason:

- the repo has repeatedly shown that unguided online optimization is not trustworthy enough yet

### Bad move 3: rebuilding the representation stack again before changing the improver

Reason:

- we already paid the cost of making the teacher stronger
- the online learner is now the limiting factor

### Bad move 4: treating the current plateau as "close enough"

Reason:

- a teacher tie is not the same as a teacher beat
- if we stop here, we have not actually solved the core project problem

## Why We Are Stuck

I think there are four interacting reasons.

### 1. We are still using the wrong improver

The current online improver is APPO with extra teacher losses bolted on.

That can preserve a teacher.
It does not automatically make the policy better than the teacher.

The current large run says exactly that:

- early checkpoint equals the teacher
- later checkpoints do not beat it

That means APPO is functioning more like a fragile fine-tuner than a reliable constrained improver.

This is the deepest change in how we should reason about the repo:

- APPO is no longer the unquestioned center of gravity
- APPO is now just one candidate improver

If it cannot move beyond preservation with a strong teacher and a repaired representation, it should lose architectural priority.

### 2. Teacher replay is too weakly integrated

In the current code, teacher replay is sampled in [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py) and added as an extra CE term on replayed trace rows.

That helps, but it is still a side loss.

It is not yet:

- prioritized
- scheduled by phase
- mixed with stronger demo-weighted value targets
- used to form the main training distribution

So the replay signal is stabilizing, but not dominating enough to guide improvement.

In practical terms:

- replay is helping the model remember what "good" looks like
- replay is not telling the model where it can safely get better

### 3. We are trying to beat the teacher with a sparse and still imperfect reward

Even with `reward_scale=0.005`, the actor-critic still sees the online scalar reward as the only real improvement signal.

That reward is not the same as the trusted objective.

So the run keeps telling us:

- teacher-aligned trace score stays flat
- reward/loss dynamics keep moving

That is classic objective mismatch.

The project has now seen this enough times that it should be treated as a design fact, not a temporary annoyance.

### 4. The current 500-step branch is still a local-skill regime

We already established:

- `200` steps was too short
- `500` steps is better, but still not remotely full-game NetHack

So the current 500-step branch is a local-skill training regime, not a full-game regime.

That means:

- it is long enough to reveal instability
- but still too short to rely on generic reward as a strong teacher-beating signal

In other words, it is exactly the regime where demonstration-guided improvement matters most.

The right conclusion from that is not:

- "RL is impossible here"

The right conclusion is:

- teacher-guided improvement matters even more here than in easier domains

## Why I Think This Is Solvable

There are three reasons I still think the problem is solvable.

### Reason 1: we already solved a string of harder-than-expected infrastructure problems

That suggests the repo is capable of converging on the next bottleneck too.

### Reason 2: the current failure mode is narrow

The project is not failing randomly anymore.
It is failing in one specific, repeated way:

- preserve early
- then plateau or drift

That is exactly the kind of repeated pattern that can usually be attacked systematically.

### Reason 3: the literature has several families of methods explicitly designed for this kind of setting

We are not trying to invent a solution from scratch.
We are trying to choose the right one for the repo.

## What "Teacher-Beating" Probably Requires

Based on the current repo and the literature, beating the teacher probably requires all of the following together:

1. a strong teacher representation
2. a faithful step-0 clone
3. a demonstration-heavy early update regime
4. a controlled release of that constraint
5. replay targeted at the states where the learner actually drifts
6. a value-learning story that is not dominated by noisy online targets too early

We already have:

- 1
- 2
- part of 3

We do not yet really have:

- 4
- 5
- enough of 6

That decomposition is useful because it tells us what work remains.

## The Value Path Problem

One under-emphasized issue in the current code is the critic.

The current branch constrains the actor more directly than the value function.

That creates a possible mismatch:

- actor is partially constrained toward the teacher
- critic is still mostly shaped by online returns
- critic then produces advantages that may push the actor off the teacher manifold

This is a strong reason to believe that:

- replaying only policy labels is not enough

and that:

- a demo-aware value-learning story may be necessary

This does not prove APPO cannot work, but it does explain why the current replay path may only preserve rather than improve.

## The Objective Path Problem

There is also a deeper objective-level issue visible from the code.

The current online learner is using:

- local shaped skill reward
- optionally a learned reward model
- optional episodic exploration bonus

But the trusted benchmark is:

- deterministic teacher trace match

Those are not the same thing.

So even if:

- the actor were stable
- the critic were stable

we should still expect some divergence between:

- what the learner is incentivized to optimize

and:

- what we use to decide whether it got better

This reinforces an important project truth:

- teacher replay is not just a stabilization tool
- it is also one of the only alignment tools currently acting inside the online learner

## The Policy Path Problem

There is also a policy-side version of the same issue.

Right now, teacher replay is acting on rows sampled from a fixed trace file.
That means the policy sees teacher supervision on:

- a static distribution of teacher states

What it does not yet get enough of is:

- supervision concentrated on the exact states it starts visiting once it deviates

That is why a better schedule alone may not be enough.
We may need:

- better replay scheduling
- and
- better replay content

at the same time.

## What The Literature Suggests

The papers point to a more specific answer than "tune PPO harder."

### Kickstarting Reinforcement Learning

Kickstarting shows that a student can surpass a teacher when:

- teacher guidance is strong early
- then decays
- while RL continues optimizing task return

Relevant lesson:

- static teacher loss is probably not enough
- we need a phase schedule, not just a fixed coefficient

Reference:

- `references/papers/1803.03835.pdf`

### Reincarnating Reinforcement Learning

Reincarnating RL studies how to improve from an existing teacher and emphasizes:

- rehearsal
- teacher data reuse
- structured transition from imitation to improvement

Relevant lesson:

- we should stop treating teacher replay as a side term
- it should be a first-class part of the learner

Reference:

- `references/papers/2206.01626.pdf`

### Demonstration-Regularized Reinforcement Learning

This line of work argues that when demonstrations are strong and online data is noisy, behavior regularization is not optional.

Relevant lesson:

- our problem looks much more like demonstration-regularized fine-tuning than plain on-policy RL

Reference:

- `references/papers/2310.17303.pdf`

### NetHack Offline Pretraining

The NetHack offline pretraining paper supports what we already observed:

- offline representation learning helps
- better representations improve sample efficiency

Relevant lesson:

- the world model was worth building
- but representation alone is not enough to solve the online improvement problem

Reference:

- `references/papers/2304.00046.pdf`

### Bootstrap-DAgger / student-state relabeling

DAgger-style results say:

- if the learner visits its own states, the teacher target must also cover those states

Relevant lesson:

- if we want improvement without drift, we probably need more student-state relabeling, not less

Reference:

- `references/papers/xudong24a.pdf`

## How The Literature Maps Onto The Current Code

This mapping is useful because it keeps the papers tied to real repo changes.

### Kickstarting maps to:

- scheduled teacher CE in [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- phase controls in [rl/config.py](/home/luc/rl-nethack/rl/config.py)

### Reincarnating RL maps to:

- stronger replay and rehearsal inside [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- possibly multi-source replay buffers instead of one static trace file

### Bootstrap-DAgger maps to:

- disagreement-focused extensions in [rl/dagger.py](/home/luc/rl-nethack/rl/dagger.py)
- disagreement/failure metadata in [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)

### Demonstration-regularized RL maps to:

- a future new trainer file rather than another patch to the current APPO branch

### NetHack offline pretraining maps to:

- keeping the world-model branch active as a teacher-building and feature-building path

## Literature Themes In Plain English

Across the papers, the recurring themes are:

### Theme 1: demonstrations should remain active during training

Not just at initialization.

### Theme 2: the teacher should be strong early, then partially released later

Not static forever, and not removed immediately.

### Theme 3: the learner should train on the states it actually visits

Not just the states the teacher happened to visit.

### Theme 4: once the teacher is strong, the bottleneck often becomes the improver, not the representation

That maps almost perfectly to the current repo state.

## What The Literature Does Not Suggest

The literature does **not** strongly support these as the next best move:

### 1. Bigger static APPO runs

Why not:

- we already know the current branch plateaus early
- more budget on a flat improver is not a strong bet

### 2. More generic intrinsic reward tuning

Why not:

- state-hash bonus variants did not solve the core teacher-drift problem

### 3. More representation churn before changing the improver

Why not:

- the world-model branch already raised the teacher floor
- the bottleneck moved

### 4. Trusting live reward as the next target

Why not:

- it has repeatedly diverged from the trusted alignment metric
- that is already empirically established in this repo

## Why APPO Is Still Worth One More Structural Attempt

Even though I am increasingly skeptical of static APPO as the final answer, one more structural APPO attempt is still justified because:

1. APPO is already integrated and debugged
2. replay scheduling is a smaller code delta than a whole new learner
3. if scheduled replay works, it gives us a cheaper path forward than a full algorithm pivot

But the key word is structural.

What is not justified anymore:

- more static-coefficient APPO runs as the main hope

## Strongest Current Hypothesis

The strongest current hypothesis is:

**we need to convert the online learner from "PPO with extra teacher losses" into "teacher-data-driven improvement with online refinement."**

That is subtle but crucial.

Under the old framing:

- teacher data is a helper
- RL is the main engine

Under the framing I now think is correct:

- teacher data is the main engine
- RL is a constrained refinement layer

That is much more consistent with:

- our current experimental evidence
- Reincarnating RL
- DQfD-style logic
- behavior-regularized fine-tuning work
- the difficulty profile of NetHack

## Second-Best Hypothesis

If the strongest hypothesis is wrong, the second-best hypothesis is:

- the current trace benchmark is too local, and the learner is finding improvements the benchmark cannot see

I do not think this is the most likely explanation.
But it should stay alive as a possibility until the repo has a better long-horizon live evaluation suite.

## The Missing Schedule Story

The current branch has no convincing narrative for *when* the learner should stop listening strongly to the teacher.

That is a real design gap.

There are really three distinct phases:

### Phase A: preserve

Goal:

- do not lose the teacher

Desired properties:

- strong replay
- strong teacher loss
- low reward pressure
- low update aggression

### Phase B: probe

Goal:

- test whether the learner can make tiny improvements without drifting

Desired properties:

- somewhat reduced replay
- same trace-gated checkpointing
- conservative optimizer settings

### Phase C: improve

Goal:

- actually beat the teacher

Desired properties:

- some reduction in teacher constraint
- only after the learner has already shown non-drifting behavior

Right now the repo is effectively trying to run all three phases at once with one static configuration.
That is very likely the central design mistake.

## The Missing Data Story

The schedule is not the only thing missing.

The current branch also lacks a good story for:

- which teacher rows matter most
- which student states should be relabeled next
- which action classes should be oversampled

Right now the replay buffer treats many rows as equally useful.

That is almost certainly false.

The learner likely needs:

- high-disagreement rows
- weak-action rows
- loop / failure rows
- recent student mistakes

more than it needs another thousand easy teacher-consistent rows.

The encouraging part is that the current codebase is already close to being able to support this.

The trace path in [rl/traces.py](/home/luc/rl-nethack/rl/traces.py) already has enough context to add:

- disagreement markers
- loop markers
- weak-action flags
- failure-oriented tags

So this is not a vague research wish.
It is a concrete repo modification that should be feasible in the next phase.

## The Missing Logging Story

Another code-level gap is logging.

Right now, the repo logs enough to tell us:

- losses moved
- checkpoints were saved
- trace match changed

But it does not yet log enough to fully explain:

- what replay distribution the learner actually saw
- how much of teacher replay came from weak actions
- how much came from disagreement states
- whether the value path and the policy path are drifting at the same time

This matters because better schedules without better observability will be hard to interpret.

The next phase should log at least:

- replay batch composition
- scheduled coefficient values over time
- disagreement action histogram over time
- maybe actor-vs-critic drift proxies if we can define them cheaply

One especially useful metric to add would be:

- teacher replay agreement on replay batches during training

because that would tell us whether the learner is remaining aligned to the replay source or merely surviving the extra loss numerically.

## Decision Tree

This is the clearest way to operationalize the next phase.

### Branch 1: can scheduled replay beat static replay?

Implement:

- scheduled replay coefficient
- prioritized replay batches
- disagreement-focused replay

Success criterion:

- a learned checkpoint beats the step-0 teacher clone on held-out trace match

If yes:

- keep APPO as the improver
- scale only after repeating the win

If no:

- stop spending most effort on APPO tuning
- move to Branch 2

### Branch 2: can a demo-regularized improver beat the teacher?

Implement:

- AWAC-like offline-to-online improvement
or
- DQfD-like actor-critic with demonstration replay as a core training source

Success criterion:

- beats teacher on held-out traces without collapsing in longer runs

If yes:

- this becomes the new mainline

If no:

- move to Branch 3

### Branch 3: stronger relabeling and curriculum

Implement:

- targeted DAgger on disagreement states
- weak-action oversampling
- horizon curriculum from `500 -> 1000 -> 5000`

Success criterion:

- offline teacher improves further and online improver stops plateauing instantly

This branch is supportive, not primary.

## A More Concrete Fork In The Road

If I compress the whole project into one decision, it is this:

### Option A: keep adapting APPO to become demonstration-centered enough

Pros:

- cheaper in the near term
- leverages existing infrastructure
- preserves continuity with current experiments

Cons:

- may keep us trapped in "preserve but do not improve"
- the actor-critic geometry may remain fundamentally mismatched

### Option B: build the next improver now

Pros:

- may fit the problem better
- may align the learner with the data geometry more directly

Cons:

- more code
- more new failure modes
- weaker causal continuity with the current branch

My current recommendation remains:

- do one more structural APPO attempt first
- but set a hard stop

That keeps the project disciplined.

The hard stop should be explicit:

- if scheduled replay plus prioritized replay still do not produce a learned checkpoint above the teacher clone, APPO should stop being the default online improver branch

## Timeline For The Next Phase

If I were planning the next phase concretely, I would think in three horizons.

### Horizon 1: next 1-2 coding sessions

Goal:

- implement scheduled replay and replay prioritization

Deliverables:

- new config fields
- new replay sampler
- new tests
- one short trace-gated run

### Horizon 2: next 3-6 coding sessions

Goal:

- find out whether scheduled replay APPO can actually beat the teacher

Deliverables:

- short and medium run results
- one clear go/no-go decision on APPO as the mainline improver

### Horizon 3: following week of work

Goal:

- if APPO still only ties the teacher, stand up the next improver class

Deliverables:

- AWAC-like or demo-actor-critic skeleton
- offline-to-online fine-tuning loop
- first trace-gated comparisons

## What We Should Stop Doing

To stay unstuck, we also need to stop doing some low-value work.

### Stop 1: blind long runs on static settings

If a branch does not beat the teacher in short and medium trace-gated runs, it should not get another large run.

### Stop 2: using reward improvement as a reason to trust a branch

That is no longer justified by the repo’s own evidence.

### Stop 3: generic DAgger with uniform state mixing

That path already showed weak value.

### Stop 4: world-model feature replacement experiments

Those already underperformed.
The useful branch is augmentation, not replacement.

## What We Should Keep Doing

There are also several things the repo should keep doing exactly because they worked.

### Keep 1: trace-gated checkpoint selection

This has been one of the highest-value additions in the whole project.

### Keep 2: dense checkpointing in unstable regimes

This already saved useful policies that later updates did not preserve.

### Keep 3: short and medium debug loops before large runs

The current project state proves this was the right discipline.

### Keep 4: weak-action slice analysis

This repeatedly helped identify where the learner was actually drifting.

### Keep 5: world-model augmentation as the current representation baseline

It is the strongest current representation path and should stay the baseline until something else clearly beats it.

### Keep 6: intentional use of short and medium runs as scientific filters

The current branch only reached its present level because the repo stopped over-trusting long runs.

That discipline should not be relaxed.

## What We Should Probably Remove Later

Not now, but eventually the repo should probably retire or demote:

- older unstable APPO presets that are no longer scientifically useful
- feature-replacement world-model modes that clearly underperformed augmentation
- generic DAgger workflows that are no longer the intended path

This matters because the repo is accumulating many branches, and clarity will matter more as the next algorithmic phase starts.

## Immediate Risks

The next phase has several clear risks.

### Risk 1: overfitting to the held-out trace set

Mitigation:

- rotate or expand held-out traces
- keep disagreement reports, not just scalar match rate

### Risk 2: scheduled replay makes the learner even more conservative

Mitigation:

- require evidence of a genuine teacher-beating short checkpoint before scaling

### Risk 3: spending too long on APPO because it is already built

Mitigation:

- set a clear stop condition for Branch 1

### Risk 4: confusing preservation improvements for true learning improvements

Mitigation:

- always compare:
  - step-0 teacher clone
  - best learned checkpoint
  - final checkpoint

### Risk 5: adding too many new ideas at once

Mitigation:

- change the schedule first
- change the replay distribution second
- only then change the learner class

This is important because the project is finally in a state where causal attribution is possible, and we should not throw that away.

## Stop Conditions

To stay disciplined, the next research phase should have explicit stop conditions.

### Stop condition for scheduled replay APPO

If after a bounded number of short and medium runs:

- no learned checkpoint beats the step-0 clone

then scheduled replay APPO should stop being the mainline.

I would make this explicit:

- if two or three structurally different scheduled replay branches still do not produce a teacher-beating learned checkpoint, APPO should stop being the main bet

### Stop condition for targeted DAgger

If disagreement-focused DAgger does not improve the offline teacher on held-out traces, it should remain auxiliary only.

### Stop condition for world-model expansion

If new world-model objectives do not improve the offline teacher or the online improver, they should not absorb most of the project’s attention.

## Concrete Metrics To Track Next

The next phase should not track only one scalar.

For every serious branch, track at least:

- step-0 trace match
- best learned trace match
- final trace match
- time/env steps until best checkpoint
- disagreement histogram by action
- invalid action rate
- repeated action rate
- loop / repeat-state proxy if available

These should become the standard scoreboard for the next few weeks of work.

If possible, add:

- replay-batch agreement
- replay source composition

because those will be especially informative once replay scheduling and prioritization land.

## Suggested Scoreboard Format

For future runs, the project would benefit from a standard scoreboard row like:

- experiment name
- teacher clone trace match
- best learned trace match
- final trace match
- env steps at best checkpoint
- weak-action recall summary
- invalid action rate
- repeat-action / loop proxy
- notes on replay schedule

That would make future comparison much easier than relying on prose memory.

## Concrete Repo Changes

### A. Add scheduled replay fields

In [rl/config.py](/home/luc/rl-nethack/rl/config.py):

- `teacher_replay_final_coef`
- `teacher_replay_warmup_env_steps`
- `teacher_replay_decay_env_steps`
- `teacher_replay_focus_mode`

In [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py):

- implement replay coefficient scheduling analogous to `_scheduled_teacher_coef`
- stratify replay minibatch sampling by metadata
- record replay schedule state in summaries so phase behavior is visible in logs

This is probably the single highest-value small code change after the current run finishes.

Additional useful fields:

- `teacher_replay_priority_power`
- `teacher_replay_disagreement_weight`
- `teacher_replay_weak_action_weight`
- `teacher_replay_failure_weight`

Potential additional field:

- `teacher_replay_source_mode`

where the source mode could later support:

- full teacher trace
- disagreement subset
- weak-action subset
- mixed source

Potential companion summaries:

- per-source replay contribution
- replay agreement
- replay weak-action fraction

### B. Extend trace rows with replay priorities

In [rl/traces.py](/home/luc/rl-nethack/rl/traces.py):

- add optional fields for:
  - disagreement flag
  - teacher action class
  - weak-action indicator
  - student failure / loop marker

Then teacher replay can sample smarter than uniform.

### C. Add a two-phase APPO preset

In [cli.py](/home/luc/rl-nethack/cli.py) and [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py):

- add a preset for:
  - Phase A: anchor
  - Phase B: improve

At minimum, this can be implemented as two sequential runs that resume from the best trace checkpoint.
That is likely easier and safer than trying to hot-swap schedules in one monolithic trainer at first.

### D. Make disagreement-focused DAgger easy

In [rl/dagger.py](/home/luc/rl-nethack/rl/dagger.py):

- add a mode that only aggregates:
  - mismatched states
  - weak-action states
  - high-repeat / loop states

This should be treated as:

- "relabel only the hard frontier"

not:

- "collect more data in general"

The current generic DAgger machinery should become a substrate for this more focused version, not a separate competing workflow.

It would be a mistake to fork the data-generation story into too many unrelated paths.

### E. Prepare a real behavior-regularized online branch

In new code, likely:

- `rl/train_awac.py`
or
- `rl/train_demo_actor_critic.py`

That should become the next algorithmic branch if scheduled replay APPO still only ties the teacher.

The most important thing is that this branch should be allowed to be opinionated.
It should not be forced to look like APPO just because APPO already exists.

This is one of the most important strategic points in the whole document.

## Immediate Experiment Plan

This is the concrete next sequence I would run.

### Experiment 1: scheduled replay on current APPO branch

Goal:

- test whether a phase schedule can turn preservation into improvement

Config idea:

- Phase A:
  - high replay
  - high teacher CE
  - very low reward scale
- Phase B:
  - lower replay
  - lower CE
  - same or lower learning rate

Gate:

- beats `0.9375` on held-out traces in a short or medium run

This should be treated as the last good-faith APPO structural attempt before escalating to a new improver class.

Interpretation:

- if this works, great
- if it does not, we have a principled reason to pivot

### Experiment 2: prioritized replay

Goal:

- see if replaying only disagreement / weak-action rows improves more than uniform replay

Gate:

- any learned checkpoint beats the teacher clone

### Experiment 3: disagreement-focused DAgger refresh

Goal:

- improve teacher support on student-induced hard states before another online run

Gate:

- offline teacher improves on the same held-out trace set

This is important because a stronger offline teacher is still useful even if the APPO branch is eventually replaced.

### Experiment 4: demo-regularized improver branch

Goal:

- replace "static PPO + side losses" with a learner whose data geometry matches the problem

Gate:

- stable teacher-preserving medium run
- then teacher-beating short checkpoint

The ordering here is intentional: stability first, then improvement.

### Experiment 5: objective introspection pass

Goal:

- understand whether policy drift or critic drift dominates once scheduled replay is added

This is not a full new branch.
It is a short instrumentation pass.

If replay agreement stays high while trace quality still falls, that would make the critic/value story much more suspect.

## Concrete Work Queue

If I were opening the repo tomorrow, I would do these in order:

1. finish and archive the current x10 static-replay run
2. add scheduled replay fields and tests
3. add replay-priority metadata to traces
4. implement prioritized replay sampling
5. run short and medium trace-gated scheduled-replay experiments
6. if none beat the teacher clone, stop and build the next improver branch

And one more rule:

7. if one does beat the teacher clone, repeat it before trusting it

That queue is intentionally short. The point is to create a crisp go/no-go decision, not another month of drift.

## Final View

We should be honest about both sides.

### Why this is encouraging

- the repo is much healthier than it used to be
- we have solved several real hard problems
- the current failure is scientifically meaningful
- the current teacher is strong enough that beating it is nontrivial but plausible

### Why this is frustrating

- the current branch no longer gives easy wins
- preserving a teacher is easier than improving one
- APPO-with-static-regularizers has probably reached its natural limit in the current form

### Why I still think WAGMI

Because the remaining problem is now narrow:

- not "why does nothing work?"
- but "what is the right online improver once the teacher and representation are strong?"

That is a much better problem to have.

## What Someone New To The Repo Should Believe

If a new engineer opened this repo tomorrow and asked, "what should I believe?", my answer would be:

1. the teacher is real
2. the trace benchmark is real
3. the world-model branch is worth keeping
4. the current APPO branch is no longer broken, but it is not enough
5. the next breakthrough will likely come from better use of teacher data during online learning

That is the simplest accurate worldview of the repo right now.

## What Someone New To The Repo Should Not Believe

A new engineer should not believe:

- that the current APPO branch is obviously "almost there"
- that a bigger model is the most likely next fix
- that reward gains imply real progress
- that world models already solved the core RL problem
- that more generic DAgger by itself is likely to save the branch

## One-Sentence Summary

The repo now knows how to build a strong teacher and preserve it briefly online, but it still does not know how to turn that teacher into a stronger policy, and the next work should focus on teacher-data scheduling, replay prioritization, and possibly a new improver class rather than more representation-side work.

## Decision

We are not done, and we are not actually stuck.

But we should stop thinking of the problem as:

"how do we make APPO better?"

and think of it as:

"how do we turn a strong offline teacher into a safe online improver?"

That is the correct problem now.

The answer is most likely:

- scheduled teacher replay
- targeted DAgger
- then a behavior-regularized improver if APPO still cannot beat the teacher

That is the branch I would push next.

## Closing Statement

The repo is now in the phase where the next big win should come from *reasoning correctly*, not from debugging heroics.

That is a good sign.

The system is finally coherent enough that:

- when it fails, the failure teaches us something
- when it succeeds, the success should mean something

That was not always true.

So the project is not stalled because nothing works.
It is stalled because we have reached the first genuinely strategic choice:

- whether to keep reshaping APPO around the teacher
- or move to a learner that is built around demonstration-regularized improvement from the start

That is exactly the kind of choice this document is meant to support.

## Last Word

The project no longer needs optimism by default.
It needs disciplined discrimination between:

- what is already working,
- what is merely stabilizing,
- and what is actually improving.

`wagmi.md` should be read in that spirit.

The strongest conclusion is still:

- the repo is healthy enough to support a real next move
- the next move should center teacher-data scheduling and prioritization
- and if that does not beat the teacher, the project should pivot learners rather than pretending APPO tuning is still the main frontier

## Experimental Ledger

This section exists to make the project history more concrete.

It is not a full command log. It is a distilled record of the runs that changed our beliefs.

### Ledger entry: early APPO-from-scratch

Observed behavior:

- learned policy collapsed into repetitive directional actions
- reward and trace quality were both poor

Belief update:

- pure APPO from scratch was not competitive enough to be the mainline

### Ledger entry: BC bridge

Observed behavior:

- BC from teacher traces clearly outperformed early APPO
- BC became the obvious warm-start backbone

Belief update:

- teacher-guided learning was not optional in this repo

### Ledger entry: deterministic trace eval

Observed behavior:

- repeated live evaluations on the same seeds disagreed
- fixed-trace evaluation was stable and reproducible

Belief update:

- live seeded evaluation could no longer be treated as the source of truth

### Ledger entry: warm-start bridge fixes

Observed behavior:

- after matching width, activation, and normalization, step-0 APPO matched the teacher

Belief update:

- the initialization bridge was no longer the dominant problem

### Ledger entry: targeted teacher-action boosts

Observed behavior:

- action-specific weighting improved certain weak actions
- but still did not create a teacher-beating branch

Belief update:

- the learner was sensitive to targeted constraints
- but scalar action weighting alone was not the breakthrough

### Ledger entry: `v4 + wm_concat_aux`

Observed behavior:

- offline teacher quality improved materially
- weak-direction slices improved

Belief update:

- representation still mattered
- but the online learner still lagged behind the improved teacher

### Ledger entry: conservative 500-step APPO with teacher replay

Observed behavior:

- catastrophic collapse became much less common
- some runs stayed close to the teacher for meaningful stretches
- best checkpoint still occurred early

Belief update:

- teacher replay was directionally right
- static teacher replay was not enough to produce improvement

### Ledger entry: x10 large run on current best static branch

Observed behavior:

- training remained operationally stable
- best trace checkpoint still sat at `352` env steps with `0.9375`
- later checkpoints did not improve on it

Belief update:

- the branch was mature enough to trust
- the branch still plateaued
- the next change must be structural, not just another run of the same config

## Why The Current Plateau Is Informative

The current plateau is frustrating, but it is scientifically useful.

If the current branch had still been collapsing violently, we would not know whether the next step should be:

- more debugging
- more representation work
- or a new learner

But the plateau tells us something more specific:

- the learner can now hold onto competence
- it cannot yet translate that competence into safe incremental gains

That is a much cleaner failure mode than the repo had earlier.

This means future work should be graded against the current plateau, not against the old collapse baseline.

## Interpretation Of The Early-Best Pattern

The most striking pattern in the current best branch is:

- best checkpoint occurs extremely early
- later checkpoints do not surpass it

There are several possible interpretations of that pattern.

### Interpretation A: the first few RL updates are genuinely useful, then the policy overfits to the wrong objective

This is plausible.

If true, it implies:

- the first part of the online signal is valuable
- but the learner stays in the "unsafe improvement" regime too long

This interpretation supports:

- phase schedules
- earlier stopping
- checkpoint-gated continuation

### Interpretation B: the learner is not really improving at all, only preserving the step-0 teacher clone for a while

This is also plausible.

If true, it implies:

- the current branch is not yet an improver in a meaningful sense
- the online updates are mostly destructive after a brief preservation period

This interpretation supports:

- a new learner class
- demonstration-centered updates

### Interpretation C: the trace benchmark is too narrow to see a real improvement

This is possible, but currently weak as the main explanation.

If true, it would imply:

- the learner found a better policy that the trace benchmark cannot reward

Why I do not currently treat this as primary:

- the repo has repeatedly shown that reward and live metrics are noisy and misleading
- the trace metric has been the most stable and useful instrument we have

So while Interpretation C could contain some truth, it is not strong enough to override A and B.

## Where APPO Still Helps

Even if APPO stops being the mainline improver, it still has value in this repo.

### APPO as a stress test

It is useful for asking:

- does this teacher survive online updates at all?
- does this representation remain stable under actor-critic training?

That is already useful information.

### APPO as a continuation engine

If we eventually build a stronger demo-regularized improver, APPO or APPO-like training may still be useful for:

- final policy refinement
- online adaptation
- stress-testing the policy outside the pure teacher distribution

### APPO as a baseline

The current APPO branch is now good enough to function as a real baseline:

- it is not a toy anymore
- it is not obviously broken anymore

That makes it scientifically useful even if it is not the final answer.

## Where The World Model Still Helps

The world model is not the next main bottleneck, but it is still important.

### 1. Representation learning

This is already the clearest success case.

### 2. Auxiliary prediction

The world model could still help the online branch if it is used as:

- an auxiliary latent prediction loss
- an action-logit prediction aid
- a short-horizon future summary target

That has not been fully exploited yet.

### 3. Better teacher building

Even if the online learner changes completely, the world-model path may still remain the best way to build stronger offline teachers.

### 4. Candidate planning signal

Later, the world model might help with:

- short-horizon action ranking
- option / skill outcome prediction

But that should come after the improver problem is solved, not before.

## Open Questions

These are the questions I think remain genuinely open.

### Question 1: is the main bottleneck actor drift or critic drift?

The current implementation constrains the policy more directly than the value path.

If the critic is the main source of bad updates, then:

- more policy replay alone will not be enough

This is a major reason to consider demo-aware actor-critic variants.

### Question 2: how much of the remaining gap is due to replay sampling quality?

Uniform teacher replay is almost certainly suboptimal.

We do not yet know how much improvement we can get from:

- disagreement-only replay
- weak-action replay
- loop-state replay

### Question 3: is the next win likely to come from schedule or from learner class?

This is the central fork.

My current answer is:

- schedule first, because it is cheaper
- learner-class change next, if schedule still only ties the teacher

### Question 4: do we need a demo-aware value target?

This is one of the most important open design questions.

### Question 5: how far can the current trace benchmark take us?

It is the right benchmark today, but not necessarily the forever benchmark.

Eventually, the repo will need:

- stronger long-horizon evaluation
- broader held-out trace sets
- maybe task-specific success suites

## Operational Rules For The Next Phase

The next phase should obey a few explicit rules.

### Rule 1: no large run without a short-run learned checkpoint beating the teacher clone

This is the biggest time-saving rule.

### Rule 2: every branch must report three checkpoints

Always compare:

- step-0 teacher clone
- best learned checkpoint
- final checkpoint

Without that, it is too easy to confuse preservation with improvement.

### Rule 3: every replay experiment should expose its actual training distribution

That means logging:

- proportion of weak-action rows
- proportion of disagreement rows
- proportion of ordinary teacher rows

### Rule 4: world-model changes should be judged first by offline teacher quality

If a world-model change does not improve or preserve the offline teacher, it should not move into the online branch.

### Rule 5: schedule experiments should be treated as structural experiments, not scalar sweeps

The point is not to try a hundred coefficients.
The point is to change the shape of learning over time.

## Success Criteria For The Next Milestone

The next milestone should not be "run a bigger job."

It should be:

### Minimum success

- a learned checkpoint beats `0.9375` on held-out trace match

### Better success

- the best learned checkpoint occurs meaningfully after the first few hundred steps

### Strong success

- a medium run beats the teacher and does not immediately decay below it

Only after that should another large-scale run become the main focus.

## Repo Modification Plan In Plain Language

If I had to explain the next phase to another engineer in one page, I would say:

1. keep the current `v4 + wm_concat_aux` representation
2. keep the current deterministic trace evaluation setup
3. stop expecting static APPO + static replay to beat the teacher
4. add a replay schedule with explicit phases
5. prioritize replay rows where the student is currently weakest
6. if that still only preserves, build a demo-regularized improver

That is the shortest accurate description of what the repo needs now.

## If We Are Wrong

It is also worth asking what would falsify the current thesis.

The current thesis would be weakened if:

- a simple larger model immediately beats the teacher without schedule changes
- a static replay branch suddenly produces delayed improvements in repeated runs
- a stronger live evaluation suite contradicts the trace benchmark in a convincing and repeatable way

If any of those happen, this document should be revised aggressively.

Right now, none of them have happened.

## Final Bottom Line

The repo is not stuck in the old sense.

It is stuck in a new, much better sense:

- the base systems mostly work
- the teacher is strong
- the representation is strong enough
- the trusted metric is real
- and now the only thing left is to find the right constrained online improver

That is hard, but it is exactly the kind of problem where disciplined iteration should pay off.

So the current recommendation stands:

- finish the current static-replay branch cleanly
- move next to scheduled replay and prioritized replay
- then, if that still ties the teacher, stop forcing APPO to be the hero and build the next learner class

That is how we get unstuck.
