# Project Status And Next Steps

## Executive Summary

The project is now past the "does the RL harness work at all?" stage.

That question has been answered `yes`.

The current problem is narrower and more useful:

- the offline teacher pipeline is now strong,
- the warm-start bridge from offline teacher to APPO is now correct,
- deterministic trace evaluation is trustworthy enough to gate experiments,
- but online RL still drifts away from teacher-aligned behavior after initialization.

This is no longer primarily an infrastructure problem. It is an **objective alignment and online improvement** problem.

The repo should now be shaped around that fact.

## Current State Of The Codebase

The project currently has four meaningful learning/control layers:

1. **Teacher trace generation and debugging**
   - multi-turn trace generation and verification in [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
   - deterministic trace evaluation and disagreement analysis in [rl/trace_eval.py](/home/luc/rl-nethack/rl/trace_eval.py)
   - fast debug commands wired in [cli.py](/home/luc/rl-nethack/cli.py)

2. **Offline policy training**
   - BC in [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py)
   - behavior-regularized training in [rl/train_behavior_reg.py](/home/luc/rl-nethack/rl/train_behavior_reg.py)

3. **Online RL**
   - Sample Factory APPO backend in [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
   - env adapter and action masking in [rl/env_adapter.py](/home/luc/rl-nethack/rl/env_adapter.py) and [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)
   - teacher-regularized APPO loss in [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)

4. **Representation work**
   - feature encoders in [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)
   - current best representation path is `v4`

The repo also now has supporting infrastructure that matters a lot for reliable iteration:

- deterministic trace ranking for checkpoints
- best-by-trace checkpoint aliasing
- atomic writes / experiment locking
- short debug loops instead of blind long APPO runs

## What We Learned

### 1. Deterministic trace evaluation changed the project

The main scientific unlock was not a bigger model. It was getting a trusted metric.

Before deterministic trace evaluation, live seeded eval and reward numbers were too noisy to tell whether a change was actually good.

Now the trusted metric is:

- **trace match on a fixed held-out teacher trace set**

That metric is not perfect, but it is good enough to gate changes and catch regressions quickly.

This lines up with "Deep RL That Matters": if the evaluation protocol is weak, apparent gains are often not real.  
Paper: [Deep RL That Matters](https://arxiv.org/abs/1709.06560)

### 2. The offline teacher became good enough to matter

The `v4` representation plus behavior-regularized offline training substantially improved the teacher policy.

The relevant result is:

- full `v4` behavior-regularized teacher reached about `0.95` held-out trace match

That is the first point where the repo had a teacher strong enough that online RL improvement over it became a meaningful question.

### 3. The warm-start bridge was broken, then fixed

Earlier APPO warm starts were weaker than they should have been even at step `0`.

The cause was a three-way mismatch between the offline teacher and the APPO actor:

- hidden width mismatch
- input normalization mismatch
- nonlinearity mismatch

After fixing those in [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py) and [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py), step-0 APPO finally matched the `v4` teacher exactly.

That was a critical result, because it showed that the remaining regression after training is caused by online learning, not by corrupted initialization.

### 4. Generic bonuses are not the main bottleneck

The episodic exploration bonus path is implemented, but the experiments so far suggest it is not the main lever.

The current problem is not that the policy needs more generic novelty pressure. The current problem is that online optimization moves the policy away from the teacher manifold.

### 5. Broad regularization helped less than targeted regularization

Two important short-loop outcomes:

- conservative PPO settings helped preserve the teacher much better than the old online path
- broad parameter anchoring was not good enough and could make things worse
- targeted teacher-action weighting was better than broad anchoring

The most useful short-loop result from the latest work was:

- plain conservative APPO first learned checkpoint: `0.9125`
- conservative APPO + `west=2.0` teacher-action boost: `0.925`

That is a real improvement, and the disagreement report showed why:

- the original main drift was `teacher west -> model east`
- the targeted weighting removed that failure mode

That means the teacher-loss shaping mechanism is doing something real and useful.

## What The Literature Says

The current repo state matches a fairly specific literature regime:

- **strong teacher / demonstrations available**
- **sparse, long-horizon task**
- **on-policy RL drifts away from demonstrated behavior**

The most relevant papers support the same conclusion:

### DAgger

DAgger says one-shot imitation is not enough because the learner visits its own states and accumulates errors.  
Paper: [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html)

Implication for this repo:

- DAgger should stay in the codebase,
- but only as a controlled, held-out-gated loop,
- not as blind data accretion.

### Kickstarting

Kickstarting says the teacher signal should stay active during RL, not only at initialization.  
Paper: [Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835)

Implication for this repo:

- teacher-regularized APPO should remain the default online RL path,
- not plain APPO from a BC checkpoint.

### DQfD and related demo-in-the-loop methods

DQfD argues that demonstrations should remain part of the learning signal instead of being used only for pretraining.  
Paper: [Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732)

Implication:

- the project should keep teacher supervision active during online improvement,
- and should likely make more use of demo-aware update rules rather than standard on-policy improvement alone.

### AWAC / BRAC / behavior-regularized RL

These papers argue that when you have a behavior distribution you trust, policy improvement should be constrained around it rather than allowed to drift arbitrarily.

- [AWAC](https://arxiv.org/abs/2006.09359)
- [BRAC](https://arxiv.org/abs/1911.11361)

Implication:

- if teacher-reg APPO continues to plateau, the next algorithmic branch should not be "more PPO tuning"
- it should be a more explicitly behavior-regularized online improvement method

### Demonstration-Regularized RL

More recent work continues to support the same general pattern: for hard tasks with sparse or misleading rewards, demonstrations act as a regularizer, not just a bootstrap.

Locally pulled reference:

- [demo_reg_rl_arxiv2023.pdf](/home/luc/rl-nethack/references/papers/demo_reg_rl_arxiv2023.pdf)

### Relay / staged improvement

Relay-style methods reinforce the broader lesson that difficult long-horizon tasks often need staged improvement rather than a single flat optimizer that is expected to handle the entire task distribution at once.

Locally pulled reference:

- [relay_policy_learning_corl2019.pdf](/home/luc/rl-nethack/references/papers/relay_policy_learning_corl2019.pdf)

## Current Diagnosis

The most accurate current diagnosis is:

> The offline policy is strong, the initialization path is fixed, but the online objective still perturbs the policy away from the teacher faster than it improves behavior on the trusted held-out trace objective.

More concretely:

- the best teacher is around `0.95`
- step-0 APPO can now preserve `0.95`
- the first learned checkpoint still falls below that
- conservative PPO settings reduce the fall
- targeted teacher-action weighting reduces it further
- but the policy is still not improving beyond the teacher, only degrading more slowly

That means the next bottleneck is not "how do we optimize more?"

It is:

- how do we do **teacher-constrained online improvement** without immediately sacrificing trace alignment?

## What We Should Push Toward Next

### 1. Make the fast loop the primary workflow

The repo should treat this as the canonical loop:

1. choose a fixed held-out trace set
2. run a short offline or short online experiment
3. rank by deterministic trace match
4. inspect disagreement report
5. only scale runs that win on the short benchmark

This is already partly true in practice. It should become the explicit project norm.

### 2. Continue targeted teacher-loss shaping

The latest useful result came from targeted action weighting, not generic reward changes.

That strongly suggests the next work should stay in that family:

- tune targeted teacher-action boosts
- add focused reports for action-class drift
- prefer fixing identified confusion pairs over adding broad regularization

Immediate candidates:

- `west=1.5`
- `west=2.0,east=1.1`
- `west=2.0` with a slightly different teacher coefficient or schedule

### 3. Add teacher-loss schedules, not only static coefficients

Right now the teacher loss is static.

A better path may be:

- strong teacher preservation at the start
- very slow decay
- checkpoint gating by trace match throughout

This is closer to the spirit of Kickstarting than a constant weak coefficient.

### 4. Keep behavior-regularized offline training as the teacher factory

The offline side is currently the strongest part of the project.

That means:

- keep improving the offline teacher
- do not regress to weaker BC baselines
- make the offline teacher the canonical initializer and online teacher source

### 5. Treat APPO as the current online baseline, not the final method

The repo should keep APPO because:

- it works,
- it is integrated,
- it provides a real online control benchmark.

But the literature and the current results both say it may not be the final best algorithm for this regime.

If the next rounds of teacher-shaped APPO still plateau, the project should branch toward:

- behavior-regularized online improvement
- demo replay in the learner loop
- or an AWAC/BRAC-style improvement path

### 6. Keep DAgger experimental and gated

DAgger is still conceptually correct, but in this repo the naive schedule currently hurts more than it helps.

So DAgger should stay:

- implemented,
- cheap to run,
- held-out-gated,

but not promoted as the mainline until a schedule actually wins the short trusted benchmark.

## Repo Shape Recommendation

The project should now be structured mentally as:

`teacher traces -> offline teacher -> exact warm-start bridge -> conservative teacher-shaped online improvement -> trace-gated model selection`

Not:

`reward shaping -> long APPO runs -> hope`

That is the main conceptual change.

## Concrete TODO

1. Add teacher-loss schedules to [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py) and [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py).
2. Add first-class action-drift reporting to the training loop so checkpoint selection can see which action classes are moving.
3. Keep tuning targeted teacher-action boosts with the short trace loop.
4. Promote the best short-loop teacher-shaped APPO config into the default online baseline.
5. If that still stalls, add a separate behavior-regularized online branch rather than tuning PPO indefinitely.

## Bottom Line

The project is in a much better state than it was.

The main value now is not that "RL works." It is that the repo can now answer a more precise question quickly:

> does this online update preserve or improve teacher-aligned behavior on trusted traces?

That is the right question for the current stage of the project.

The next wins will come from making online improvement more constrained, more teacher-aware, and more selectively targeted, not from scaling reward optimization by itself.
