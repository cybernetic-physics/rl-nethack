# RL Postmortem And Next Steps (2026-04-05)

This document reflects on the latest large APPO rerun and uses relevant papers
to decide what the repo should do next.

It is grounded in:

- [RUN-REPORT-2026-04-05-X10-RERUN.md](/home/luc/rl-nethack/RUN-REPORT-2026-04-05-X10-RERUN.md)
- [FAST-DEBUG-LOOP-REPORT-2026-04-05.md](/home/luc/rl-nethack/FAST-DEBUG-LOOP-REPORT-2026-04-05.md)
- current RL code in:
  - [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
  - [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)
  - [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)
  - [rl/trace_eval.py](/home/luc/rl-nethack/rl/trace_eval.py)
  - [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
  - [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)

It is also informed by the following papers now saved under
[references/papers](/home/luc/rl-nethack/references/papers):

- `dagger_aistats2011.pdf`
- `kickstarting_arxiv2018.pdf`
- `skillhack_arxiv2022.pdf`
- `maestromotif_iclr2025.pdf`
- `brac_arxiv2019.pdf`
- `dqfd_arxiv2017.pdf`
- `scheduled_sampling_nips2015.pdf`
- `deep_rl_that_matters_arxiv2017.pdf`

## 1. What Worked

The repo is in a much better scientific and engineering state than before.

### 1.1 The fast debug loop works

The recent tooling additions were valuable:

- deterministic trace evaluation in
  [rl/trace_eval.py](/home/luc/rl-nethack/rl/trace_eval.py)
- trace sharding and verification in
  [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
- richer disagreement reporting via
  [cli.py](/home/luc/rl-nethack/cli.py)

This mattered because the x10 rerun looked good under:

- training reward
- checkpoint logs
- live diagnostic evaluation

but looked bad under the trusted deterministic trace benchmark.

Without the new fast loop, that regression would have been easy to miss.

### 1.2 BC remains a useful teacher / bootstrap

The BC policy is still stronger than the APPO student on the trusted trace set:

- BC trace match rate: `0.6395`
- latest APPO rerun checkpoint: `0.4651`
- best-by-training-reward APPO checkpoint: `0.3837`

That means BC is not a dead end. It is the current best aligned policy object
in the repo.

### 1.3 The harness is operationally stable

The latest run did not fail operationally:

- APPO launched cleanly
- BC warm start loaded cleanly
- checkpoints were written and reloaded
- deterministic trace eval worked on both latest and best checkpoints

So the main problem is no longer “the system does not run.”

The main problem is now “the system optimizes the wrong thing.”

## 2. What Did Not Work

### 2.1 More APPO scale did not improve the trusted target

The latest rerun trained stably for `230,400` frames and reached a positive
training-reward regime, but deterministic teacher-state behavior got worse.

This is the most important result from the run.

The current APPO setup is not failing because it is unstable.
It is failing because it is drifting.

### 2.2 Best training reward is not best policy behavior

The saved “best” checkpoint by training reward was actually worse than the latest
checkpoint on the trusted trace benchmark.

This means:

- the current checkpoint selection criterion is not aligned with the real target
- the RL objective is not aligned with teacher agreement

That is the strongest current diagnosis in the repo.

### 2.3 The failure mode is now very specific

The directional confusion reports show the problem is not generic collapse.

For the rerun, APPO over-optimized toward `west`, with:

- high `west` recall
- almost no `east` or `south`

That is more specific than “the model is weak.”
It suggests that the policy is exploiting a shallow movement bias under the
current reward/observation setup.

### 2.4 Live eval is still not trustworthy enough for regression

The repo-level conclusion remains:

- live seeded eval is useful for diagnostics only
- deterministic trace eval is the trusted source of truth

This is consistent with the literature on deep RL evaluation instability.

## 3. What The Papers Say

### 3.1 DAgger: one-shot imitation is not enough

Ross et al. (2011) argue that imitation in sequential settings fails because the
learner induces a different state distribution than the expert, leading to
compounding errors under its own rollouts.

That is directly relevant here.

This repo currently does:

- teacher trace generation
- BC on teacher states
- APPO fine-tuning from that BC start

But it does not yet do iterative dataset aggregation from student-induced states.

That means the repo is still exposed to the exact distribution-shift problem that
DAgger was designed to fix.

Implication:

- the next imitation-stage upgrade should be DAgger-style trace aggregation
- not another blind APPO sweep

### 3.2 Kickstarting: teacher-guided RL is right, but not unconstrained drift

Kickstarting shows that student-on-policy trajectories plus a teacher-matching
auxiliary loss can dramatically improve data efficiency and let the student
surpass the teacher.

That supports the repo’s BC-to-APPO direction in principle.

But the paper’s lesson is not “pretrain once, then let RL run free forever.”
The lesson is:

- keep teacher supervision active while the student learns
- let the influence decay carefully
- use the teacher on student trajectories

Right now the repo warm-starts APPO from BC weights, but it does not keep a
teacher KL / imitation constraint active inside the APPO loss.

Implication:

- add teacher-regularized APPO, not just BC initialization

### 3.3 SkillHack / Hierarchical Kickstarting: long-horizon sparse-reward tasks need stronger inductive bias

SkillHack’s result is that pre-defined skills and teacher transfer provide a very
useful inductive bias in NetHack-like tasks with large state-action spaces and
sparse rewards.

That is consistent with what we are seeing:

- generic end-to-end RL is not cleanly improving the current policy
- teacher-guided behavior remains the best signal

Implication:

- keep the skill/task structure
- shrink the effective action space further for `explore`
- keep skill-aware transfer, not flat unguided RL

### 3.4 MaestroMotif: use language/AI for rewards and skills, but RL still needs the right objective

MaestroMotif supports:

- skill decomposition
- reward modeling from AI feedback
- planning/composition over skills

But it does not say “skip careful low-level control design.”

In our repo, the lesson is:

- continue toward explicit skills/options and learned reward models
- but do not expect a weak low-level APPO loop to fix itself just by scaling

### 3.5 BRAC: constrain policy drift toward the behavior policy

BRAC’s main offline RL lesson is that constraining the learned policy toward the
behavior distribution matters a lot, and that simple behavior-regularized
baselines can outperform more complicated methods.

Our current result looks exactly like a behavior-drift failure:

- the student starts from a teacher-aligned BC policy
- then RL improves its own reward objective
- then it drifts away from the teacher distribution and becomes worse on the
  trusted regression set

Implication:

- add explicit behavior regularization to APPO fine-tuning
- use BC as the behavior policy anchor

This does not mean “do offline RL.”
It means the same regularization idea is likely useful here.

### 3.6 DQfD: demonstrations help most when they remain part of the learning signal

DQfD’s core idea is that combining TD updates with demonstration supervision is
more effective than simply starting from demonstrations and then discarding them.

This maps almost directly to the repo’s current gap.

Right now the repo does:

- BC
- then APPO

What it should test next is:

- BC / teacher supervision continuing during RL

### 3.7 Scheduled Sampling: training-time distribution should move toward inference-time distribution

Scheduled Sampling is about sequence models, not RL, but the lesson is still
useful:

- training on clean teacher prefixes only creates a train/test mismatch
- the learner must be exposed to its own induced states

This again points toward:

- DAgger-style aggregation
- student-on-policy relabeling
- not one-shot expert-only traces

### 3.8 Deep RL That Matters: use robust evaluation or you will optimize noise

This paper is the meta-level justification for the fast loop.

The repo’s recent history is a direct example:

- live eval and training reward looked better
- deterministic trace eval said the actual target got worse

That is exactly why the trace benchmark must remain the gating metric.

## 4. Best Current Diagnosis

The current APPO system is overfitting to a shallow shaped-reward strategy that
does not preserve teacher-like directional control.

More specifically:

1. The policy starts from a decent BC policy.
2. APPO finds a local strategy that improves its own reward.
3. That strategy collapses directional diversity.
4. Trace regression gets worse.

This is most consistent with:

- behavior drift away from the teacher
- insufficient teacher-on-student supervision
- underconstrained action selection for exploration
- incomplete directional state representation

## 5. What We Should Do Next

This is the recommended order.

### Step 1: Stop using plain APPO fine-tuning as the main next experiment

The latest rerun is enough evidence.

More of the same will likely waste GPU time.

### Step 2: Add teacher-regularized APPO

Concretely:

- keep BC warm start
- during APPO training, add an auxiliary loss that keeps the policy close to the
  BC teacher on student trajectories
- make the weight scheduled, not fixed forever

This is the most direct lesson from Kickstarting and DQfD.

Suggested implementation location:

- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/model.py](/home/luc/rl-nethack/rl/model.py)
- possibly a new module such as:
  - `rl/teacher_regularization.py`

### Step 3: Add DAgger-style trace aggregation

Concretely:

1. roll out current BC or APPO student
2. collect student-visited states
3. relabel them with `task_greedy`
4. merge with existing trace set
5. retrain BC
6. then fine-tune RL

This is the strongest paper-backed next move.

Suggested implementation:

- new script:
  - `rl/run_dagger_iteration.py`
- or CLI commands:
  - `rl-generate-student-traces`
  - `rl-relabel-traces-with-teacher`
  - `rl-train-bc --append ...`

### Step 4: Change checkpoint selection to include the trusted trace benchmark

Current “best checkpoint” uses training reward.
That is not good enough.

The repo should start saving:

- best-by-training-reward
- best-by-trace-match-rate

This will prevent misleading conclusions from long runs.

### Step 5: Improve directional observation features before another RL sweep

The current `v2` encoder still appears too weak for symmetric movement choices.

The next feature work should target:

- local geometry around the player beyond immediate adjacency
- stronger frontier encoding
- short directional history
- whether the last few moves changed the local view in a meaningful way

This should be tested with BC first, not APPO first.

### Step 6: Replace the `fork()`-based teacher counterfactual path

This is not the main scientific bottleneck, but it is now the clearest technical
smell in the harness.

The `os.fork()` path in
[src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py)
still emits warnings and is a brittle dependency for the teacher.

Even if it is not causing the current regression, it should be removed before the
teacher becomes more central via DAgger.

## 6. Concrete 1-Week Plan

### Priority A

1. teacher-regularized APPO
2. best-by-trace checkpoint selection
3. DAgger-lite one-iteration loop

### Priority B

4. directional encoder upgrade
5. BC regression on focused `east/south` shards

### Priority C

6. replace `fork()` teacher path

## 7. Bottom Line

The repo should not respond to the latest rerun by doing a larger APPO sweep.

The papers and the run both point to the same conclusion:

- the next gains will come from keeping the student tied to the teacher while it
  explores its own state distribution
- not from more unconstrained APPO scale

So the next correct direction is:

- BC teacher anchor
- teacher-regularized RL
- DAgger-style aggregation
- stronger directional features

That is the highest-confidence path to improving the project from its current
state.
