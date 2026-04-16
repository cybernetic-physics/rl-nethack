# Two Candidate Directions After APPO Saturation

This memo compares the two most plausible next algorithmic directions for this repo:

1. behavior-regularized offline-to-online RL
2. teacher-replay / on-policy distillation learners

The context is the current codebase state:

- the best trusted result is still the short scheduled-replay APPO checkpoint at `0.95` trace match
- medium and large APPO runs drift
- world-model-augmented `v4 + wm_concat_aux` features help representation, but they do not solve online drift
- repeated schedule tuning, action reweighting, and staged APPO continuation did not beat the teacher

So the question is no longer “how do we tune APPO better?” It is “what training paradigm is best matched to a strong teacher, a trusted trace benchmark, and a weak/misaligned online reward?”

## Executive Summary

Both directions are reasonable, but they solve different failure modes.

**Teacher-replay / on-policy distillation**
- best if the main problem is early on-policy drift away from a strong teacher
- easiest extension of the current repo
- strongest continuity with the current code
- already partially implemented here
- likely best *incremental* next step

**Behavior-regularized offline-to-online RL**
- best if the main problem is that on-policy APPO itself is the wrong improver
- stronger strategic bet if we believe the teacher and offline data are already the real asset
- harder implementation change
- better aligned with the repo’s current evidence that online PPO-style updates are the bottleneck

My current recommendation:

- short-term engineering move: finish one last serious teacher-replay / on-policy distillation branch with better structure than current APPO schedule tuning
- strategic bet: begin a behavior-regularized offline-to-online improver branch in parallel, because the evidence increasingly suggests this is the stronger long-term fit

## Option 1: Behavior-Regularized Offline-to-Online RL

### Core idea

Train from offline data first, then improve online while **explicitly constraining the policy to stay near the behavior distribution** when uncertainty is high. The main goal is to avoid destructive policy drift during the transition from offline to online learning.

This is a direct fit for our repo because:

- we have a strong offline teacher
- we have a trusted trace dataset
- the online learner repeatedly degrades a good policy

### Why this family exists

Offline-to-online RL papers keep returning to the same problem: a good offline policy becomes brittle when online updates are too aggressive or when the policy leaves the offline data manifold too quickly.

Recent papers and relevant takeaways:

- **Behavior-Adaptive Q-Learning: A Unifying Framework for Offline-to-Online RL (2025)**  
  https://arxiv.org/abs/2511.03695  
  Key idea: use an implicit behavior model from offline data to keep the online policy behavior-consistent when uncertainty is high, then relax the constraint as trustworthy online data accumulates.

- **From Static to Dynamic: Enhancing Offline-to-Online RL via Energy-Guided Diffusion Stratification (2025)**  
  https://arxiv.org/abs/2511.03828  
  Key idea: separate samples into offline-like and online-like regions and apply different update rules instead of pretending the whole buffer should be optimized the same way.

- **Robust Policy Expansion for Offline-to-Online RL under Diverse Data Corruption (2025)**  
  https://arxiv.org/abs/2509.24748  
  Key idea: online expansion from offline data is fragile; careful regularization and exploration constraints improve robustness.

- **Offline-to-Online Reinforcement Learning with Classifier-Free Diffusion Generation (2025)**  
  https://arxiv.org/abs/2508.06806  
  Key idea: augment the online learner with generated offline-like samples that keep training near the useful support of the original data distribution.

Relevant foundational references:

- **AWAC**  
  https://arxiv.org/abs/2006.09359  
  Advantage-weighted actor updates from offline data plus online fine-tuning.

- **BRAC**  
  https://arxiv.org/abs/1911.11361  
  Explicit behavior regularization to prevent leaving the behavior manifold.

### What this means for our repo

This family matches our current failure mode extremely well.

Current evidence from the repo:

- offline teachers are strong
- APPO can warm start from them
- online updates move away from teacher-aligned behavior
- the teacher metric and the shaped reward are not the same thing

Behavior-regularized O2O methods are designed exactly for:

- “good offline data, weak online objective”
- “don’t blow up the policy during online fine-tuning”
- “let the learner improve while uncertainty is still high”

### How it would look in this repo

Instead of:

- BC/teacher -> APPO on-policy updates with auxiliary teacher losses

we would use something more like:

- offline teacher / behavior policy
- behavior-constrained critic / actor updates
- online fine-tuning that becomes less conservative only when supported by newer online data

Concretely, that likely means:

- replay buffer becomes first-class, not auxiliary
- trace / teacher data are not just regularization batches; they define the behavior prior
- policy improvement depends on uncertainty or offline-likeness

### Strengths

- most directly targets our observed failure
- likely better fit than APPO if the problem is truly on-policy drift
- makes stronger use of the dataset and teacher we already have
- aligns with the world-model feature path, because offline-to-online methods naturally benefit from strong latent features

### Weaknesses

- harder to implement than incremental APPO changes
- larger codebase change
- more likely to require a new learner path rather than patching Sample Factory hooks
- may slow experimentation in the short term

### Overall assessment

This is the strongest **strategic** direction if we accept that APPO itself is probably the wrong improver.

If I had to pick one direction with the highest probability of eventually beating the teacher in this repo, this is currently my top choice.

## Option 2: Teacher Replay / On-Policy Distillation

### Core idea

Keep the student aligned to the teacher on the **student’s own trajectories**, not just on the original teacher dataset. Use replay, dense distillation, or KL-style regularization on-policy so the learner does not immediately drift after warm start.

This is the more natural extension of what we already built.

### Why this family exists

This family is built around a simple observation: one-shot imitation is not enough, and pure RL is too weak or too misaligned. The student needs the teacher to remain active during online learning.

Recent papers and relevant takeaways:

- **Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation (2026)**  
  https://arxiv.org/abs/2602.12125  
  Key idea: on-policy distillation can be seen as KL-constrained RL; by changing the relative weight between reward and distillation, the student can exceed a teacher boundary rather than merely copy it.

- **Continual Policy Distillation from Distributed Reinforcement Learning Teachers (2026)**  
  https://arxiv.org/abs/2601.22475  
  Key idea: decouple “teachers solve tasks with RL” from “student learns by stable replay-based distillation”; replay is central for stability.

Relevant foundational references:

- **Kickstarting Deep Reinforcement Learning**  
  https://arxiv.org/abs/1803.03835  
  Teacher-guided RL where distillation remains active during training.

- **Reincarnating Reinforcement Learning**  
  https://papers.neurips.cc/paper_files/paper/2022/file/ba1c5356d9164bb64c446a4b690226b0-Paper-Conference.pdf  
  Strong survey/benchmark of teacher-aware RL methods including rehearsal and kickstarting.

- **DQfD**  
  https://arxiv.org/abs/1704.03732  
  Demonstrations remain inside the training loop, not just in pretraining.

### What this means for our repo

This family already partly exists in the repo:

- teacher CE / KL
- teacher replay
- trace-based checkpoint ranking
- disagreement-aware trace tooling

The good news:

- we already know this works partially
- our best branch (`0.95`) came from this family

The bad news:

- all the schedule variants we tried afterward still plateaued or drifted
- staged APPO continuation also failed
- simple action reweighting did not solve the last mile

So the question is not whether this family is valid. It is whether our current version is still too weak or too static.

### How it would look in this repo

The more serious version of this path would likely include:

- stronger on-policy distillation on student states
- replay that is not only static trace replay but also student-trajectory teacher relabeling
- maybe reward-extrapolation style weighting between teacher KL and RL reward
- maybe stronger dense policy regularization while critic/value is allowed to adapt

The key distinction from what we have now is:

- not more coefficient twiddling
- but a better formulation of how reward and teacher alignment are combined on student rollouts

### Strengths

- much closer to existing code
- reuses current world-model, teacher, trace-eval, and replay machinery
- lower implementation risk
- already shown to produce the repo’s best online result

### Weaknesses

- we may already be near the ceiling of what this APPO-based version can do
- if the base improver is wrong, better distillation may only preserve the teacher rather than surpass it
- easy to keep spending time on schedule tuning without real progress

### Overall assessment

This is the strongest **incremental** direction. If the team wants the cheapest next bet, this is still it.

But the repo’s evidence increasingly suggests that this family, when implemented as “APPO plus teacher extras,” may top out at preservation rather than improvement.

## Direct Comparison

### Which direction better matches the current evidence?

**Behavior-regularized offline-to-online RL** is the better match to the repo’s deepest problem:

- good offline policy
- bad online fine-tuning dynamics
- reward / proxy mismatch
- repeated on-policy drift

**Teacher-replay / on-policy distillation** is the better match to the repo’s current implementation:

- we already have most of the machinery
- it gave the best current run
- it is the easiest next thing to extend

### Which is more likely to beat the teacher?

If “beat the teacher” means a real, repeatable improvement on the trusted trace metric:

- **teacher replay / OPD** is more likely to get another small step quickly
- **behavior-regularized O2O RL** is more likely to be the real long-term answer

### Which one should the repo do next?

My recommendation is a split:

1. Finish one more meaningful teacher-distillation branch only if it is structurally different from current static APPO tuning.
   Example:
   - dense KL / OPD-style formulation
   - stronger student-state relabeling
   - less dependence on PPO value updates as the dominant force

2. In parallel, start the behavior-regularized offline-to-online branch as the likely successor.

If only one branch can be funded:

- pick **behavior-regularized offline-to-online RL**

## Recommendation For This Repo

### Near-term

Do not launch another large APPO run from the current schedule family.

The current debug evidence says:

- APPO schedule tuning is saturated
- staged continuation did not help
- the world model is not the bottleneck

### Next build target

Implement a new learner branch with these properties:

- offline teacher / behavior policy is first-class
- online updates are explicitly constrained by behavior uncertainty or offline-likeness
- world-model features stay in the observation path
- teacher traces and held-out trace match remain the primary gate

In other words:

- keep the current data
- keep the current representation
- change the online improver

## Bottom Line

The two options are both defensible, but they are not equally strategic.

- **Teacher replay / on-policy distillation** is the best continuation of the current code.
- **Behavior-regularized offline-to-online RL** is the best answer to what the repo is actually telling us.

If the goal is only to get one more better short-run checkpoint, teacher replay is the safer extension.

If the goal is to build the branch most likely to eventually beat the teacher, behavior-regularized offline-to-online RL is the better bet.
