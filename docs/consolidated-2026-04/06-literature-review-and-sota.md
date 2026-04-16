# Literature Review And SOTA Context

## Purpose

This document consolidates the repo’s research-facing markdown into one literature review. It is not a generic survey of all NetHack or RL work. It is a task-oriented review of the approaches that the repo repeatedly used to justify design decisions.

Primary source notes for this review:

- [references/README.md](/home/luc/rl-nethack-worktree-20260416/references/README.md)
- [RESEARCH-NOTES-WORLD-MODELS-NETHACK.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/research-notes/RESEARCH-NOTES-WORLD-MODELS-NETHACK.md)
- [RESEARCH-NOTES-NETHACK-HORIZONS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/research-notes/RESEARCH-NOTES-NETHACK-HORIZONS.md)
- [MAESTROMOTIF-INTEGRATION.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/research-notes/MAESTROMOTIF-INTEGRATION.md)
- [RL-HARNESS-TASKS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/RL-HARNESS-TASKS.md)
- [REPORT-LLM-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-LLM-WORLD-MODEL-2026-04-06.md)
- [REPORT-TWO-OPTIONS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-TWO-OPTIONS-2026-04-06.md)
- [PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md)

## The Problem Class The Repo Is Actually In

The repo’s literature framing converges on a specific regime:

- long-horizon sparse-reward control,
- partial observability,
- strong offline teachers or demonstrations,
- weak or misaligned online reward,
- and a need for trustworthy evaluation under shifting representations.

That framing explains why the docs repeatedly prefer:

- teacher-centric methods over plain PPO tuning,
- skill decompositions over one flat reward,
- offline pretraining and representation learning,
- and benchmark hygiene over noisy live evaluation.

## Baseline Systems The Repo Uses As Anchors

### NetHack Learning Environment

Reference:

- NLE paper: https://arxiv.org/abs/2006.13760

Repo-local takeaway:

- NLE is the foundation, but raw score and ascension are not convenient early objectives.
- NetHack is presented as an exploration, planning, and generalization benchmark rather than a simple dense-reward control problem.

This supports the repo’s shift toward shaped tasks and teacher-aligned metrics.

### AutoAscend

Reference:

- AutoAscend repo described in [references/README.md](/home/luc/rl-nethack-worktree-20260416/references/README.md)

Repo-local takeaway:

- AutoAscend remains the strongest expert-behavior source in the repo’s reference set.
- It is useful as an expert trace source and strategic behavior reference, not as a learning algorithm template.

### NetPlay

Reference:

- NetPlay paper and repo summarized in [references/README.md](/home/luc/rl-nethack-worktree-20260416/references/README.md)

Repo-local takeaway:

- structured observations plus skill framing are valuable for LLM-driven NetHack interaction,
- but pure LLM low-level control is not enough for the repo’s target outcome.

## Skill And Hierarchy Literature

### SkillHack

Reference:

- https://arxiv.org/abs/2207.11584

Repo-local takeaway:

- predefined skills are a useful inductive bias in sparse-reward NetHack-like settings.
- This supports the repo’s task-family decomposition and options framing.

Used in:

- [RL-HARNESS-TASKS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/RL-HARNESS-TASKS.md)
- [MAESTROMOTIF-INTEGRATION.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/research-notes/MAESTROMOTIF-INTEGRATION.md)

### MaestroMotif / Motif

References:

- MaestroMotif ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/file/2dc5a0faac8102fd47363795f71126ee-Paper-Conference.pdf
- Motif ICLR 2024: https://openreview.net/pdf/7fafd8a9cedfff67f51f1129cf8c1a76f436d271.pdf

Repo-local takeaway:

- use language and AI feedback to define and judge skills,
- do not use an LLM as the low-level primitive-action controller,
- and promote skill/task decomposition into real options.

This literature is the clearest external justification for the repo’s task harness and options-oriented thinking.

### Long-Horizon Planning With Predictable Skills

Reference:

- https://openreview.net/pdf/bf6fafd191fb8d7ac2b6d26a8420b15744d5b005.pdf

Repo-local takeaway:

- long-horizon partially observed control is better handled through abstract skill-conditioned modeling than primitive one-step planning alone.

This is why the world-model notes repeatedly prefer k-step or skill-conditioned outcomes over pure one-step prediction.

## World Models And Representation Learning

### NetHack Offline Pretraining

Reference:

- https://arxiv.org/abs/2304.00046

Repo-local takeaway:

- offline predictive pretraining can improve representation learning and sample efficiency in NetHack-like regimes.
- This is one of the strongest pieces of support for the repo’s world-model and teacher-building work.

### DreamerV3

Reference:

- https://arxiv.org/abs/2301.04104

Repo-local takeaway:

- useful as a general world-model RL reference,
- but too heavyweight to be copied directly into the current repo.

### TD-MPC2

Reference:

- https://arxiv.org/abs/2310.16828

Repo-local takeaway:

- short-horizon latent planning and model-predictive control are relevant,
- especially for action ranking or constrained imagination,
- but full replacement of the current stack is not the immediate next step.

### MuZero

Reference:

- https://arxiv.org/abs/1911.08265

Repo-local takeaway:

- the model should focus on value-relevant dynamics, not only reconstruction.

This directly influenced the repo’s interest in predicting teacher-relevant quantities such as frontier gain, revisit risk, and short-horizon progress.

### Text-Game World Models And LLM-Backed World Models

References:

- EMNLP 2022 text-game world-model paper: https://aclanthology.org/2022.emnlp-main.86.pdf
- COLING 2025 fine-tuning warning: https://aclanthology.org/2025.coling-main.445.pdf
- TEXT2WORLD: https://aclanthology.org/2025.findings-acl.1337.pdf
- WorldLLM: https://arxiv.org/abs/2506.06725

Repo-local takeaway:

- pretrained language semantics can help with text-like world modeling,
- frozen or lightly adapted text backbones are safer than fully RL-finetuned LMs,
- and direct language-model world modeling is still limited in structured domains.

This literature directly supports the repo’s text-conditioned world-model decision:

- use a frozen pretrained text encoder over trace prompts,
- train only the fusion and predictive heads,
- and judge success by downstream teacher quality rather than hype around “LLM world models.”

Primary repo interpretation:

- [REPORT-LLM-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-LLM-WORLD-MODEL-2026-04-06.md)
- [RESEARCH-NOTES-WORLD-MODELS-NETHACK.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/research-notes/RESEARCH-NOTES-WORLD-MODELS-NETHACK.md)

### JOWA And WHALE

References:

- JOWA: https://arxiv.org/abs/2410.00564
- WHALE: https://arxiv.org/abs/2411.05619

Repo-local takeaway:

- offline world-model pretraining and action modeling can share useful structure,
- but distribution shift and uncertainty remain the central dangers.

This matches the repo’s later conclusion that world-model branches help more as conservative representation builders than as unconstrained online planners.

## Teacher-Aware RL And Demonstration-Constrained Improvement

### DAgger

Reference:

- https://proceedings.mlr.press/v15/ross11a.html

Repo-local takeaway:

- one-shot imitation is not enough because the learner visits its own states.
- This is the cleanest justification for keeping DAgger-like relabeling and replay loops in the codebase.

### Kickstarting

Reference:

- https://arxiv.org/abs/1803.03835

Repo-local takeaway:

- teacher supervision should stay active during RL, not only during initialization.

This strongly supports teacher-regularized APPO over plain APPO from a BC checkpoint.

### DQfD

Reference:

- https://arxiv.org/abs/1704.03732

Repo-local takeaway:

- demonstrations should remain inside the training loop instead of being used only for pretraining.

### AWAC And BRAC

References:

- AWAC: https://arxiv.org/abs/2006.09359
- BRAC: https://arxiv.org/abs/1911.11361

Repo-local takeaway:

- when we trust the offline behavior distribution more than the online reward, policy improvement should be behavior-constrained.

These papers are the main reason the repo later compared:

- teacher-replay / on-policy distillation
- versus behavior-regularized offline-to-online RL

### Reincarnating RL

Reference:

- https://papers.neurips.cc/paper_files/paper/2022/file/ba1c5356d9164bb64c446a4b690226b0-Paper-Conference.pdf

Repo-local takeaway:

- teacher-aware RL is a real family of methods, not just an ad hoc patch.
- rehearsal and distillation are principled design choices when a strong teacher exists.

### Recent Offline-to-Online And Distillation Directions

References used in the repo’s strategic comparison:

- https://arxiv.org/abs/2508.06806
- https://arxiv.org/abs/2509.24748
- https://arxiv.org/abs/2511.03695
- https://arxiv.org/abs/2511.03828
- https://arxiv.org/abs/2601.22475
- https://arxiv.org/abs/2602.12125

Repo-local takeaway:

- the literature direction most aligned with the repo’s current bottleneck is not “more PPO tuning,”
- it is either better teacher-replay / on-policy distillation,
- or a more behavior-regularized offline-to-online improver.

Primary repo interpretation:

- [REPORT-TWO-OPTIONS-2026-04-06.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/REPORT-TWO-OPTIONS-2026-04-06.md)

## Horizon And Budget References

The repo’s horizon note compares its budgets to prior NetHack work.

Important references:

- NLE baseline scale via the NLE repo
- Sample Factory NetHack docs
- nle-language-wrapper
- Motif
- HiHack
- NetHack Challenge rules

Primary source:

- [RESEARCH-NOTES-NETHACK-HORIZONS.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/research-notes/RESEARCH-NOTES-NETHACK-HORIZONS.md)

Repo-local takeaway:

- rollout chunk length is not the main mismatch,
- episode and trace horizons are still short relative to serious NetHack work,
- and total training budget remains tiny compared with prior large-scale systems.

## Literature Conclusions That Best Match The Repo

The literature most strongly supports the following repo-local conclusions:

1. Skill decomposition is a good fit for NetHack.
2. Offline pretraining and latent representation learning are useful.
3. A frozen pretrained text encoder is a safer world-model ingredient than fully fine-tuning an LLM in the RL loop.
4. Demonstrations and teachers should remain active during online training.
5. Behavior-constrained offline-to-online RL is a strong strategic alternative when on-policy PPO-style updates drift.
6. Horizon and budget still matter, but better evaluation and teacher alignment were the more urgent repo-local breakthroughs.

## Practical Implication For This Repo

The literature does not say:

- replace everything with an LLM,
- run larger APPO jobs and hope,
- or trust generic online reward curves.

It says something much narrower and more useful for this codebase:

- build a strong teacher,
- use trustworthy benchmarks,
- learn better representations,
- keep teacher information inside the online loop,
- and move toward skill-aware or behavior-constrained improvers when plain APPO saturates.

