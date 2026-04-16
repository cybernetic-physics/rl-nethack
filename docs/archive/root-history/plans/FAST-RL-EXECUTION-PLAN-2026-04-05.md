# Fast RL Execution Plan (2026-04-05, Post Non-RNN X100)

This document defines the next execution plan for improving the repo after:

- deterministic trace evaluation was established,
- the BC-to-APPO warm-start bridge was repaired,
- the non-RNN APPO path reached a real improvement,
- and the main race-condition hazards in the harness were hardened.

It is grounded in:

- [RUN-REPORT-2026-04-05-X100-NORNN.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/run-reports/RUN-REPORT-2026-04-05-X100-NORNN.md)
- [POSTMORTEM-NEXT-STEPS-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/POSTMORTEM-NEXT-STEPS-2026-04-05.md)
- [FAST-RL-UPGRADE-PLAN-2026-04-05.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/FAST-RL-UPGRADE-PLAN-2026-04-05.md)

It is also informed by these papers already pulled into the repo:

- [kickstarting_arxiv2018.pdf](/home/luc/rl-nethack/references/papers/kickstarting_arxiv2018.pdf)
- [dagger_aistats2011.pdf](/home/luc/rl-nethack/references/papers/dagger_aistats2011.pdf)
- [dqfd_arxiv2017.pdf](/home/luc/rl-nethack/references/papers/dqfd_arxiv2017.pdf)
- [brac_arxiv2019.pdf](/home/luc/rl-nethack/references/papers/brac_arxiv2019.pdf)
- [awac_arxiv2020.pdf](/home/luc/rl-nethack/references/papers/awac_arxiv2020.pdf)
- [skillhack_arxiv2022.pdf](/home/luc/rl-nethack/references/papers/skillhack_arxiv2022.pdf)
- [maestromotif_iclr2025.pdf](/home/luc/rl-nethack/references/papers/maestromotif_iclr2025.pdf)
- [option_critic_arxiv2016.pdf](/home/luc/rl-nethack/references/papers/option_critic_arxiv2016.pdf)
- [deep_rl_that_matters_arxiv2017.pdf](/home/luc/rl-nethack/references/papers/deep_rl_that_matters_arxiv2017.pdf)


## 1. Current State

The repo is no longer blocked on basic RL infrastructure.

What now works:

- deterministic trace evaluation
- trace sharding and disagreement reports
- BC training/eval
- APPO with a working Sample Factory backend
- non-RNN BC warm start that actually loads
- trace-based checkpoint ranking
- atomic writes and experiment locking around key RL artifacts

Most important current metrics:

- BC baseline on trusted trace set: `0.6395`
- repaired non-RNN APPO warm start at step 0: `0.6047`
- best repaired APPO checkpoint so far: `0.6240`

So:

- APPO is now improving from the repaired warm start
- but it still does not beat BC

That means the repo has crossed the line from “fix the harness” to
“improve the learning recipe.”


## 2. Main Diagnosis

The current bottleneck is not throughput and not basic correctness.

The bottleneck is:

- **teacher drift during RL**

The current low-level story is:

1. BC gives the strongest teacher-aligned policy.
2. APPO from the repaired non-RNN warm start improves relative to the old broken path.
3. But APPO still does not surpass the BC teacher.

The papers suggest the same answer:

- Kickstarting: keep the teacher loss active during RL
- DAgger: expose the student to its own state distribution
- DQfD / demo-RL methods: keep demonstrations inside the learning loop
- BRAC / AWAC: constrain drift toward the behavior policy


## 3. Rule For Fast Iteration

Do **not** use long APPO runs as the first tool.

Every change should go through this ladder:

1. deterministic trace benchmark
2. BC-only fast loop
3. short APPO run
4. medium APPO run
5. long APPO run only if the short loop wins

If a change cannot improve a cheap trusted proxy, it should not get more GPU time.


## 4. Source Of Truth

The only trusted regression gate is:

- deterministic trace evaluation on fixed trace files

Primary trace:

- `data/tracefix_v2_explore_traces.jsonl`

Main commands:

```bash
uv run python cli.py rl-trace-report --input data/tracefix_v2_explore_traces.jsonl --bc-model output/tracefix_v2_explore_bc.pt --appo-experiment <exp>

uv run python cli.py rl-rank-checkpoints --experiment <exp> --trace-input data/tracefix_v2_explore_traces.jsonl

uv run python cli.py rl-trace-disagreements --input data/tracefix_v2_explore_traces.jsonl --bc-model output/tracefix_v2_explore_bc.pt --appo-experiment <exp>
```

Live seeded eval remains diagnostic-only.


## 5. Fast Loop Architecture

### Layer A: Direction-Focused Trace Slices

The trace disagreement reports already show stable directional biases:

- too much `west`
- weak `search`
- imperfect `east` / `south`

So every meaningful policy change should be tested on:

- full trace set
- `east,south` slice
- future `search` slice once enough examples exist

Use:

```bash
uv run python cli.py rl-shard-traces \
  --input data/tracefix_v2_explore_traces.jsonl \
  --output /tmp/east_south.jsonl \
  --teacher-actions east,south
```

Required metrics:

- overall match rate
- per-action precision / recall / f1
- common confusion pairs


### Layer B: BC-First Feature Loop

Any observation or representation change must go through BC first.

Why:

- BC is cheap
- BC is deterministic enough
- BC directly exposes feature quality
- BC failure is easier to interpret than APPO failure

Loop:

1. modify features
2. retrain BC on fixed trace set
3. evaluate on full trace set
4. evaluate on directional slice
5. only if BC improves, move to APPO


### Layer C: Short APPO Drift Loop

Use a short APPO ladder to measure how quickly the student drifts:

- `10k` frames
- `25k` frames
- `50k` frames
- `100k` frames only if still improving

At each checkpoint:

- rank by deterministic trace metric
- compare against the warm-start checkpoint
- compare against BC

If trace match regresses early, stop.


## 6. Immediate Workstreams

### Workstream 1: Teacher-Regularized APPO

This is the highest-value next model change.

Goal:

- stop APPO from drifting too far away from BC while training

Implementation:

- on student rollout states, compute BC teacher logits or argmax targets
- add a policy imitation term to the APPO loss
- schedule its weight over time rather than dropping it instantly to zero

Simplest first version:

- cross-entropy to the BC teacher argmax action

Better version:

- KL from student distribution to teacher distribution if the BC policy is exposed as logits

New code likely needed:

- `rl/teacher_regularization.py`
- updates to [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- updates to the APPO config/CLI

Fast validation:

- run the short APPO ladder above
- require that trace match does not drop below the warm-start checkpoint

Success criterion:

- best trace match exceeds `0.6240`


### Workstream 2: DAgger-Lite Done Properly

The repo already has a first DAgger-lite path, but the naive small merged retrain did not help.

That does not mean the idea is wrong.
It means the schedule is wrong.

Next DAgger loop:

1. roll out current student
2. relabel visited states with teacher
3. do not replace too much of the base data
4. test several merge schedules on short loops

Sweep:

- merge ratio: `0.05`, `0.10`, `0.20`, `0.33`
- student episodes: `4`, `8`, `16`
- retrain epochs: `5`, `10`, `20`

Gate:

- deterministic trace match on original trace set
- deterministic trace match on student-induced relabeled shard

Best outcome to look for:

- improved robustness on student states
- no meaningful regression on teacher traces


### Workstream 3: Behavior-Regularized Fine-Tuning

If teacher-regularized APPO is still unstable, test stronger policy anchoring.

This is motivated by BRAC and AWAC.

Idea:

- keep the APPO policy close to the BC policy using an explicit behavior penalty
- especially in the early and middle training phase

This is not “offline RL.”
It is using the same regularization principle inside the current online fine-tuning setup.


### Workstream 4: Feature Improvements For Directionality

The next feature work should be extremely targeted.

Do not do a broad encoder rewrite first.

Instead add only features that directly attack the known confusion:

- relative frontier direction counts
- immediate corridor orientation
- short recent displacement vector
- signed position change over the last few steps
- local occupancy summaries in front/left/right/back directions

Run these changes through BC first.


## 7. What Not To Do Next

Do not:

- launch another blind multi-hour APPO run first
- reintroduce recurrence before the recurrent bridge is teacher-equivalent
- use live seeded eval as the main success metric
- spend most effort on more infra or server work right now

The next gains are more likely to come from:

- teacher regularization
- DAgger schedule tuning
- better directional features


## 8. Concrete Command Plan

### 8.1 Baseline Snapshot

```bash
uv run python cli.py rl-trace-report \
  --input data/tracefix_v2_explore_traces.jsonl \
  --bc-model output/tracefix_v2_explore_bc.pt \
  --appo-experiment appo_tracefix_v2_bc_nornn_x100 \
  --detailed
```

### 8.2 Direction Slice

```bash
uv run python cli.py rl-shard-traces \
  --input data/tracefix_v2_explore_traces.jsonl \
  --output /tmp/tracefix_east_south.jsonl \
  --teacher-actions east,south
```

### 8.3 BC Feature Loop

```bash
uv run python cli.py rl-train-bc \
  --input data/tracefix_v2_explore_traces.jsonl \
  --output /tmp/bc_candidate.pt \
  --epochs 20 \
  --hidden-size 512 \
  --observation-version v2

uv run python cli.py rl-evaluate-bc \
  --model /tmp/bc_candidate.pt \
  --trace-input data/tracefix_v2_explore_traces.jsonl
```

### 8.4 Short APPO Ladder

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-appo \
  --experiment appo_short_candidate \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 10000 \
  --enabled-skills explore \
  --observation-version v2 \
  --bc-init-path output/tracefix_v2_explore_bc.pt \
  --no-rnn

uv run python cli.py rl-rank-checkpoints \
  --experiment appo_short_candidate \
  --trace-input data/tracefix_v2_explore_traces.jsonl
```


## 9. Success Criteria For The Next Phase

The next phase is successful if we achieve all three:

1. Beat the current best APPO deterministic trace score:
   - current target to beat: `0.6240`
2. Match or beat BC:
   - current target: `0.6395`
3. Improve the directional confusion profile:
   - reduce `west` over-prediction
   - improve `east` / `south` recall
   - produce non-trivial `search`


## 10. Bottom Line

The repo has reached the point where the right question is no longer:

- “can the RL harness run?”

The right question is:

- “how do we keep RL improving from the teacher instead of drifting away from it?”

The fastest credible answer is:

- teacher-regularized APPO
- DAgger-lite with tuned merge schedules
- BC-first feature work
- short trace-gated APPO ladders

That is the next plan.
