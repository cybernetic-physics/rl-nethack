# Plan WAGMI

## Goal

Turn the current repo from:

- strong offline teacher
- stable but flat online improver

into:

- strong offline teacher
- teacher-beating online improver

without losing the disciplined debug loop that made the current diagnosis trustworthy.

The immediate target is:

- beat the current teacher on held-out deterministic trace match

The broader target is:

- establish a reliable path from offline teacher -> online improvement

## Planning Principles

This plan is built around a few explicit principles.

### Principle 1: preserve interpretability

We should prefer a slower but interpretable progression over a faster but ambiguous one.

### Principle 2: promote only on evidence

A branch earns promotion to medium or large scale only by beating the current baseline on the trusted metric.

### Principle 3: keep the baseline alive

The current best branch should always remain runnable and comparable while new work lands.

### Principle 4: separate infrastructure work from learner work

We should not bundle:

- replay instrumentation
- replay scheduling
- new learner class

into the same change unless there is a compelling reason.

## Scope

This plan is intentionally focused.

It is not trying to solve:

- full-game NetHack
- final long-horizon evaluation design
- every possible alternative learner at once

It is trying to solve:

- how to get from the current strong teacher to a learned checkpoint that actually beats that teacher on the current trusted benchmark

## Plan Structure

The plan is organized around three types of work:

1. structural changes to the current APPO branch
2. support changes that make teacher data more useful
3. an explicit escape hatch into a new improver class if APPO still stalls

That separation matters because the repo has already spent too much time mixing:

- instrumentation
- feature work
- learner work

inside single experiment loops.

## Ground Rules

1. The source of truth remains held-out deterministic trace match.
2. No large run is allowed unless a short or medium run produces a learned checkpoint above the step-0 teacher clone.
3. Each serious run must report:
   - step-0 teacher clone
   - best learned checkpoint
   - final checkpoint
4. We change one layer of the system at a time:
   - replay schedule
   - replay content
   - learner class
5. We do not demote the current `v4 + wm_concat_aux` representation unless a replacement clearly wins.
6. Every phase must produce explicit artifacts:
   - code changes
   - tests
   - one short result
   - if promoted, one medium result
7. We do not declare a branch "promising" unless it beats the current teacher clone in a learned checkpoint, not just at step 0.
8. Every promoted branch must remain reproducible with one command path.
9. If a branch adds new metrics, those metrics must be logged in a way that can be compared across runs.

## Diagnosis Recap

From [wagmi.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/wagmi.md), the strongest current diagnosis is:

- representation is no longer the main bottleneck
- the warm-start bridge is no longer the main bottleneck
- the current APPO + static teacher replay branch preserves the teacher early
- it still does not improve beyond the teacher
- the next likely gains must come from better use of teacher data during online learning

That implies the next plan should focus on:

1. scheduled replay
2. prioritized replay
3. targeted DAgger
4. only then, if needed, a new demo-regularized improver

## Baseline To Beat

The current baseline row should be treated as fixed until a new branch clearly beats it.

### Current baseline

- representation: `v4 + wm_concat_aux`
- teacher source: offline teacher from the current world-model branch
- online branch: conservative APPO + static teacher replay
- teacher clone trace match: `0.9375`
- best learned checkpoint trace match: `0.9375`
- best checkpoint env steps: `352`

This is the row every new branch has to displace.

## Promotion Criteria

The current branch structure needs explicit promotion criteria.

### Promote to medium run if:

- a short run produces a learned checkpoint above the teacher clone

### Promote to large run if:

- a medium run preserves that win or improves on it

### Promote a new learner class to serious attention if:

- APPO fails the hard stop criteria
- and the new learner produces a short-run learned checkpoint that is at least competitive with the APPO baseline

### Do not promote if:

- the branch only ties the teacher clone
- the branch improves reward but not trace match
- the branch cannot be reproduced cleanly

## Phase 0: Freeze Baseline

### Objective

Make sure the current best branch is fully documented and reproducible before changing it.

### Actions

1. Archive the current large static-replay run result.
2. Record the exact baseline teacher score for the current held-out trace set.
3. Record the exact baseline online branch score:
   - step-0
   - best learned
   - final

### Files

- [wagmi.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/wagmi.md)
- [CURRENT-RL-SYSTEM.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/reports/CURRENT-RL-SYSTEM.md)
- [rl/checkpoint_tools.py](/home/luc/rl-nethack/rl/checkpoint_tools.py)

### Exit Criteria

- a single baseline row exists that future branches can compare against

### Artifacts

- one baseline summary row
- one short note or report capturing the exact branch and checkpoint paths

### Prerequisites

None. This phase should happen immediately and only once per baseline reset.

## Phase 1: Scheduled Teacher Replay

### Objective

Turn teacher replay from a static side loss into a phase-based constraint.

### Why

The current static replay branch appears to:

- preserve
- then plateau

The most plausible next improvement is to make replay stronger early and then relax it later.

### Code Changes

#### 1. Config surface

Add to [rl/config.py](/home/luc/rl-nethack/rl/config.py):

- `teacher_replay_final_coef`
- `teacher_replay_warmup_env_steps`
- `teacher_replay_decay_env_steps`
- `teacher_replay_priority_power`
- `teacher_replay_source_mode`

#### 2. APPO CLI/config plumbing

Update:

- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [cli.py](/home/luc/rl-nethack/cli.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)

to pass all new replay schedule fields through end to end.

#### 3. Learner logic

Update [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py):

- add `_scheduled_teacher_replay_coef(...)`
- apply scheduled replay coefficient instead of only a static scalar
- record current replay coefficient in summaries

#### 4. Optional preset support

Add one or two named schedule presets to reduce CLI ambiguity, for example:

- `anchor_then_probe`
- `anchor_then_release`

### Validation

#### Unit tests

Add/update [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py):

- schedule values appear in config and argv
- replay coefficient schedule produces expected values across env-step ranges

#### Debug runs

Run:

1. short run with high early replay, lower late replay
2. medium run with same schedule

### Suggested Experiment Matrix

Keep the matrix intentionally small.

#### Candidate schedules

1. strong replay early -> moderate replay later
2. strong replay early -> weak replay later
3. moderate replay early -> weak replay later

Keep the first matrix intentionally small.
If none of these schedules work, do not explode the search space before reviewing the results.

#### Fixed settings to preserve

- keep current `v4 + wm_concat_aux`
- keep current teacher source
- keep current dense checkpointing
- keep the current conservative optimizer settings unless the schedule itself clearly requires a change

### What To Measure

For each schedule:

- step-0 trace match
- first learned checkpoint trace match
- best learned checkpoint trace match
- final trace match
- replay coefficient over time

### How To Interpret Outcomes

If a branch:

- beats `0.9375` in a learned checkpoint:
  it is promotion-worthy

- ties `0.9375` but remains stable longer:
  it is a preservation improvement, not yet an online improvement

- trails `0.9375` immediately:
  it is a regression

### Prerequisites

- Phase 0 baseline frozen
- current branch still runnable

### Success Criteria

- a learned checkpoint beats the step-0 teacher clone on held-out trace match

### Failure Criteria

- learned checkpoints still only tie or trail the clone across 2-3 structurally different schedules

### Artifacts

- tests for scheduled replay
- one short-run table
- one medium-run table if promoted

### Rollback Criteria

Rollback or abandon a schedule if:

- it regresses badly below the teacher clone in the first learned checkpoint
- or it introduces logging/summary ambiguity

## Phase 2: Prioritized Replay

### Objective

Stop sampling teacher replay uniformly.

### Why

The current learner likely needs more supervision on:

- disagreement states
- weak-action states
- loop/failure states

than on already-solved teacher states.

### Code Changes

#### 1. Enrich trace rows

Update [rl/traces.py](/home/luc/rl-nethack/rl/traces.py) to support optional row metadata:

- `is_disagreement_candidate`
- `is_weak_action`
- `is_loop_risk`
- `is_failure_slice`
- `teacher_action_index`

These can initially be generated offline from:

- disagreement reports
- action-class rules
- repeated-state / repeated-action markers

#### 2. Replay sampler

Update [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py):

- load replay metadata fields
- sample according to `teacher_replay_source_mode`
- sample according to weighting fields

Possible source modes:

- `uniform`
- `weak_action`
- `disagreement`
- `mixed`

#### 2b. Priority weighting contract

Define clearly whether weights mean:

- sampling probability only
- or sampling probability plus loss weighting

Do not leave that ambiguous in implementation.

#### 3. Logging

Also in [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py), log:

- replay source composition
- weak-action replay fraction
- disagreement replay fraction
- replay agreement

### Validation

#### Unit tests

Add tests that:

- enriched trace rows serialize and reload
- replay sampler oversamples targeted subsets correctly

#### Debug runs

Run short experiments for:

1. `uniform`
2. `weak_action`
3. `disagreement`
4. `mixed`

### Suggested Source Modes

The first pass should stay narrow:

1. `uniform`
2. `disagreement`
3. `weak_action`
4. `mixed`

Do not add more source modes until one of these is clearly useful.

### Prerequisites

- Phase 1 schedule path exists
- replay schedule summaries are already visible

### What To Measure

For each source mode:

- replay source composition
- replay agreement
- weak-action disagreement summary
- best learned trace match

### How To Interpret Outcomes

If:

- `disagreement` beats `uniform`, then state targeting is likely the main missing lever
- `weak_action` beats `uniform`, then action imbalance remains the sharper issue
- `mixed` beats both, then the next phase should keep mixed replay as default

### Artifacts

- replay-enriched trace schema
- replay sampling tests
- short-run comparison table across source modes

### Rollback Criteria

Rollback or abandon a source mode if:

- it degrades trace match consistently without improving any useful slice metric

### Success Criteria

- at least one prioritized replay branch beats the teacher clone in a short run

### Failure Criteria

- none of the source modes improve over scheduled replay alone

## Phase 3: Targeted DAgger

### Objective

Upgrade DAgger from generic iterative relabeling to a focused support mechanism for the learner’s failure states.

### Why

The current DAgger machinery is mechanically sound but too blunt. It needs to help where the learner drifts, not just add more teacher data.

### Prerequisites

- replay metadata exists, or can be generated from trace disagreement reports

### Code Changes

#### 1. Add focused aggregation modes

Update [rl/dagger.py](/home/luc/rl-nethack/rl/dagger.py):

- `merge_policy=disagreement_only`
- `merge_policy=weak_action_focus`
- `merge_policy=loop_focus`
- `merge_policy=mixed_focus`

#### 2. Add trace filtering helpers

Update [rl/traces.py](/home/luc/rl-nethack/rl/traces.py):

- filtering by disagreement metadata
- filtering by weak actions
- filtering by loop/failure markers

#### 3. Held-out gating stays mandatory

Every DAgger iteration must remain gated by held-out trace performance.

### Validation

#### Unit tests

Add coverage for:

- disagreement-only merge
- weak-action-focused merge
- held-out stop behavior under focused modes

#### Offline evaluation

Require that targeted DAgger improves the offline teacher before it is allowed back into the online loop.

### Suggested Focus Order

Do not try every focus mode equally.
Try in this order:

1. `disagreement_only`
2. `weak_action_focus`
3. `mixed_focus`
4. `loop_focus`

Why:

- disagreement and weak-action failures are the most supported by current evidence
- loop/failure slices are useful, but less clearly the next sharpest lever

### What To Measure

- held-out offline teacher improvement
- weak-action recall change
- disagreement histogram change

### Interpretation

If targeted DAgger improves the offline teacher but the online learner still ties the teacher, that strongly suggests the bottleneck has moved more fully into the online learner.

### Artifacts

- focused DAgger merge modes
- offline comparison table

### Rollback Criteria

Rollback or abandon a focused mode if:

- it lowers held-out offline teacher quality
- and does not clearly improve any relevant slice metric

### Success Criteria

- offline teacher improves on held-out trace match

### Failure Criteria

- targeted DAgger does not improve the offline teacher after several focused variants

## Phase 4: Objective Introspection

### Objective

Determine whether remaining drift is mostly:

- actor-side
- critic-side
- or objective mismatch against the trace benchmark

### Why

Before building a new improver class, we should extract the strongest possible explanation from the current branch.

### Prerequisites

- at least one scheduled/prioritized replay branch has been run

### Code Changes

#### 1. Replay agreement metrics

Update [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py):

- log replay agreement per batch
- log replay agreement by source type if available

#### 2. Extend checkpoint metadata

Update [rl/checkpoint_tools.py](/home/luc/rl-nethack/rl/checkpoint_tools.py):

- optionally write disagreement slices alongside `best_trace_match.json`

#### 3. Optional training summaries

If cheap enough, log:

- actor-vs-teacher agreement on current minibatch
- actor-vs-teacher agreement on replay batch

### Validation

Run short introspection-focused training jobs and inspect whether:

- replay agreement remains high while trace match falls
- or replay agreement also falls

### Interpretation

If replay agreement stays high while trace quality falls:

- suspect the critic/value path more strongly

If replay agreement falls together with trace quality:

- suspect replay scheduling/content more strongly

### Artifacts

- added summaries
- one short introspection run
- one short memo or report on the interpretation

### Rollback Criteria

Do not keep introspection-only metrics that:

- are expensive
- and do not change decision quality

### Success Criteria

- we can state with more confidence whether critic/value dynamics are the next likely root problem

## Phase 5: Decide Whether APPO Still Deserves To Be Mainline

### Decision Rule

APPO remains the mainline improver only if:

- scheduled replay and prioritized replay produce a learned checkpoint above the teacher clone

APPO stops being the mainline if:

- 2-3 structurally different replay-aware branches still only preserve or trail the clone

This is a hard stop.

### Explicit Decision Meeting

After Phases 1 through 4, stop and answer one question:

- "Does APPO still deserve another cycle as the mainline online improver?"

Allowed answers:

1. yes, because a learned checkpoint beat the teacher clone
2. no, because multiple structural attempts only preserved or trailed the teacher

Not allowed:

- "maybe one more static variation"

### Inputs To The Decision

The decision should use:

- baseline row
- Phase 1 result table
- Phase 2 result table
- Phase 3 offline DAgger result
- Phase 4 introspection memo

## Phase 6: New Improver Branch

### Objective

If APPO still only preserves, move to a learner whose training geometry better matches the repo’s data situation.

### Candidate Directions

#### Option A: AWAC-like branch

New file:

- `rl/train_awac.py`

Would provide:

- offline-to-online fine-tuning
- advantage-weighted policy improvement
- stronger reuse of teacher data

#### Option B: DQfD-style demo actor-critic

New file:

- `rl/train_demo_actor_critic.py`

Would provide:

- demonstration replay as a core part of learning
- actor-critic updates built around demo data, not merely regularized by it

### Preferred Order

1. try the smallest APPO structural upgrade first
2. if that fails, build AWAC-like or demo-actor-critic branch

### Design Constraints For The New Branch

The new branch should:

1. reuse the current trusted evaluation path
2. reuse the current teacher and trace infrastructure
3. avoid rewriting the whole environment stack
4. make teacher/demo data structurally central, not an auxiliary afterthought

### Initial Deliverable

The first deliverable for the new improver branch is not a full training success.
It is:

- a minimal runnable trainer
- one short-run trace-gated comparison against the current APPO baseline

### Prerequisites

- Phase 5 concluded that APPO should no longer be the mainline improver

## Metrics

Every serious branch must log at least:

- step-0 trace match
- best learned trace match
- final trace match
- env steps at best checkpoint
- invalid action rate
- weak-action disagreement summary
- repeated action / loop proxy
- replay source composition
- replay agreement

Optional but useful:

- actor-vs-teacher minibatch agreement
- critic loss trend at the best checkpoint
- replay priority histogram if prioritized replay is active

### Metrics We Should Avoid Over-Interpreting

Be careful with:

- raw live reward
- raw value loss
- raw FPS

These are useful diagnostics, but not promotion metrics.

### Scoreboard Format

Each branch should produce one summary row with:

- experiment name
- representation
- teacher source
- replay schedule
- replay source mode
- step-0 score
- best learned score
- final score
- steps at best checkpoint
- notes

## Validation Ladder

No branch should skip this ladder.

### Level 1: unit tests

Add or update tests in [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

### Level 2: short debug run

Goal:

- does a learned checkpoint beat the clone quickly?

### Level 3: medium run

Goal:

- does the branch remain stable after the first improvement?

### Level 4: large run

Only allowed if:

- Level 2 or Level 3 produced a learned checkpoint above the teacher clone

### Rollback Rule

If a branch regresses badly at Level 2, do not patch it ad hoc and immediately rerun large.

Instead:

1. inspect metrics
2. write a short conclusion
3. either revise structurally or discard the branch

## Failure Handling

Each failed branch should still leave behind:

- one short note on why it failed
- the best checkpoint it produced
- whether the failure was:
  - stability
  - objective
  - data
  - representation

That prevents repeated rediscovery of the same dead ends.

## Stop Conditions

### Stop condition for scheduled replay APPO

If short and medium runs do not beat the clone after a bounded number of structural attempts, stop using APPO as the primary improver.

### Stop condition for targeted DAgger

If targeted DAgger does not improve the offline teacher, keep it auxiliary.

### Stop condition for new world-model work

If a world-model change does not improve the offline teacher or the online improver, it should not absorb the project’s main attention.

## Ownership By File

This is the likely ownership map for implementation.

### Replay schedule and replay sampling

- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)

### Config and CLI

- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [cli.py](/home/luc/rl-nethack/cli.py)

### Trace schema and metadata enrichment

- [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)

### Focused relabeling

- [rl/dagger.py](/home/luc/rl-nethack/rl/dagger.py)

### Objective introspection and checkpoint reporting

- [rl/checkpoint_tools.py](/home/luc/rl-nethack/rl/checkpoint_tools.py)
- [rl/trace_eval.py](/home/luc/rl-nethack/rl/trace_eval.py)

### Future new improver branch

- new file, likely:
  - `rl/train_awac.py`
  - or `rl/train_demo_actor_critic.py`

## Execution Order Constraints

The order in this plan is not arbitrary.

### Must happen before scheduled replay experiments

- baseline freeze
- replay schedule config plumbing

### Must happen before prioritized replay experiments

- replay metadata schema
- replay schedule logging

### Must happen before APPO go/no-go decision

- at least one scheduled replay matrix
- at least one prioritized replay comparison

### Must happen before new improver branch becomes main focus

- explicit APPO hard-stop decision

## Practical Sequence

If starting today, the best sequence is:

1. finish and archive the current x10 static-replay run
2. implement scheduled replay
3. implement replay-priority metadata
4. implement prioritized replay
5. run short and medium scheduled/prioritized replay experiments
6. if none beat the teacher clone, implement the next improver branch
7. if one does beat the clone, repeat it before trusting it
8. only then run a large-scale branch again

## What Not To Do

To keep the plan robust, explicitly do not:

1. launch another large static-replay run on the current branch
2. expand the world-model branch again before testing scheduled/prioritized replay
3. run a wide parameter sweep without first defining what decision it is meant to support
4. add a new learner class before the APPO hard-stop decision is made

## Concrete Deliverables Checklist

### Phase 0 deliverables

- [ ] baseline row recorded
- [ ] exact checkpoint paths recorded

### Phase 1 deliverables

- [ ] replay schedule config fields added
- [ ] replay schedule tests added
- [ ] short schedule comparison run completed
- [ ] medium schedule run completed if promoted

### Phase 2 deliverables

- [ ] replay metadata fields added to traces
- [ ] replay sampler supports source modes
- [ ] replay source composition logged
- [ ] short prioritized replay comparison completed

### Phase 3 deliverables

- [ ] focused DAgger modes added
- [ ] held-out offline comparison completed

### Phase 4 deliverables

- [ ] replay agreement metric added
- [ ] one introspection run completed
- [ ] short diagnosis note written

### Phase 5 deliverables

- [ ] explicit go/no-go decision on APPO as mainline improver

### Phase 6 deliverables

- [ ] new improver skeleton exists if needed
- [ ] first short comparison exists if needed

## Suggested Milestone Names

Using fixed milestone names will make reports and commits easier to read.

### Milestone A

- baseline frozen
- replay schedule implemented

### Milestone B

- prioritized replay implemented
- source-mode comparison completed

### Milestone C

- targeted DAgger support completed
- offline teacher comparison completed

### Milestone D

- APPO go/no-go decision made

### Milestone E

- new improver branch started if needed

## Deliverables

At the end of this plan, the repo should have:

1. a stronger replay-aware APPO branch
2. a cleaner answer on whether APPO can still be the mainline improver
3. a targeted DAgger support path
4. a clearer scoreboard for every branch
5. if needed, a new demo-regularized improver branch

## Review Points

The plan should be reviewed after:

1. Phase 1 short-run results
2. Phase 2 prioritized replay comparison
3. Phase 4 introspection pass
4. Phase 5 go/no-go decision

Those are the natural control points where the repo may need a strategic update rather than just implementation work.

## Final Statement

This plan assumes the main lesson from [wagmi.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/wagmi.md) is correct:

- the next work should center teacher-data scheduling, replay prioritization, and then, if necessary, a better improver class

That is the most justified branch to push now.

The plan is intentionally strict because the repo is finally in a state where disciplined iteration can pay off.
The main risk now is not lack of ideas.
It is wasting time on branches that no longer deserve it.

That is why this plan is biased toward:

- narrow experiment matrices
- explicit gates
- explicit stop conditions
- and hard decisions when a branch stops earning more time
