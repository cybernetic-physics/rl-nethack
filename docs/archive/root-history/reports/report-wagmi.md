# Report WAGMI

## Executive Summary

This report answers two questions:

1. What have the latest training runs shown?
2. Are we going to make it?

The short answer is:

- the repo is in a much better state than it was earlier
- the harness is now trustworthy enough to diagnose failure modes
- the latest scheduled-replay branch is the first branch that looks structurally correct under the current plan
- but the current large run has **not** yet beaten the teacher

So the honest answer is:

- **not yet**
- **but yes, this still looks solvable**

The project is no longer failing because of obvious infra bugs, broken warm starts, or invalid evaluation. It is failing because the current online improver is still too weak or too static to move beyond the teacher on the trusted metric.

## What We Ran

The current validation branch is:

- experiment: `appo_wm_v4_schedreplay_large_a`
- representation: `v4 + wm_concat_aux`
- teacher source: offline `v4` world-model-augmented BC teacher
- online learner: APPO with:
  - teacher CE loss
  - scheduled teacher replay
  - conservative optimizer settings
  - dense checkpointing
  - held-out trace-gated checkpoint ranking

Key settings:

- `env_max_episode_steps = 500`
- `rollout_length = 8`
- `learning_rate = 1e-4`
- `gamma = 0.99`
- `gae_lambda = 0.9`
- `value_loss_coeff = 0.1`
- `reward_scale = 0.005`
- `teacher_loss_coef = 0.01`
- `teacher_replay_coef = 0.02`
- `teacher_replay_final_coef = 0.005`
- `teacher_replay_warmup_env_steps = 128`
- `teacher_replay_decay_env_steps = 768`
- `teacher_replay_source_mode = uniform`

This branch was promoted because a short run produced the first learned checkpoint that matched the strongest teacher-clone baseline under the current `500`-step world-model regime.

## What Worked

### 1. The harness is finally good enough to trust

This matters more than it sounds.

Earlier in the project we were fighting:

- nondeterministic eval
- warm-start mismatch
- checkpoint-selection mistakes
- hidden actor/teacher architecture mismatch
- weak feature paths that looked plausible offline but collapsed online

Those are much less central now.

The current branch has:

- deterministic trace-based ranking
- fixed warm-start bridge
- consistent `v4 + wm_concat_aux` feature usage
- teacher replay and schedule plumbing
- dense checkpointing that preserves early policies

That means the results are now scientifically meaningful.

### 2. Scheduled replay is a real lever

The new scheduled replay path was not just paperwork.

In the short validation run, it produced:

- a learned checkpoint at `568` env steps
- held-out trace match `0.95`

That was the first strong signal that changing how teacher data is used online can help more than changing representation alone.

### 3. Large-run behavior is much healthier than the old collapsing branches

The current large run has shown:

- normal startup
- correct warm-start loading
- stable checkpoint production
- trace ranking happening on schedule
- recovery after at least one real value-loss spike

This is very different from earlier branches that immediately degraded into obvious behavioral collapse.

## What Did Not Work

### 1. The large run has not beaten the teacher

As of the current report state, the best trace-ranked checkpoint is still:

- checkpoint: `checkpoint_000000016_128.pth`
- env steps: `128`
- held-out trace match: `0.9375`

That means:

- the branch is preserving the teacher clone
- it is not yet producing a later checkpoint that clearly improves beyond it

### 2. The medium run exposed the same core failure pattern

The medium scheduled-replay run produced:

- best checkpoint at `176` env steps with `0.9375`
- final checkpoints down at `0.575`

So the medium run told us something very important:

- scheduled replay can help early
- but the current static replay-source setup still drifts badly over longer horizons

### 3. Value pressure is still the main numeric instability signal

Even in the healthier large run, we still observed:

- occasional large value-loss spikes
- recovery after some spikes, but not full confidence that return learning is well-scaled

This means the current branch is **more stable**, not **fully solved**.

## Current Best Interpretation

The repo is now in a new regime:

- before: broken or ambiguous
- now: stable enough to expose the real bottleneck

That bottleneck is:

- the online learner still does not know how to **improve beyond the teacher** without drifting away from the teacher manifold

That is why the best checkpoint remains early.

This pattern means:

- the teacher and representation are strong enough
- the online objective is still not producing the right kind of improvement pressure

In plain language:

- we have mostly solved “how do we start in the right place?”
- we have not yet solved “how do we move to a better place without falling off the map?”

## Are We Going To Make It?

### Short answer

Yes, probably.

But not by continuing to hope that the current static online update rule will somehow become teacher-beating on its own.

### Why I think the answer is still yes

There are several reasons to remain optimistic.

#### 1. The problem has become narrow

We are no longer debugging everything at once.

The current evidence strongly suggests:

- representation is not the main blocker anymore
- evaluation is not the main blocker anymore
- warm-start wiring is not the main blocker anymore

That means the remaining problem is much more specific:

- how teacher data is used during online improvement

That is a solvable engineering and research problem.

#### 2. The literature says this exact pattern is solvable

The current failure mode is common:

- good teacher
- online learner ties or trails teacher
- unconstrained RL drifts

Other works get past this by using:

- stronger rehearsal of teacher data
- scheduled guidance, not static guidance
- targeted replay, not uniform replay
- behavior-regularized improvement
- disagreement-focused aggregation

So the shape of the problem is not mysterious anymore.

#### 3. We already saw the first positive signal from the new plan

The short scheduled-replay run hitting `0.95` matters.

That is not the final answer, but it shows:

- the plan in `plan-wagmi.md` is not nonsense
- structural changes to replay usage can move the metric in the right direction

That is enough to justify further work.

### Why I do not think the current branch alone is enough

Because the longer runs still show:

- early best checkpoint
- then flatness or drift

That means this branch currently looks like:

- a preservation mechanism

not yet:

- a true improver

So if the question is:

- “Will this exact large run beat the teacher?”

the answer is:

- probably not

If the question is:

- “Will this repo eventually produce a teacher-beating learner if we keep following the current disciplined plan?”

the answer is:

- yes, that still looks plausible

## What The Current Run Means Strategically

The current large run is still worth finishing because it validates something important:

- whether scheduled replay simply delays collapse
- or whether it creates a branch that remains near the teacher for long enough to support the next improvement

Even if it never beats `0.9375`, that is still useful:

- it would mean scheduled replay belongs in the mainline
- and that the next missing ingredient is probably **prioritized replay** or **targeted teacher support**

## What We Should Believe Right Now

We should believe:

- the project is not doomed
- the current branch is not yet enough
- the current branch is the first one that deserves to be built on rather than thrown away

We should not believe:

- that reward is telling us the truth
- that more scale alone will solve it
- that bigger models alone will solve it
- that the world model by itself is the improver

## Next Likely Winning Direction

If this large run finishes without beating the teacher, the most justified next move is:

1. keep scheduled replay
2. add replay targeting
3. stop using uniform replay as the only replay regime

That means:

- prioritized replay over disagreement states
- replay emphasis on weak-action states
- possibly mixed replay composition

That is exactly the path already laid out in [plan-wagmi.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/misc/plan-wagmi.md).

## Bottom Line

Are we going to make it?

- **Yes, likely, if we keep following the disciplined loop**

Did we already make it?

- **No**

Did the latest run prove something useful?

- **Yes**

It proved that:

- the repo can now support structurally correct online teacher-guided RL experiments
- scheduled replay is a real lever
- but uniform scheduled replay still looks like a stabilizer, not yet a teacher-beating improver

That is progress. It is not victory. But it is the kind of progress that usually comes right before the next real algorithmic step matters.
