# Upgrade Plan

This document is a grounded plan to upgrade the current repo from a working
but weak hybrid SFT/BC/APPO stack into a stronger NetHack control system.

It is based on direct review of the current code as of 2026-04-05, not on an
idealized design.


## 1. Current System Review

The repo currently has four distinct learning/control layers:

1. forward-model SFT in [train.py](/home/luc/rl-nethack/train.py),
2. heuristic closed-loop control and evaluation in
   [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py),
3. trace-driven BC in [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py),
4. low-level RL through Sample Factory APPO in
   [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py) and
   [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py).

That means the repo is no longer “just SFT”, but it also does not yet have a
coherent hierarchy where:

- the SFT model helps control directly,
- BC initializes RL,
- skills are truly learned options,
- and reward learning replaces heuristic shaping in a principled way.


## 2. What The Code Actually Does Today

### 2.1 Forward model

[train.py](/home/luc/rl-nethack/train.py) trains a LoRA adapter with Unsloth on
ShareGPT-format conversations. It does not train a policy. It trains a
language model to predict deltas / outcomes from prompt-formatted state-action
examples.

Important current facts:

- this path now supports multi-GPU `torchrun`,
- the recent DDP bug was fixed by pinning each rank to its CUDA device before
  model load,
- but this model still has no direct weight bridge into APPO.

So the forward model is useful as:

- a world-model research path,
- a teacher for trace generation,
- a planning component later,

but not yet as the backbone of the low-level control policy.


### 2.2 Task harness

[src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py) is the current
best controller in practice.

It works by:

- enumerating a small candidate action set per state,
- forking the live env with `os.fork()`,
- rolling each candidate action one step,
- scoring each branch with
  [src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py),
- choosing the highest-scoring action.

This is very important because it makes the current teacher:

- high quality relative to the other learned policies,
- slow and expensive,
- one-step greedy,
- deterministic and debuggable.

This controller is not scalable as a production policy, but it is a very good
oracle/teacher for:

- multi-turn trace generation,
- reward preference generation,
- policy regression tests.


### 2.3 Current RL env path

[rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py) wraps
[rl/env_adapter.py](/home/luc/rl-nethack/rl/env_adapter.py), which wraps a real
`nle.env.NLE()` instance.

The env currently:

- tracks memory with `MemoryTracker`,
- switches among skills with a scheduler,
- computes shaped reward via [rl/rewards.py](/home/luc/rl-nethack/rl/rewards.py),
- exposes a `Discrete(13)` action space,
- provides a compact vector observation.

This is a real RL environment. The problem is not that RL is fake. The problem
is that the environment interface is still too lossy and too hand-built.


### 2.4 Observation encoder

[rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py) is the
current bottleneck.

The policy sees:

- 12 scalar features,
- 4 adjacent-tile one-hot vectors,
- active skill one-hot,
- allowed-action mask.

Total dimension: `106`.

This is much too thin for a long-horizon game like NetHack.

What is missing:

- wider local map structure,
- a notion of frontiers,
- repeated-state / repeated-action counters,
- recent action history,
- richer monster geometry,
- richer inventory context,
- stronger memory summaries.

This is the most likely reason BC and APPO collapse into directional repetition.


### 2.5 BC path

[rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py) trains a simple MLP from
trace feature vectors to primitive actions.

The BC model in [rl/bc_model.py](/home/luc/rl-nethack/rl/bc_model.py) is:

- 2 hidden layers,
- ReLU MLP,
- masked at train and inference time with `allowed_actions`.

This path is primitive, but it already outperforms APPO-from-scratch on the
current `explore` benchmark. That is a strong signal:

- the repo has a supervision path that already works better than RL,
- so RL should be built as policy improvement on top of BC, not from scratch.


### 2.6 Reward learning

[rl/train_reward_model.py](/home/luc/rl-nethack/rl/train_reward_model.py)
trains a preference model from pairs extracted out of the task harness.

Important limitation:

- the current preference data is generated from the same one-step heuristic
  teacher,
- and the reward model in [rl/reward_model.py](/home/luc/rl-nethack/rl/reward_model.py)
  is a tiny MLP over hand-engineered one-step features.

So “learned reward” currently means:

- distilling the heuristic reward ranking into another small model,
- not learning a rich long-horizon task objective from trajectories.

This is useful, but still shallow.


### 2.7 Scheduler

[rl/scheduler.py](/home/luc/rl-nethack/rl/scheduler.py) and
[rl/options.py](/home/luc/rl-nethack/rl/options.py) define the skill layer.

What exists:

- `explore`, `survive`, `combat`, `descend`, `resource`
- heuristic `can_start`
- heuristic `should_stop`
- heuristic allowed-action sets
- rule-based or learned scheduler selection

What does not exist yet:

- learned option initiation,
- learned option termination,
- skill-specific low-level policies,
- hierarchical credit assignment,
- training that respects option boundaries as first-class rollout units.

The current “options” layer is therefore a thin control shell, not yet a true
hierarchical RL system.


### 2.8 APPO backend

[rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py) correctly launches Sample
Factory APPO. This part is real and functioning.

The current APPO learner is still generic:

- Sample Factory builds an MLP+GRU actor-critic from the observation space,
- it does not use a custom skill-conditioned torso yet,
- it does not initialize from BC,
- it does not use a richer map encoder.

This means the current APPO setup is operational but structurally underpowered.


## 3. Main Problems To Solve

These are the actual repo problems in priority order.

### Problem A: No BC-to-APPO bridge

The biggest miss is in [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py):

- there is no path to initialize the APPO actor-critic from a BC checkpoint,
- so APPO always starts from scratch,
- even though BC already beats APPO on the same task.

This should be the first upgrade.


### Problem B: Observation is too weak

The `106`-dim observation in [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)
throws away too much structure.

Current failure symptom:

- BC and APPO collapse to “go east” / “go west” style repetitive behaviors,
- because the observation is too local and too aliased.

This is the second most important upgrade.


### Problem C: Trace generation is expensive

[rl/traces.py](/home/luc/rl-nethack/rl/traces.py) calls
[src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py), which forks the
env and scores each candidate action one step ahead.

That means the best teacher data path is:

- high quality,
- but too slow to scale cheaply.

This needs a staged replacement, not a blind rewrite.


### Problem D: Reward learning is too local

The learned reward path is still trained from one-step teacher comparisons
extracted from the same heuristic.

So it does not yet solve:

- long-horizon exploration quality,
- loop aversion beyond local proxies,
- skill-level success semantics.


### Problem E: Skills are still shallow

[rl/options.py](/home/luc/rl-nethack/rl/options.py) is useful, but it is still
mostly:

- action masks,
- start/stop heuristics,
- directives on paper.

It is not yet an option-centric training setup.


## 4. Upgrade Strategy

The right strategy is not “more APPO tuning.”

The right strategy is:

1. strengthen the teacher path,
2. strengthen the observation path,
3. bridge BC into RL,
4. only then tune RL harder,
5. only after that deepen the hierarchy.


## 5. Concrete Upgrade Plan

### Phase 1: Make BC the canonical bootstrap

Goal:

- turn BC into the standard starting point for low-level policy learning.

Changes:

1. Add checkpoint import / warm-start support to APPO.
2. Add a CLI flag to `rl-train-appo`, for example:
   - `--bc-init path/to/bc.pt`
3. Map BC MLP layers into the APPO encoder/policy head when architectures
   match.
4. If exact parameter mapping is messy, implement a supervised pretrain stage
   inside the APPO model code before RL starts.

Why:

- this directly targets the current failure that APPO-from-scratch is worse
  than BC.

Files to extend:

- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/bc_model.py](/home/luc/rl-nethack/rl/bc_model.py)
- likely a new bridge module:
  - `rl/bc_init.py`

Success criterion:

- APPO initialized from BC should match or nearly match BC before any RL
  updates.


### Phase 2: Replace the observation vector

Goal:

- make the control policy observe enough structure to distinguish good
  exploration from repetitive local motion.

Changes:

1. Add a richer encoder in a new file, for example:
   - `rl/feature_encoder_v2.py`
2. Keep the current encoder for backward compatibility.
3. Include:
   - local map patch around the player,
   - frontier indicators,
   - recent action history,
   - repeated-state and repeated-position counters,
   - richer monster/item summaries,
   - stance on stairs / door / corridor topology,
   - current skill and steps-in-skill,
   - action mask.
4. Preserve a compact MLP-friendly path first.
5. Only later consider a CNN or transformer encoder over local tiles.

Why:

- current representation is the main source of aliasing.

Files to extend:

- [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)
- [rl/env_adapter.py](/home/luc/rl-nethack/rl/env_adapter.py)
- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)

Success criterion:

- BC trained on the richer features should beat the current BC baseline on the
  same fixed trace set.


### Phase 3: Build a fixed regression suite

Goal:

- stop debugging by intuition.

Changes:

1. Add a fixed evaluation suite file, for example:
   - `data/eval/explore_regression_seeds.json`
2. Add a report command that always compares:
   - `task_greedy`
   - BC
   - APPO
3. Report:
   - shaped reward,
   - unique tiles,
   - rooms discovered,
   - repeated action rate,
   - invalid action rate,
   - action histogram.
4. Save outputs as JSON for diffable regression checks.

Why:

- current evaluation exists, but the comparison workflow is still manual.

Files to extend:

- [rl/evaluate.py](/home/luc/rl-nethack/rl/evaluate.py)
- [rl/evaluate_bc.py](/home/luc/rl-nethack/rl/evaluate_bc.py)
- [cli.py](/home/luc/rl-nethack/cli.py)

Success criterion:

- every policy experiment produces the same comparable metrics automatically.


### Phase 4: Scale multi-turn teacher traces

Goal:

- produce enough trace data for BC without relying exclusively on slow
  counterfactual rollouts.

Changes:

1. Keep `task_greedy` traces as the high-quality teacher set.
2. Add a second cheaper teacher:
   - policy rollout with occasional one-step correction from `task_greedy`.
3. Add trace deduplication and filtering:
   - drop excessive repeated-action segments,
   - tag skill / context / episode summary,
   - measure action balance.
4. Split trace generation into:
   - `gold` traces from `task_greedy`,
   - `silver` traces from cheaper policies with verification.

Why:

- the current generator in [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
  is too slow to be the only path.

Files to extend:

- [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
- maybe add:
  - `rl/trace_filters.py`

Success criterion:

- produce a large trace corpus without making the teacher path intractable.


### Phase 5: Make reward learning more honest

Goal:

- stop pretending one-step heuristic distillation is enough.

Changes:

1. Move reward preference generation from one-step branch comparisons toward
   short segment comparisons.
2. Compare 3-8 step trace fragments instead of only one-step deltas.
3. Add explicit labels for:
   - looping,
   - frontier gain,
   - survival under threat,
   - stairs approach / stairs use.
4. Keep the current MLP reward model as a baseline.
5. Add a trajectory-fragment reward model later if needed.

Why:

- current reward learning in [rl/train_reward_model.py](/home/luc/rl-nethack/rl/train_reward_model.py)
  is still too local to guide long-horizon improvement.

Files to extend:

- [rl/train_reward_model.py](/home/luc/rl-nethack/rl/train_reward_model.py)
- [rl/reward_model.py](/home/luc/rl-nethack/rl/reward_model.py)

Success criterion:

- learned reward ranking matches human/common-sense ordering on held-out trace
  fragments better than the current one-step model.


### Phase 6: Turn skills into real training units

Goal:

- move from “skill labels and action masks” to actual options.

Changes:

1. Promote option boundaries to first-class rollout annotations.
2. Track:
   - option start,
   - option stop,
   - option success/failure,
   - option return.
3. Add skill-specific success metrics:
   - `explore`: frontier / rooms / stairs found
   - `survive`: alive, low HP recovery, avoided damage
   - `combat`: threat reduction at acceptable HP cost
   - `descend`: stairs reached and used
   - `resource`: valid pickup/use
4. Train the scheduler on option outcomes, not just rule imitation.

Why:

- current scheduler learning is only imitating the rule-based policy in
  [rl/train_scheduler.py](/home/luc/rl-nethack/rl/train_scheduler.py).

Files to extend:

- [rl/options.py](/home/luc/rl-nethack/rl/options.py)
- [rl/scheduler.py](/home/luc/rl-nethack/rl/scheduler.py)
- [rl/train_scheduler.py](/home/luc/rl-nethack/rl/train_scheduler.py)
- [rl/env_adapter.py](/home/luc/rl-nethack/rl/env_adapter.py)

Success criterion:

- skill transitions and skill quality become measurable and learnable.


### Phase 7: Upgrade the APPO model path

Goal:

- make the low-level learner match the repo’s intended architecture.

Changes:

1. Replace the implicit generic Sample Factory torso with a real custom
   skill-conditioned encoder.
2. Keep the input/output interfaces stable.
3. Use:
   - richer observation encoder,
   - explicit skill embedding,
   - shared trunk,
   - recurrent core,
   - policy/value heads.
4. Optionally add an auxiliary BC loss or imitation loss during RL.

Why:

- [rl/model.py](/home/luc/rl-nethack/rl/model.py) currently only stores a spec;
  the actual learner still uses Sample Factory defaults.

Files to extend:

- [rl/model.py](/home/luc/rl-nethack/rl/model.py)
- potentially a new custom Sample Factory model wrapper

Success criterion:

- the learner is no longer bottlenecked by the generic MLP+GRU default.


## 6. Debugging Plan

The repo needs a strict debugging ladder.

### Debug Step 1: Single-episode exact replay

Use one tiny `explore` trace.

Targets:

- BC should memorize it,
- replay should match action-by-action,
- closed-loop rollout should remain aligned.

If not, the bug is in:

- features,
- trace labels,
- policy implementation,
- or evaluation.


### Debug Step 2: Small fixed dataset

Use `10-20` episodes only.

Measure:

- BC train accuracy,
- BC closed-loop quality on train seeds,
- BC closed-loop quality on held-out seeds.

If BC fails here, do not train RL yet.


### Debug Step 3: Oracle gap

For the same states, log:

- `task_greedy` action,
- BC action,
- APPO action,
- allowed actions,
- repeated-state flags,
- key feature slices.

This will tell us whether the failure is:

- representation,
- optimization,
- or reward.


### Debug Step 4: Action-space ablation

For `explore`, run:

- full action space,
- narrowed action space.

If the narrowed space improves BC or APPO sharply, the old action set is
hurting learning.


### Debug Step 5: Reward-rank sanity

For sampled states, print:

- heuristic task reward by candidate action,
- learned reward score by candidate action.

If learned reward prefers obviously bad actions, stop RL tuning and fix the
reward model first.


### Debug Step 6: BC-initialized RL retention

After BC initialization is added:

1. evaluate the BC checkpoint,
2. initialize APPO from BC,
3. evaluate before any RL update,
4. evaluate after short RL,
5. evaluate after longer RL.

If performance drops immediately, the RL objective or optimizer dynamics are
destroying competent behavior.


## 7. Recommended Execution Order

This is the actual order I recommend implementing.

1. Add BC-to-APPO initialization.
2. Add fixed regression evaluation comparing `task_greedy`, BC, and APPO.
3. Replace the observation encoder with a richer v2 encoder.
4. Rerun BC on the same trace set and verify improvement.
5. Scale teacher traces with a gold/silver split.
6. Only then rerun APPO from BC initialization.
7. After low-level control improves, deepen reward learning and option logic.


## 8. Non-Goals Right Now

These are tempting, but not the next best move.

- More APPO hyperparameter sweeps before adding BC initialization.
- Direct SFT-to-APPO weight loading.
- Bigger reward models before improving the teacher and observation paths.
- More serving infrastructure work.
- Jumping straight to GRPO for low-level control.


## 9. Bottom Line

The repo is already strong enough to support meaningful improvement work.

The highest-value next change is not more RL scale. It is to connect the parts
that already work:

- `task_greedy` as the best teacher,
- multi-turn traces as the supervision format,
- BC as the best current learned controller,
- APPO as the improvement stage after BC initialization.

The shortest path to a better system is:

1. better traces,
2. better observations,
3. BC-first training,
4. BC-initialized APPO,
5. option and reward upgrades after the low-level controller stops collapsing.
