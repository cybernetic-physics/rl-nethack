# Run Report: v2 BC + Warm-Start APPO

Date: 2026-04-05

This report documents a larger validation run after the `34ca262` upgrade
slice, which added:

- BC-to-APPO warm start support,
- `v2` observation support,
- compact policy comparison tooling,
- and the repo-local [UPGRADE-PLAN.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/plans/UPGRADE-PLAN.md).


## Goal

Validate the upgraded RL path end to end:

1. generate multi-turn `v2` traces,
2. train a `v2` BC model with hidden size aligned to APPO,
3. warm-start a larger APPO run from that BC checkpoint,
4. evaluate the resulting policies,
5. debug any blocking issues encountered during the run.


## Code Issues Found And Fixed During This Run

### 1. Trace generation was still emitting `v1` feature vectors

Problem:

- [rl/traces.py](/home/luc/rl-nethack/rl/traces.py) still hardcoded
  `encode_observation(...)` with the default version.
- That meant the new `v2` BC/APPO path could not actually be exercised
  end-to-end.

Fix:

- added `observation_version` to `generate_multi_turn_traces(...)`
- stored `observation_version` in each trace row
- exposed `--observation-version` on `cli.py rl-generate-traces`

Result:

- the trace artifact used in this run is explicitly `v2`


## Pipeline Run

### 1. Generate `v2` multi-turn traces

Command used:

```bash
uv run python cli.py rl-generate-traces \
  --output data/run_v2_explore_traces.jsonl \
  --num-episodes 20 \
  --max-steps 20 \
  --task explore \
  --policy task_greedy \
  --observation-version v2
```

Important note:

- the `task_greedy` teacher is still slow because it uses the counterfactual
  fork-per-action harness.
- to keep the run moving, I stopped the process once the artifact had become
  large enough to train on cleanly.

Verified artifact:

- [data/run_v2_explore_traces.jsonl](/home/luc/rl-nethack/data/run_v2_explore_traces.jsonl)

Verified stats:

- episodes: `14`
- rows: `273`
- max steps in episode: `20`
- avg steps per episode: `19.5`
- all episodes multi-turn: `true`
- observation version stored in rows: `v2`


### 2. Train aligned `v2` BC model

Command used:

```bash
uv run python cli.py rl-train-bc \
  --input data/run_v2_explore_traces.jsonl \
  --output output/run_v2_explore_bc.pt \
  --epochs 40 \
  --lr 0.001 \
  --hidden-size 512 \
  --observation-version v2
```

Why `hidden-size 512`:

- this matches the first two APPO encoder layers much more closely than the old
  `256`-hidden BC models,
- so the BC warm-start checkpoint copy is more meaningful.

Result:

- num examples: `273`
- input dim: `160`
- hidden size: `512`
- final loss: `0.8032`
- train accuracy: `0.6630`

Artifact:

- [output/run_v2_explore_bc.pt](/home/luc/rl-nethack/output/run_v2_explore_bc.pt)


### 3. Run larger warm-started APPO training

Command used:

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-appo \
  --experiment appo_v2_bc_large \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --recurrence 16 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 20000 \
  --enabled-skills explore \
  --observation-version v2 \
  --bc-init-path output/run_v2_explore_bc.pt
```

What was validated here:

- `v2` observation space used by Sample Factory:
  - `obs` shape `(160,)`
- BC warm-start checkpoint was created and loaded:
  - initial checkpoint:
    [train_dir/rl/appo_v2_bc_large/checkpoint_p0/checkpoint_000000000_0.pth](/home/luc/rl-nethack/train_dir/rl/appo_v2_bc_large/checkpoint_p0/checkpoint_000000000_0.pth)
- training completed successfully

Final training stats:

- collected frames: `21,504`
- final FPS: `174.7`
- final saved checkpoint:
  [train_dir/rl/appo_v2_bc_large/checkpoint_p0/checkpoint_000000021_21504.pth](/home/luc/rl-nethack/train_dir/rl/appo_v2_bc_large/checkpoint_p0/checkpoint_000000021_21504.pth)

Important training observation:

- the run started better than old APPO-from-scratch runs
- mid-training shaped episode return briefly went positive
- but the policy still drifted back toward a collapsed repetitive regime by the
  end


## Evaluation Results

### Stable conclusions

These conclusions are reliable:

1. the new end-to-end `v2` path runs successfully
2. BC warm-start is operational and does influence APPO training dynamics
3. final APPO policy is still action-collapsed
4. strict action masking is still preventing invalid-action requests
5. policy evaluation is currently not stable enough to trust a single numeric
   result


### Snapshot comparison result

One compact comparison snapshot was written to:

- [output/run_v2_compare.json](/home/luc/rl-nethack/output/run_v2_compare.json)

That snapshot reported:

- `task_greedy`
  - avg task reward: `4.55`
  - avg unique tiles: `70.67`
  - repeated action rate: `0.0167`
- `bc`
  - avg unique tiles: `51.0`
  - avg env reward: `0.6667`
- `appo`
  - avg task reward: `-9.75`
  - avg unique tiles: `71.67`
  - repeated action rate: `0.7833`

Interpretation of that snapshot:

- APPO still underperforms `task_greedy` on shaped reward
- BC is still simpler but less loop-collapsed
- APPO can cover ground, but does so with highly repetitive behavior


## Critical New Bug: Evaluation Nondeterminism

This run uncovered a major issue:

- repeated evaluation of the **same model** on the **same seeds** produced
  materially different summaries.

This happened for both:

- BC evaluation via [rl/evaluate_bc.py](/home/luc/rl-nethack/rl/evaluate_bc.py)
- APPO evaluation via [rl/evaluate.py](/home/luc/rl-nethack/rl/evaluate.py)

Examples observed during this run:

### BC repeated evals differed substantially

For the same command:

```bash
uv run python cli.py rl-evaluate-bc \
  --model output/run_v2_explore_bc.pt \
  --task explore \
  --seeds 42,43,44 \
  --max-steps 20
```

I observed summaries ranging from roughly:

- avg unique tiles `41.67`
- avg unique tiles `50.67`
- avg unique tiles `71.0`

with different action histograms each time.

### APPO repeated evals also differed

For the same command:

```bash
uv run python cli.py rl-evaluate-appo \
  --experiment appo_v2_bc_large \
  --seeds 42,43,44 \
  --max-steps 20
```

I observed summaries ranging from roughly:

- avg task reward `-12.23`, avg unique tiles `46.0`
- avg task reward `-9.28`, avg unique tiles `78.67`
- avg task reward `15.1`, avg unique tiles `105.0`

This is too large to treat as noise.


## What This Means

This nondeterminism bug is now one of the highest-priority issues in the repo.

It means:

- we can trust that training completed,
- we can trust that warm-start loading worked,
- but we cannot yet trust fine-grained evaluation deltas between policies.

That does **not** make the run useless.

It still proves:

- the upgraded pipeline works technically,
- the `v2` observation path is live,
- BC warm-start is correctly connected,
- and the remaining bottleneck is now partly an **evaluation correctness**
  problem, not just a learning problem.


## Current Best Read

The current control ranking is still:

1. `task_greedy` is the strongest trustworthy controller
2. BC is the best learned controller baseline
3. warm-started APPO is operational but still not reliably better than BC

The warm-start path is still worth keeping because:

- it removed the “APPO always starts from scratch” problem,
- it produced better early training behavior than the old APPO runs,
- and it is the right bridge to keep improving.

But the repo is not yet at the point where “large warm-start APPO” is clearly
uplifting quality over BC.


## Recommended Next Steps

1. Debug evaluation determinism before running more policy comparisons.
2. Audit the eval stack for process-global or env-global nondeterminism.
3. Once eval is stable, rerun:
   - `task_greedy`
   - `v2` BC
   - `v2` BC-warm-start APPO
4. Only then decide whether the next bottleneck is:
   - reward shaping,
   - observation quality,
   - or RL optimization dynamics.
