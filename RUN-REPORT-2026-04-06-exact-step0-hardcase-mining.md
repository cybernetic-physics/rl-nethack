## Purpose

Test whether the remaining `0.9875 -> 1.0` teacher gap is a real local-geometry data hole, and if so, whether mining off-heldout examples from that geometry is enough to beat the current baseline.

Baseline before this run:

- teacher artifact: `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- held-out trace match: `0.9875`
- remaining mismatch:
  - one `east -> south` disagreement on held-out row `seed=202 step=0`

## Hypothesis

The remaining miss is caused by missing slice coverage, not by model size alone.

More concretely:

- the training data under-covers the local geometry
  - `north=monster_*`
  - `south=floor`
  - `east=monster_*`
  - `west=floor`
- if we mine off-heldout rows from that exact geometry and train on them, the teacher should change on the held-out miss

## Code Paths Touched

- [cli.py](/home/luc/rl-nethack/cli.py)
- [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## Infrastructure Added

Extended `rl-shard-traces` / `shard_trace_file(...)` with `--adjacent-signature` filtering.

Example:

```bash
uv run python cli.py rl-shard-traces \
  --input traces.jsonl \
  --output exact.jsonl \
  --adjacent-signature 'north=monster_*,south=floor,east=monster_*,west=floor'
```

This keeps full episodes that contain at least one row matching the requested local geometry.

## Validation

Targeted regression:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k 'shard_trace_file_can_filter_by_adjacent_signature or shard_trace_file_can_filter_by_teacher_action or shard_trace_file_preserves_full_episodes'
```

Result:

- `3 passed`

Full scaffold suite:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- see final validation line from this cycle

## Offline Diagnostics

### 1. Existing data really misses the held-out geometry

Earlier diagnosis held up:

- `train200` had zero `north monster + east monster + south floor + west floor` rows
- `train680` and `train1400` also had zero exact rows
- the only `north monster + east monster` rows in training were a different corridor geometry and all were labeled `south`

### 2. Off-heldout reset mining found the exact geometry

A reset-only scan over new seeds found exact step-0 matches off-heldout.

Then a hardened miner over seeds `300..4999` collected exact off-heldout rows into:

- `/tmp/x100_v4_exact_monster_step0_mined.jsonl`

Summary:

- rows mined: `3`
- teacher actions:
  - `east`: `1`
  - `south`: `2`
- reset errors:
  - `3`

Example mined rows:

- `seed=551`: `north=monster_f south=floor east=monster_F west=floor` -> teacher `east`
- `seed=1251`: `north=monster_F south=floor east=monster_f west=floor` -> teacher `south`
- `seed=3980`: `north=monster_F south=floor east=monster_d west=floor` -> teacher `south`

Important note:

- repeated raw NLE resets still showed instability:
  - `RuntimeError: NetHack done right after reset`
- the miner was hardened by recreating the env periodically and on reset failure

## Augmented Training Sets

Built from:

- base: `/tmp/x100_v4_train_traces.jsonl` (`200` rows)
- mined exact slice: `/tmp/x100_v4_exact_monster_step0_mined.jsonl`

Derived datasets:

- `/tmp/x100_v4_train_exacteast_x8.jsonl`
- `/tmp/x100_v4_train_exacteast_x16.jsonl`
- `/tmp/x100_v4_train_exacteast_x32.jsonl`
- `/tmp/x100_v4_train_exacteast_x64.jsonl`
- `/tmp/x100_v4_train_exactall_x16.jsonl`

## Exact Commands Run

Pure supervised, exact-east oversample `x64`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_train_exacteast_x64.jsonl \
  --output /tmp/x100_v4_exacteast_x64_sup_l3_h1024.pt \
  --epochs 80 --lr 0.0005 --hidden-size 1024 --num-layers 3 \
  --observation-version v4 \
  --supervised-loss-coef 1.0 \
  --device cuda \
  --select-by-heldout \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

Mixed distill + supervision, exact-east oversample `x64`:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_train_exacteast_x64.jsonl \
  --output /tmp/x100_v4_exacteast_x64_mix_l3_h1024.pt \
  --epochs 80 --lr 0.0005 --hidden-size 1024 --num-layers 3 \
  --observation-version v4 \
  --distill-teacher-bc-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --distill-loss-coef 0.2 --distill-temperature 2.0 \
  --supervised-loss-coef 1.0 \
  --device cuda \
  --select-by-heldout \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

Pure supervised, lighter exact-east oversampling:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_train_exacteast_x8.jsonl \
  --output /tmp/x100_v4_exacteast_x8_sup_l3_h1024.pt \
  --epochs 80 --lr 0.0005 --hidden-size 1024 --num-layers 3 \
  --observation-version v4 \
  --supervised-loss-coef 1.0 \
  --device cuda \
  --select-by-heldout \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

```bash
CUDA_VISIBLE_DEVICES=1 uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_train_exacteast_x16.jsonl \
  --output /tmp/x100_v4_exacteast_x16_sup_l3_h1024.pt \
  --epochs 80 --lr 0.0005 --hidden-size 1024 --num-layers 3 \
  --observation-version v4 \
  --supervised-loss-coef 1.0 \
  --device cuda \
  --select-by-heldout \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_train_exacteast_x32.jsonl \
  --output /tmp/x100_v4_exacteast_x32_sup_l3_h1024.pt \
  --epochs 80 --lr 0.0005 --hidden-size 1024 --num-layers 3 \
  --observation-version v4 \
  --supervised-loss-coef 1.0 \
  --device cuda \
  --select-by-heldout \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

## Primary Results

Baseline:

- `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- held-out trace match: `0.9875`
- mismatch: `east -> south`

Exact-east `x64`, pure supervised:

- best held-out trace match: `0.9875`
- mismatch moved to:
  - `east -> west`

Exact-east `x64`, mixed distill + supervision:

- best held-out trace match: `0.9875`
- mismatch stayed:
  - `east -> south`

Exact-east pure supervised with lighter oversampling:

- `x8`: `0.9875`
- `x16`: `0.9875`
- `x32`: `0.9875`

Common failure for `x8/x16/x32`:

- original `east` row was fixed
- a different row regressed:
  - `south -> north`

So the stable pattern is:

- the mined exact slice is strong enough to correct the original miss
- but the slice is too narrow to improve the total benchmark

## Interpretation

What held up:

- the remaining gap is a real data-coverage issue
- the exact held-out-style geometry exists off-heldout
- the teacher does not always choose `south` in that geometry
- adding even one off-heldout `east` example can move the model on the benchmarked hard case

What did not hold up:

- a tiny exact-slice augmentation is not enough to beat `0.9875`
- simply oversampling the one `east` example trades one benchmark error for another
- mixed distillation against the current best teacher preserves too much of the old mistake

The strongest read is:

- the local hard-case mining direction is real
- but the current mined slice is too small and too imbalanced
- the next version should scale breadth, not just oversampling weight

## Recommended Next Move

Keep the new adjacency-slice tooling.

Do not promote any of these augmented teachers as a new baseline.

Next correct step:

- mine a broader off-heldout slice around the same failure family
  - exact geometry plus nearby variants
  - more seeds
  - more than one positive `east` example
- then retrain with held-out selection

The important update to belief is:

- the repo is not just “missing scale”
- it is missing specific teacher coverage for a narrow but benchmark-relevant local regime
