## Purpose

Test whether the remaining `0.9875 -> 1.0` teacher gap can be closed by broader off-heldout reset-slice mining, then by an exact-species oversampling probe for the final held-out miss.

Baseline before this run:

- teacher artifact: `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- held-out trace match: `0.9875`
- remaining mismatch:
  - one `east -> south` disagreement on held-out row `seed=202 step=0`
  - prompt:
    - `Adjacent: north=monster_f south=floor east=monster_o west=floor`

## Hypothesis

The earlier exact-slice augmentation was too narrow.

More concretely:

- if we mine a much broader off-heldout reset slice for the held-out local geometry
  - `north=monster_*`
  - `south=floor`
  - `east=monster_*`
  - `west=floor`
- then train held-out-selected teachers on that wider slice,
- the repo should either:
  - beat `0.9875`, or
  - show that the remaining gap depends on context not captured by the loose adjacency signature

Secondary hypothesis:

- if the true missing signal is the exact monster-species pair
  - `north=monster_f`
  - `east=monster_o`
- then even a tiny exact-species oversample should fix the last row cleanly

## Code Paths Touched

- [cli.py](/home/luc/rl-nethack/cli.py)
- [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## Infrastructure Added

Added a reset-slice mining path that scans fresh NLE resets and writes teacher-labeled step-0 trace rows directly:

```bash
uv run python cli.py rl-mine-reset-slice \
  --output out.jsonl \
  --seed-start 300 \
  --num-seeds 30000 \
  --task explore \
  --observation-version v4 \
  --adjacent-signature 'north=monster_*,south=floor,east=monster_*,west=floor'
```

This path is stricter and more useful than episode-level sharding for rare step-0 geometries:

- it mines directly from raw resets,
- it records teacher-labeled rows only when the requested signature matches,
- it survives reset instability by recreating the env periodically and on failure,
- and it makes rare hard-case slices cheap to collect.

## Validation

Full scaffold suite:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `76 passed`

## Exact Commands Run

### 1. Broader reset-slice mining

```bash
uv run python cli.py rl-mine-reset-slice \
  --output /tmp/x100_v4_exact_monster_step0_mined_30k.jsonl \
  --seed-start 300 \
  --num-seeds 30000 \
  --task explore \
  --observation-version v4 \
  --adjacent-signature 'north=monster_*,south=floor,east=monster_*,west=floor' \
  --recreate-every 250
```

### 2. Broader-slice teacher probes

Pure supervised, broader slice `x4`:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_train_exact30k_all_x4.jsonl \
  --output /tmp/x100_v4_exact30k_all_x4_sup_l3_h1024.pt \
  --epochs 80 --lr 0.0005 --hidden-size 1024 --num-layers 3 \
  --observation-version v4 \
  --supervised-loss-coef 1.0 \
  --device cuda \
  --select-by-heldout \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl \
  --teacher-report-output /tmp/x100_v4_exact30k_all_x4_sup_l3_h1024.teacher_report.json
```

Pure supervised, broader slice `x8`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_train_exact30k_all_x8.jsonl \
  --output /tmp/x100_v4_exact30k_all_x8_sup_l3_h1024.pt \
  --epochs 80 --lr 0.0005 --hidden-size 1024 --num-layers 3 \
  --observation-version v4 \
  --supervised-loss-coef 1.0 \
  --device cuda \
  --select-by-heldout \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl \
  --teacher-report-output /tmp/x100_v4_exact30k_all_x8_sup_l3_h1024.teacher_report.json
```

Mixed distill + supervision:

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_train_exact30k_allx4_eastx4.jsonl \
  --output /tmp/x100_v4_exact30k_allx4_eastx4_mix_l3_h1024.pt \
  --epochs 80 --lr 0.0005 --hidden-size 1024 --num-layers 3 \
  --observation-version v4 \
  --distill-teacher-bc-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --distill-loss-coef 0.2 --distill-temperature 2.0 \
  --supervised-loss-coef 1.0 \
  --device cuda \
  --select-by-heldout \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl \
  --teacher-report-output /tmp/x100_v4_exact30k_allx4_eastx4_mix_l3_h1024.teacher_report.json
```

### 3. Exact-species mining and probes

Mine the exact held-out species pair:

```bash
uv run python cli.py rl-mine-reset-slice \
  --output /tmp/x100_v4_exact_f_o_step0_mined_30k.jsonl \
  --seed-start 300 \
  --num-seeds 30000 \
  --task explore \
  --observation-version v4 \
  --adjacent-signature 'north=monster_f,south=floor,east=monster_o,west=floor' \
  --recreate-every 250
```

Pure supervised, exact-species oversample:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_train_exactfo_x64.jsonl \
  --output /tmp/x100_v4_exactfo_x64_sup_l3_h1024.pt \
  --epochs 80 --lr 0.0005 --hidden-size 1024 --num-layers 3 \
  --observation-version v4 \
  --supervised-loss-coef 1.0 \
  --device cuda \
  --select-by-heldout \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl \
  --teacher-report-output /tmp/x100_v4_exactfo_x64_sup_l3_h1024.teacher_report.json
```

Mixed distill + exact-species oversample:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_train_exactfo_x64.jsonl \
  --output /tmp/x100_v4_exactfo_x64_mix_l3_h1024.pt \
  --epochs 80 --lr 0.0005 --hidden-size 1024 --num-layers 3 \
  --observation-version v4 \
  --distill-teacher-bc-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --distill-loss-coef 0.2 --distill-temperature 2.0 \
  --supervised-loss-coef 1.0 \
  --device cuda \
  --select-by-heldout \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl \
  --teacher-report-output /tmp/x100_v4_exactfo_x64_mix_l3_h1024.teacher_report.json
```

## Artifacts

Mined reset slices:

- `/tmp/x100_v4_exact_monster_step0_mined_30k.jsonl`
- `/tmp/x100_v4_exact_f_o_step0_mined_30k.jsonl`

Teacher probes:

- `/tmp/x100_v4_exact30k_all_x4_sup_l3_h1024.pt`
- `/tmp/x100_v4_exact30k_all_x8_sup_l3_h1024.pt`
- `/tmp/x100_v4_exact30k_allx4_eastx4_mix_l3_h1024.pt`
- `/tmp/x100_v4_exactfo_x64_sup_l3_h1024.pt`
- `/tmp/x100_v4_exactfo_x64_mix_l3_h1024.pt`

Reports:

- `/tmp/x100_v4_exact30k_all_x4_sup_l3_h1024.teacher_report.json`
- `/tmp/x100_v4_exact30k_all_x8_sup_l3_h1024.teacher_report.json`
- `/tmp/x100_v4_exact30k_allx4_eastx4_mix_l3_h1024.teacher_report.json`
- `/tmp/x100_v4_exactfo_x64_sup_l3_h1024.teacher_report.json`
- `/tmp/x100_v4_exactfo_x64_mix_l3_h1024.teacher_report.json`

## Primary Results

### 1. Broader reset-slice mining did find real additional coverage

Broader geometry miner summary:

- rows mined: `25`
- action split:
  - `north`: `13`
  - `east`: `7`
  - `south`: `5`
- reset errors: `25`

This is much broader than the earlier `3`-row exact-slice result, but the action distribution is not dominated by `east`.

### 2. Broader-slice teacher probes all regressed to `0.975`

Results:

- `x100_v4_exact30k_all_x4_sup_l3_h1024.pt`
  - held-out trace match: `0.975`
  - mismatches:
    - `east -> west`
    - `east -> north`
- `x100_v4_exact30k_all_x8_sup_l3_h1024.pt`
  - held-out trace match: `0.975`
  - mismatches:
    - `west -> east`
    - `east -> north`
- `x100_v4_exact30k_allx4_eastx4_mix_l3_h1024.pt`
  - held-out trace match: `0.975`
  - mismatches:
    - `west -> east`
    - `east -> north`

So the broader slice did not beat `0.9875`. It added enough pressure to change the failure mode, but not enough to preserve the rest of the benchmark.

### 3. The exact held-out species pair is extremely rare off-heldout

Exact-species miner summary:

- signature:
  - `north=monster_f`
  - `south=floor`
  - `east=monster_o`
  - `west=floor`
- rows mined in `30k` fresh seeds: `1`
- teacher action on that row: `east`

This is strong evidence that the final held-out miss belongs to a very thin slice under the current reset distribution.

### 4. Even exact-species oversampling still regressed to `0.975`

Results:

- `x100_v4_exactfo_x64_sup_l3_h1024.pt`
  - held-out trace match: `0.975`
  - mismatches:
    - `east -> west`
    - `east -> south`
- `x100_v4_exactfo_x64_mix_l3_h1024.pt`
  - held-out trace match: `0.975`
  - mismatches:
    - `east -> west`
    - `east -> south`

So even the cleanest exact-species off-heldout row did not produce a `1.0` teacher.

## Interpretation

What held up:

- the reset-slice miner is useful infrastructure and works on real rare cases
- the last baseline miss really does live in a thin slice of the data distribution
- the loose `monster_*` geometry is not enough context to determine the teacher action
- the exact `north=f / east=o` held-out pattern is extremely rare off-heldout

What did not hold up:

- broader off-heldout mining alone is not enough to beat `0.9875`
- exact-species oversampling alone is not enough to beat `0.9875`
- the remaining teacher gap is not just “one missing example away”

The most useful new diagnosis is:

- the last `0.9875 -> 1.0` gap is now too thin and branch-specific for simple hard-case oversampling to be a reliable mainline improvement strategy

In other words:

- the miner found real additional data,
- but the benchmark remained constrained by collateral regressions elsewhere

## Recommended Next Move

Do not promote this branch as a new teacher baseline.

Keep the reset-slice miner as reusable data-refinement infrastructure, then shift focus away from single-slice oversampling.

The best next options are:

1. mine broader nearby-family failures rather than one exact slice
2. improve the online improver, which is still the main bottleneck by the trusted benchmark
3. treat `0.9875` as the current teacher ceiling under this small-data MLP family unless a broader data or objective change wins offline
