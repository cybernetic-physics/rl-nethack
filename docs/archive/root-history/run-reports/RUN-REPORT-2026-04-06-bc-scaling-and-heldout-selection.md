## Purpose

Test whether scaling the offline BC teacher path with:

- larger hidden sizes,
- deeper MLPs,
- longer training,
- more teacher data,
- and GPU training

can beat the current strongest cheap teacher on the deterministic held-out trace benchmark.

Baseline before this run:

- teacher artifact: `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- held-out trace match: `0.9875`

## Code Paths Touched

- [cli.py](/home/luc/rl-nethack/cli.py)
- [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## What Changed

Added two scaling-oriented BC features:

- explicit device selection for BC training
- optional checkpoint selection by deterministic held-out trace match

Why this matters:

- the repo now has enough VRAM to run much larger offline teacher jobs cheaply
- but without benchmark-aware checkpoint selection, larger runs can regress late even when they briefly hit the old best score

## Validation

Command:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `73 passed`

## Benchmark Regime

- trusted metric:
  - deterministic held-out trace match on `/tmp/x100_v4_heldout_traces.jsonl`
- observation version:
  - `v4`
- training regimes:
  - `680`-row `trainplus + teachergen` set
  - `1400`-row merged `trainplus + teachergen + dagger` set

## Exact Commands Run

Scaled teacher on merged `1400` rows, no held-out selection:

```bash
uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_trainplus_daggerboth_1400.jsonl \
  --output /tmp/x100_v4_scale1400_distill_l4_h2048_cuda.pt \
  --epochs 120 --lr 0.0005 --hidden-size 2048 --num-layers 4 \
  --observation-version v4 \
  --distill-teacher-bc-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --distill-loss-coef 1.0 --distill-temperature 2.0 \
  --supervised-loss-coef 0.0 \
  --device cuda \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

Scaled teacher on merged `1400` rows, with held-out selection:

```bash
uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_trainplus_daggerboth_1400.jsonl \
  --output /tmp/x100_v4_scale1400_distill_l4_h2048_cuda_select.pt \
  --epochs 120 --lr 0.0005 --hidden-size 2048 --num-layers 4 \
  --observation-version v4 \
  --distill-teacher-bc-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --distill-loss-coef 1.0 --distill-temperature 2.0 \
  --supervised-loss-coef 0.0 \
  --device cuda \
  --select-by-heldout \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

Scaled teacher on cleaner `680` rows, with held-out selection:

```bash
uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_trainplus_teachergen_ensemble.jsonl \
  --output /tmp/x100_v4_trainplus_scale_l4_h2048_cuda_select.pt \
  --epochs 120 --lr 0.0005 --hidden-size 2048 --num-layers 4 \
  --observation-version v4 \
  --distill-teacher-bc-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --distill-loss-coef 1.0 --distill-temperature 2.0 \
  --supervised-loss-coef 0.0 \
  --device cuda \
  --select-by-heldout \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

## Artifacts

- no-selection scaled teacher:
  - `/tmp/x100_v4_scale1400_distill_l4_h2048_cuda.pt`
  - `/tmp/x100_v4_scale1400_distill_l4_h2048_cuda.pt.teacher_report.json`
- selection-scaled teacher on `1400` rows:
  - `/tmp/x100_v4_scale1400_distill_l4_h2048_cuda_select.pt`
  - `/tmp/x100_v4_scale1400_distill_l4_h2048_cuda_select.pt.teacher_report.json`
- selection-scaled teacher on `680` rows:
  - `/tmp/x100_v4_trainplus_scale_l4_h2048_cuda_select.pt`
  - `/tmp/x100_v4_trainplus_scale_l4_h2048_cuda_select.pt.teacher_report.json`

## Primary Metrics

Baseline:

- `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- held-out trace match: `0.9875`

Merged `1400` rows, `2048x4`, CUDA, no selection:

- final held-out trace match: `0.975`

Merged `1400` rows, `2048x4`, CUDA, held-out selection:

- best held-out trace match: `0.9875`
- selected epoch: `47`
- final late-epoch train accuracy: `0.8857`

Cleaner `680` rows, `2048x4`, CUDA, held-out selection:

- best held-out trace match: `0.9875`
- selected epoch: `94`
- late epochs drift back to `0.975`

## Interpretation

What held up:

- scaling the offline teacher path is technically viable on GPU
- larger BC teachers can now be trained and selected safely with the trusted metric
- held-out checkpoint selection is not optional once runs get large enough to overfit or drift late

What did not hold up:

- larger width, depth, dataset size, and longer training did not beat `0.9875`
- more rows from the existing DAgger-style mixtures did not fix the remaining held-out miss
- final epoch quality is a bad proxy for teacher quality in the scaled regime

The strongest read is:

- scaling alone is not the breakthrough
- scale only becomes useful when tied tightly to the deterministic benchmark
- the right next scaling target is not blind APPO rollout length
- the right next scaling target is better teacher-data coverage and harder-case collection under the same trusted benchmark

## Recommended Next Move

Keep this infrastructure and use it as the default for future serious BC teacher runs:

- train on GPU
- always emit epoch-wise held-out scores
- select checkpoints by deterministic held-out trace match, not final epoch

Do not promote large APPO rollout scaling yet.

Current evidence still says:

- the offline teacher path benefits from scale
- the online improver path does not yet justify more scale because the short gate is still failing
