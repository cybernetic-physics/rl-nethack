## Purpose

Test whether prompt-text-conditioned BC teachers can beat the current strongest numeric cheap teacher on the deterministic held-out trace benchmark.

Current baseline before this probe:

- teacher artifact: `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- held-out trace match: `0.9875`
- known remaining miss: one `east -> south` disagreement on held-out row `seed=202 step=0`

## Code Paths Touched

- [cli.py](/home/luc/rl-nethack/cli.py)
- [rl/bc_model.py](/home/luc/rl-nethack/rl/bc_model.py)
- [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py)
- [rl/trace_eval.py](/home/luc/rl-nethack/rl/trace_eval.py)
- [rl/evaluate_bc.py](/home/luc/rl-nethack/rl/evaluate_bc.py)
- [rl/debug_tools.py](/home/luc/rl-nethack/rl/debug_tools.py)
- [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## What Changed

Added prompt-text conditioning to the BC stack:

- optional `hash` text encoder
- optional frozen `transformer` text encoder
- prompt flow through BC train, save/load, trace eval, live eval, and trace generation
- regression coverage proving prompt text survives save/load and is used by trace evaluation

Important implementation detail:

- hash text encoding uses a stable `blake2b` token hash, not Python's process-randomized `hash()`

## Validation

Command:

```bash
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `71 passed`

## Benchmark Regime

- training data:
  - `/tmp/x100_v4_trainplus_teachergen_ensemble.jsonl`
  - `680` rows
- held-out benchmark:
  - `/tmp/x100_v4_heldout_traces.jsonl`
  - deterministic trace match

## Exact Commands Run

Hash, pure distill:

```bash
uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_trainplus_teachergen_ensemble.jsonl \
  --output /tmp/x100_v4_textbc_hash_trainplus_l3_h1024.pt \
  --epochs 80 --lr 0.0005 --hidden-size 1024 --num-layers 3 \
  --observation-version v4 \
  --distill-teacher-bc-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --distill-loss-coef 1.0 --distill-temperature 2.0 \
  --supervised-loss-coef 0.0 \
  --text-encoder-backend hash \
  --text-embedding-dim 128 \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

Hash, mixed supervision with east boost:

```bash
uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_trainplus_teachergen_ensemble.jsonl \
  --output /tmp/x100_v4_textbc_hash_trainplus_sup01_east2_l3_h1024.pt \
  --epochs 80 --lr 0.0005 --hidden-size 1024 --num-layers 3 \
  --observation-version v4 \
  --distill-teacher-bc-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --distill-loss-coef 1.0 --distill-temperature 2.0 \
  --supervised-loss-coef 0.1 \
  --action-weight-boosts east=2.0 \
  --text-encoder-backend hash \
  --text-embedding-dim 128 \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

Frozen DistilBERT, pure distill:

```bash
uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_trainplus_teachergen_ensemble.jsonl \
  --output /tmp/x100_v4_textbc_distilbert_trainplus_l3_h1024.pt \
  --epochs 80 --lr 0.0005 --hidden-size 1024 --num-layers 3 \
  --observation-version v4 \
  --distill-teacher-bc-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --distill-loss-coef 1.0 --distill-temperature 2.0 \
  --supervised-loss-coef 0.0 \
  --text-encoder-backend transformer \
  --text-model-name distilbert-base-uncased \
  --text-max-length 128 \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

Frozen DistilBERT, pure supervised:

```bash
uv run python cli.py rl-train-bc \
  --input /tmp/x100_v4_trainplus_teachergen_ensemble.jsonl \
  --output /tmp/x100_v4_textbc_distilbert_supervised_trainplus_l3_h1024.pt \
  --epochs 80 --lr 0.0005 --hidden-size 1024 --num-layers 3 \
  --observation-version v4 \
  --supervised-loss-coef 1.0 \
  --text-encoder-backend transformer \
  --text-model-name distilbert-base-uncased \
  --text-max-length 128 \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl
```

## Artifacts

- hash distill:
  - `/tmp/x100_v4_textbc_hash_trainplus_l3_h1024.pt`
  - `/tmp/x100_v4_textbc_hash_trainplus_l3_h1024.pt.teacher_report.json`
- hash mixed:
  - `/tmp/x100_v4_textbc_hash_trainplus_sup01_east2_l3_h1024.pt`
  - `/tmp/x100_v4_textbc_hash_trainplus_sup01_east2_l3_h1024.pt.teacher_report.json`
- distilbert distill:
  - `/tmp/x100_v4_textbc_distilbert_trainplus_l3_h1024.pt`
  - `/tmp/x100_v4_textbc_distilbert_trainplus_l3_h1024.pt.teacher_report.json`
- distilbert supervised:
  - `/tmp/x100_v4_textbc_distilbert_supervised_trainplus_l3_h1024.pt`
  - `/tmp/x100_v4_textbc_distilbert_supervised_trainplus_l3_h1024.pt.teacher_report.json`

## Primary Metrics

Baseline numeric teacher:

- `0.9875`

Hash, pure distill:

- held-out trace match: `0.9875`
- common mismatches: one `east -> south`

Hash, mixed supervision + east boost:

- held-out trace match: `0.9875`
- common mismatches: one `east -> south`

Frozen DistilBERT, pure distill:

- held-out trace match: `0.9875`
- common mismatches: one `east -> south`

Frozen DistilBERT, pure supervised:

- held-out trace match: `0.95`
- common mismatches:
  - `east -> south` x2
  - `west -> east` x1
  - `south -> north` x1

## Interpretation

What held up:

- prompt-conditioned BC is now a real, reusable code path
- both hash and transformer prompt encoders can match the current numeric baseline under teacher distillation
- stronger prompt encoders did not destabilize the benchmark path

What did not hold up:

- prompt text did not fix the remaining held-out error
- stronger prompt encoders did not beat the numeric `0.9875` teacher
- pure supervised prompt-aware learning was clearly worse than distillation here

The strongest read is:

- missing prompt text is not the main reason the current teacher misses the final held-out row
- the remaining miss is more likely a data or objective blind spot than a simple absence of text conditioning

The hard held-out row is:

- `seed=202`
- `step=0`
- teacher action `east`
- prompt summary:
  - `north=monster_f`
  - `south=floor`
  - `east=monster_o`
  - `west=floor`

Even with prompt-aware transformer conditioning, the distilled teacher still predicts `south`.

## Recommended Next Move

Do not promote this teacher family further as-is.

Best next move:

- keep the prompt-conditioned BC infrastructure
- stop treating missing prompt text as the leading explanation for the `0.9875 -> 1.0` gap
- redirect effort toward:
  - better data around the remaining hard cases,
  - structured teacher targets,
  - or a different online improver rather than more prompt-conditioned BC sweeps
