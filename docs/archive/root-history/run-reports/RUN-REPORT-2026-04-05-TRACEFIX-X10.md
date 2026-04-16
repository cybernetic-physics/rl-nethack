# Tracefix X10 Validation Run

Date: 2026-04-05

## Purpose

Validate the recent RL harness fixes with a larger BC-warm-start APPO run.

The main code changes under test were:

- deterministic trace-based evaluation
- robust APPO checkpoint loading
- corrected trace feature generation using pre-step timestep features
- stricter trace verification
- BC training guards against mixed observation versions and feature dimensions

## Key Conclusions

1. The fast debug loop now works.
2. The repaired x10-scale APPO run is materially better than the earlier APPO runs.
3. The learned RL policy still does not beat the BC warm start on the deterministic trace regression set.
4. Live seed-based eval remains useful only as a diagnostic, not as a trusted benchmark.

## Commands Run

### 1. Validate harness after fixes

```bash
uv run pytest -q tests/test_rl_scaffold.py
uv run python -m compileall rl cli.py src/evaluator.py
```

### 2. Generate corrected traces

```bash
uv run python cli.py rl-generate-traces \
  --output data/tracefix_v2_explore_traces.jsonl \
  --num-episodes 20 \
  --max-steps 20 \
  --task explore \
  --policy task_greedy \
  --observation-version v2
```

This run was intentionally stopped once the file had enough data and the rows had been manually inspected.

### 3. Verify trace artifact

```bash
uv run python cli.py rl-verify-traces \
  --input data/tracefix_v2_explore_traces.jsonl
```

### 4. Train BC on corrected traces

```bash
uv run python cli.py rl-train-bc \
  --input data/tracefix_v2_explore_traces.jsonl \
  --output output/tracefix_v2_explore_bc.pt \
  --epochs 20 \
  --lr 1e-3 \
  --hidden-size 512 \
  --observation-version v2
```

### 5. Run x10-scale APPO from BC warm start

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-appo \
  --experiment appo_tracefix_v2_bc_x10 \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --recurrence 16 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 500000 \
  --enabled-skills explore \
  --observation-version v2 \
  --bc-init-path output/tracefix_v2_explore_bc.pt
```

The run was stopped after `216,064` frames rather than waiting for the nominal `500,000`, because by that point the training regime had clearly changed and the deterministic evaluation target was available.

### 6. Deterministic trace evaluation

```bash
uv run python - <<'PY'
from rl.trace_eval import evaluate_trace_policy
import json
for policy, kwargs in [
    ("bc", dict(bc_model_path="output/tracefix_v2_explore_bc.pt")),
    ("appo", dict(appo_experiment="appo_tracefix_v2_bc_x10", appo_train_dir="train_dir/rl")),
]:
    result = evaluate_trace_policy(
        "data/tracefix_v2_explore_traces.jsonl",
        policy,
        summary_only=True,
        **kwargs,
    )
    print(json.dumps(result, indent=2))
PY
```

### 7. Live diagnostic evaluation

```bash
uv run python cli.py rl-evaluate-bc \
  --model output/tracefix_v2_explore_bc.pt \
  --seeds 42,43,44 \
  --max-steps 50

uv run python cli.py rl-evaluate-appo \
  --experiment appo_tracefix_v2_bc_x10 \
  --seeds 42,43,44 \
  --max-steps 50
```

### 8. Cross-check on the older held-out trace set

```bash
uv run python - <<'PY'
from rl.trace_eval import evaluate_trace_policy
import json
result = evaluate_trace_policy(
    "data/validate_v2_explore_traces.jsonl",
    "appo",
    appo_experiment="appo_tracefix_v2_bc_x10",
    appo_train_dir="train_dir/rl",
    summary_only=True,
)
print(json.dumps(result, indent=2))
PY
```

## Artifacts

### New / updated

- trace file: [data/tracefix_v2_explore_traces.jsonl](/home/luc/rl-nethack/data/tracefix_v2_explore_traces.jsonl)
- BC checkpoint: [output/tracefix_v2_explore_bc.pt](/home/luc/rl-nethack/output/tracefix_v2_explore_bc.pt)
- APPO experiment dir: [train_dir/rl/appo_tracefix_v2_bc_x10](/home/luc/rl-nethack/train_dir/rl/appo_tracefix_v2_bc_x10)

### APPO checkpoints of interest

- latest checkpoint at stop time: [train_dir/rl/appo_tracefix_v2_bc_x10/checkpoint_p0/checkpoint_000000211_216064.pth](/home/luc/rl-nethack/train_dir/rl/appo_tracefix_v2_bc_x10/checkpoint_p0/checkpoint_000000211_216064.pth)
- best training checkpoint at stop time: [train_dir/rl/appo_tracefix_v2_bc_x10/checkpoint_p0/best_000000210_215040_reward_124.315.pth](/home/luc/rl-nethack/train_dir/rl/appo_tracefix_v2_bc_x10/checkpoint_p0/best_000000210_215040_reward_124.315.pth)

## Results

### Trace verification

Corrected trace file summary:

- episodes: `13`
- rows: `258`
- max steps in episode: `20`
- avg steps in episode: `19.85`
- all multi-turn: `true`
- observation versions: `["v2"]`
- feature dims: `[160]`
- invalid action rows: `0`
- non-monotonic episode count: `0`

This confirms the repaired trace artifact is structurally sane.

### BC on corrected traces

Training metadata:

- epochs: `20`
- hidden size: `512`
- final loss: `0.9517`
- train accuracy: `0.6395`

Deterministic trace evaluation on the same corrected dataset:

- match rate: `0.6395`
- invalid action rate: `0.0`
- action counts:
  - `north: 104`
  - `east: 76`
  - `south: 37`
  - `west: 41`

This is the expected result after the trace-feature fix: trainer accuracy and deterministic replay now agree.

### APPO x10-scale run

Operational behavior:

- stable throughout
- no harness crashes
- no checkpoint load failures
- throughput around `~205 FPS`
- reward started strongly negative, then recovered through zero and into a clearly positive regime

Best observed training reward during the run:

- `124.315` at `215,040` frames

Deterministic trace evaluation on the corrected trace dataset:

- match rate: `0.5853`
- invalid action rate: `0.0`
- action counts:
  - `north: 91`
  - `east: 91`
  - `south: 28`
  - `west: 48`

This is a large improvement over the earlier APPO checkpoints, but it still trails BC on the same fixed trace benchmark:

- BC: `0.6395`
- APPO: `0.5853`

### Live diagnostic evaluation

BC on seeds `42,43,44`, `50` steps:

- avg unique tiles: `46.67`
- avg rooms discovered: `1.33`
- avg env reward: `5.3333`

APPO on the same diagnostic:

- avg task reward: `-1.1833`
- avg unique tiles: `51.33`
- avg rooms discovered: `1.0`
- repeated action rate: `0.6333`
- invalid action rate: `0.0`

Interpretation:

- APPO may be exploring more than BC in this particular live diagnostic
- but this signal is not trusted as a regression benchmark because NLE resets remain non-deterministic

### Cross-check on the older held-out trace set

APPO `appo_tracefix_v2_bc_x10` on [data/validate_v2_explore_traces.jsonl](/home/luc/rl-nethack/data/validate_v2_explore_traces.jsonl):

- match rate: `0.3866`
- invalid action rate: `0.0`
- action counts:
  - `north: 55`
  - `east: 57`
  - `south: 7`

This is still well below the corrected-trace result, which is expected because the old trace set was generated before the new trace-feature fix.

## What Worked

- The new fast debug loop is paying off.
- Deterministic trace evaluation separated real policy quality from live-env noise.
- The trace generation bug was real and the fix mattered.
- BC warm start still helps.
- Scaling APPO training budget clearly helps.

## What Did Not Work

- APPO still does not beat BC on the deterministic teacher trace benchmark.
- Live evaluations still show substantial repetition and are still noisy because of NLE reset instability.
- The latest checkpoint is not necessarily the same as the best training-reward checkpoint.

## Most Important New Understanding

Before these fixes, it was too easy to misread the system because:

- trace features were not guaranteed to match the pre-decision state
- evaluation mixed deterministic and non-deterministic paths
- APPO checkpoint loading was brittle

After this run, the picture is much clearer:

- BC is still the stronger policy on the fixed teacher-state regression task
- APPO is now close enough that continued improvement is plausible
- scale helps, but scale alone is not enough

## Recommended Next Steps

1. Add direct evaluation of an explicit checkpoint path so `best_*.pth` can be scored without relying on latest-checkpoint selection.
2. Add a trace-regression report command that prints summary only by default.
3. Train on a larger corrected trace corpus.
4. Add DAgger-style aggregation so APPO and BC are trained on student-visited states.
5. Keep using deterministic trace evaluation as the primary gate.
