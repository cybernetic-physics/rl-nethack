# Purpose

Test whether selective DAgger becomes useful when it is restricted to the exact held-out failure family instead of broad disagreement or broad hard-case flags.

# Comparable Baseline

- Trusted teacher artifact: `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- Trusted held-out trace split: `/tmp/x100_v4_heldout_traces.jsonl`
- Base training traces: `/tmp/x100_v4_train_traces.jsonl`
- Current strongest comparable constrained online branch: `train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a`
- Trusted offline baseline: `0.9875`

# Hypothesis

The previous selective-DAGGER probes failed because they were still too broad. If relabeling is restricted to the exact `behavior_action -> teacher_action` confusion family that matters for the trusted benchmark, it may add useful support without collapsing the teacher.

The concrete target here was:

- `east -> south`
- `south -> east`

# Code Paths Touched

- `/home/luc/rl-nethack/rl/dagger.py`
- `/home/luc/rl-nethack/cli.py`
- `/home/luc/rl-nethack/tests/test_rl_scaffold.py`

# What Changed

- Added `dagger_confusion_pairs` to DAgger CLI and schedule plumbing.
- Added reusable parsing for comma-separated `behavior->teacher` filters.
- Extended row selection so DAgger can filter by:
  - row policy
  - plus exact confusion pairs
- Added confusion-pair counts to DAgger row summaries.

# Validation

Commands:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k "select_dagger_rows_targets_hard_cases_and_keeps_match_anchor or run_dagger_iteration_filters_dagger_rows_before_merge or run_dagger_iteration_passes_teacher_bc_and_distill or run_dagger_iteration_accepts_appo_checkpoint_without_experiment"
uv run pytest -q tests/test_rl_scaffold.py
```

Results:

- targeted: `4 passed`
- full scaffold: `83 passed`

# Audit Before Probe

The previously harvested disagreement file `/tmp/dagger_splitgate_final_disagreement.jsonl` showed that broad disagreement was dominated by irrelevant confusions:

- `east -> west`: `41`
- `north -> south`: `18`
- `east -> north`: `14`
- `south -> west`: `13`
- `east -> south`: `5`

So the exact held-out failure family was present, but rare.

# Exact Command Run

```bash
uv run python cli.py rl-run-dagger \
  --input /tmp/x100_v4_train_traces.jsonl \
  --dagger-output /tmp/dagger_splitgate_final_espair.jsonl \
  --bc-output /tmp/dagger_splitgate_final_espair.pt \
  --merged-output /tmp/dagger_splitgate_final_espair_merged.jsonl \
  --student-policy appo \
  --task explore \
  --num-episodes 128 \
  --max-steps 40 \
  --seed-start 3000 \
  --appo-checkpoint-path train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a/checkpoint_p0/checkpoint_000000018_4608.pth \
  --teacher-bc-model-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --observation-version v4 \
  --merge-ratio 1.0 \
  --merge-policy base_only \
  --dagger-row-policy disagreement \
  --dagger-keep-match-ratio 0.0 \
  --dagger-confusion-pairs 'east->south,south->east' \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl \
  --epochs 80 \
  --hidden-size 1024 \
  --distill-teacher-bc-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --distill-loss-coef 1.0 \
  > /tmp/dagger_splitgate_final_espair_report.json
```

Held-out disagreement audit of the resulting teacher:

```bash
uv run python - <<'PY'
from rl.trace_eval import trace_disagreement_report
import json
res = trace_disagreement_report('/tmp/x100_v4_heldout_traces.jsonl', bc_model_path='/tmp/dagger_splitgate_final_espair.pt', top_k=20)
print(json.dumps(res, indent=2))
PY
```

# Artifacts

- `/tmp/dagger_splitgate_final_espair.jsonl`
- `/tmp/dagger_splitgate_final_espair_merged.jsonl`
- `/tmp/dagger_splitgate_final_espair.pt`
- `/tmp/dagger_splitgate_final_espair_report.json`

# Benchmark Regime

- Level 1 offline loop
- student-state harvest from the split-base final checkpoint
- deterministic held-out trace match for promotion

# Primary Metrics

- harvested DAgger rows: `5120`
- disagreement rows: `502`
- exact confusion-pair rows: `144`
- selected DAgger rows after exact-pair filtering: `144`
- held-out trace match: `0.975`
- invalid action rate: `0.0`

# Supporting Metrics

- harvest teacher-match rate: `0.902`
- selected rows were all true disagreement rows
- selected rows were all exact confusion-pair rows
- selected rows were all weak-action rows in practice
- resulting held-out action counts:
  - `north`: `23`
  - `east`: `26`
  - `south`: `20`
  - `west`: `11`

Held-out disagreement report for the resulting BC teacher:

- still `0.975`
- remaining mismatches:
  - `east -> south`: `2`

# Interpretation

- Exact confusion-pair filtering is materially better than the earlier broad DAgger filters.
- It avoided the catastrophic regressions of:
  - `hard_only` at `0.8625`
  - `disagreement` at `0.85`
- But it still did not beat the trusted teacher baseline `0.9875`.
- The remaining held-out failure is still the same family: `east -> south`.

# What Held Up

- The new `dagger_confusion_pairs` selector is reusable.
- The exact held-out failure family does exist in student rollouts if enough rows are harvested.
- Restricting to the exact pair made DAgger safe enough to stay near the current constrained online branch’s final `0.975` behavior.

# What Failed

- Exact-pair relabeling still did not lift the offline teacher above `0.9875`.
- Even with `144` exact `east <-> south` rows, the retrained BC model still missed two `east` rows as `south`.
- So the missing capability is not fixed by adding more of this one harvested confusion family alone.

# Preservation Or Improvement

This was not improvement. It was a safer negative result than broad DAgger, but it still fell short of the trusted baseline.

# Recommended Next Move

- Keep `dagger_confusion_pairs` as reusable selective-relabel infrastructure.
- Do not promote this branch to online RL.
- Shift away from broad online-rollout DAgger and toward:
  - offline teacher-data refinement around the exact held-out east-family states
  - or richer replay data tied to explicit benchmark regressions
- Treat this as evidence that exact student-harvested confusion relabeling is safer than naive DAgger, but still insufficient on its own.
