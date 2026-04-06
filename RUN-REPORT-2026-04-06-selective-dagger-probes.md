# Purpose

Test whether selective teacher relabeling on student-induced states can improve the offline teacher path without falling back to naive full-state DAgger.

# Comparable Baseline

- Trusted teacher artifact: `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- Trusted held-out split: `/tmp/x100_v4_heldout_traces.jsonl`
- Base training traces: `/tmp/x100_v4_train_traces.jsonl`
- Current strongest comparable online branch: `train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a`
- Trusted teacher / clone score: `0.9875`

# Hypothesis

The current replay source is too small and too teacher-only. If we relabel student states harvested from the drifted split-base checkpoint, then keep only disagreement-heavy or otherwise hard rows, the resulting BC retrain may improve held-out trace match relative to naive DAgger and reveal a better replay-data path.

# Code Paths Touched

- `/home/luc/rl-nethack/rl/dagger.py`
- `/home/luc/rl-nethack/cli.py`
- `/home/luc/rl-nethack/tests/test_rl_scaffold.py`

# What Changed

- Added `dagger_row_policy` with choices:
  - `all`
  - `disagreement`
  - `loop_risk`
  - `failure_slice`
  - `weak_action`
  - `hard_only`
- Added `dagger_keep_match_ratio` so a filtered DAgger set can retain a small on-support anchor slice.
- Added row-selection summaries to `run_dagger_iteration` and `run_dagger_schedule`.
- Exposed the new knobs through `rl-run-dagger` and `rl-dagger-iterate`.

# Validation

Commands:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k "select_dagger_rows_targets_hard_cases_and_keeps_match_anchor or run_dagger_iteration_filters_dagger_rows_before_merge or run_dagger_iteration_passes_teacher_bc_and_distill or run_dagger_iteration_accepts_appo_checkpoint_without_experiment or build_merged_trace_rows_policies"
uv run pytest -q tests/test_rl_scaffold.py
```

Results:

- targeted: `5 passed`
- full scaffold: `83 passed`

# Offline Probe Commands

Hard-only selective DAgger:

```bash
uv run python cli.py rl-run-dagger \
  --input /tmp/x100_v4_train_traces.jsonl \
  --dagger-output /tmp/dagger_splitgate_final_hardonly.jsonl \
  --bc-output /tmp/dagger_splitgate_final_hardonly.pt \
  --merged-output /tmp/dagger_splitgate_final_hardonly_merged.jsonl \
  --student-policy appo \
  --task explore \
  --num-episodes 32 \
  --max-steps 40 \
  --seed-start 2000 \
  --appo-checkpoint-path train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a/checkpoint_p0/checkpoint_000000018_4608.pth \
  --teacher-bc-model-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --observation-version v4 \
  --merge-ratio 1.0 \
  --merge-policy base_only \
  --dagger-row-policy hard_only \
  --dagger-keep-match-ratio 0.1 \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl \
  --epochs 80 \
  --hidden-size 1024 \
  --distill-teacher-bc-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --distill-loss-coef 1.0 \
  > /tmp/dagger_splitgate_final_hardonly_report.json
```

Disagreement-only selective DAgger:

```bash
uv run python cli.py rl-run-dagger \
  --input /tmp/x100_v4_train_traces.jsonl \
  --dagger-output /tmp/dagger_splitgate_final_disagreement.jsonl \
  --bc-output /tmp/dagger_splitgate_final_disagreement.pt \
  --merged-output /tmp/dagger_splitgate_final_disagreement_merged.jsonl \
  --student-policy appo \
  --task explore \
  --num-episodes 32 \
  --max-steps 40 \
  --seed-start 2000 \
  --appo-checkpoint-path train_dir/rl/appo_v4_distill_ensemble_l3pure_splitgate_f055_4k_a/checkpoint_p0/checkpoint_000000018_4608.pth \
  --teacher-bc-model-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --observation-version v4 \
  --merge-ratio 1.0 \
  --merge-policy base_only \
  --dagger-row-policy disagreement \
  --dagger-keep-match-ratio 0.1 \
  --heldout-input /tmp/x100_v4_heldout_traces.jsonl \
  --epochs 80 \
  --hidden-size 1024 \
  --distill-teacher-bc-path /tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt \
  --distill-loss-coef 1.0 \
  > /tmp/dagger_splitgate_final_disagreement_report.json
```

# Artifact Paths

- `/tmp/dagger_splitgate_final_hardonly_report.json`
- `/tmp/dagger_splitgate_final_hardonly.jsonl`
- `/tmp/dagger_splitgate_final_hardonly_merged.jsonl`
- `/tmp/dagger_splitgate_final_hardonly.pt`
- `/tmp/dagger_splitgate_final_disagreement_report.json`
- `/tmp/dagger_splitgate_final_disagreement.jsonl`
- `/tmp/dagger_splitgate_final_disagreement_merged.jsonl`
- `/tmp/dagger_splitgate_final_disagreement.pt`

# Benchmark Regime

- Level 1 offline loop
- student-state harvest from the drifted split-base final checkpoint
- evaluation by deterministic held-out trace match

# Primary Metrics

Hard-only probe:

- base rows: `200`
- generated DAgger rows: `1280`
- selected DAgger rows: `1280`
- held-out trace match: `0.8625`

Disagreement-only probe:

- base rows: `200`
- generated DAgger rows: `1280`
- selected DAgger rows: `221`
- held-out trace match: `0.85`

# Supporting Metrics

Hard-only selected-row summary:

- disagreement rows: `115`
- loop-risk rows: `1280`
- weak-action rows: `581`
- matched rows: `1165`
- teacher match rate during trace harvest: `0.9102`

Disagreement-only selected-row summary:

- disagreement rows: `103`
- loop-risk rows: `221`
- weak-action rows: `122`
- matched rows: `118`
- teacher match rate during trace harvest: `0.9195`

# Interpretation

- The new selective-DAgger plumbing is reusable and works as intended.
- The actual probes are strongly negative.
- `hard_only` was not selective in practice because every harvested row carried `is_loop_risk = True`, so the filter collapsed back to near-naive DAgger.
- Tightening to `disagreement` fixed the selection problem, but the offline teacher still regressed badly to `0.85`.
- Neither branch came close to the trusted `0.9875` teacher baseline.

# What Held Up

- DAgger row filtering and reporting now exist end-to-end.
- The code can distinguish raw harvested rows from selected relabel rows.
- The cheap offline gate worked and prevented promotion to online RL.

# What Failed

- Current DAgger row flags are not yet sharp enough for safe selective relabeling.
- `is_loop_risk` is too broad to be trusted as a main selector in this form.
- Student-state relabeling from the drifted split-base checkpoint did not improve the teacher path.

# Preservation Or Improvement

This was not improvement. Both probes were large regressions on the trusted offline benchmark.

# Recommended Next Move

- Do not promote this branch to online RL.
- Keep the selective-DAgger infrastructure.
- Refine the selection labels before using DAgger again:
  - stricter loop criteria
  - held-out-failure-family triggers
  - disagreement slices tied to the actual trusted benchmark regressions
- Prefer replay / relabel data harvested from explicitly diagnosed failure states over broad student rollouts.
