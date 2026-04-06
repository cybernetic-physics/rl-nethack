# World Model Ensemble Distill Report

## Purpose

Push the strongest world-model teacher-transfer line further and test whether the new gain survives:

- first in offline cheap-student distillation,
- then in a short online APPO gate.

The active question was:

- can we beat the old cheap-teacher baseline of `0.975` on the deterministic held-out trace benchmark,
- and if so, does that improve the online improver at all?

## Code Paths Touched

- [rl/bc_model.py](/home/luc/rl-nethack/rl/bc_model.py)
- [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py)
- [rl/relabel_traces.py](/home/luc/rl-nethack/rl/relabel_traces.py)
- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [cli.py](/home/luc/rl-nethack/cli.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## Main Hypotheses

1. The two best cheap world-model-distilled teachers are complementary enough that an ensemble teacher can beat `0.975`.
2. The current student-transfer bottleneck may be student architecture, not just objective coefficients.
3. If a stronger cheap student exists, it should be the next short online APPO initialization target.

## Infrastructure Added

### 1. Multi-teacher trace relabeling

`rl-relabel-traces-bc` can now consume a comma-separated BC teacher list and average teacher logits before relabeling.

This reuses the same ensemble semantics as BC distillation instead of inventing a second teacher-combination path.

### 2. Multi-teacher BC distillation

BC training now supports:

- multiple teacher BC checkpoints,
- reduced or zero hard-label CE via `--supervised-loss-coef`,
- and deeper BC students via `--num-layers`.

### 3. Deeper-BC warm-start fix for APPO

I found and fixed a real bug in [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py):

- APPO BC warm-start assumed every BC teacher had exactly two hidden layers,
- so deeper BC teachers were mis-copied into the APPO encoder/policy head,
- which made early online results untrustworthy.

The fix now:

- copies the first two BC hidden linear layers into the APPO encoder,
- copies the final BC linear layer into the APPO action head,
- and keeps older two-layer BC checkpoints backward compatible.

## Validation

Targeted regression gates passed:

- `uv run pytest -q tests/test_rl_scaffold.py -k 'deeper_student or warmstart or distillation or parse_teacher_bc_paths or relabel'`
- result: `7 passed`

Also passed:

- `python3 -m compileall rl/bc_model.py rl/train_bc.py rl/relabel_traces.py rl/teacher_reg.py rl/trainer.py tests/test_rl_scaffold.py cli.py`

## Offline Experiments

### A. Two-model ensemble teacher

Teachers:

- `/tmp/x100_v4_distill_textdistil_c020_t2_h512.pt`
- `/tmp/x100_v4_distill_textdistil_c025_t2_h512.pt`

Direct ensembled held-out trace result:

- `0.9875`

This is a real benchmark improvement over the old single-teacher `0.975`.

### B. Simple two-layer student transfer

I tested several two-layer cheap students against the ensemble teacher:

- `hidden=1024`, `distill=1.0`, `T=2.0`, `supervised=0.1` -> `0.975`
- `hidden=1024`, `distill=1.0`, `T=2.0`, `supervised=0.0` -> `0.975`

Conclusion:

- objective tweaks alone were not enough,
- two-layer students still saturated at the old `0.975` ceiling.

### C. Expanded teacher-state data

I generated extra teacher-like states with the existing `0.975` cheap student:

- `16 episodes x 30 steps`
- `480` rows
- no invalid actions

Then I relabeled those traces with the ensemble teacher:

- changed rows: `8 / 480`
- changed rate: `0.0167`

Merged-dataset results:

- plain BC on merged data -> `0.9625`
- merged data + ensemble distillation -> `0.975`

Conclusion:

- extra teacher-state coverage alone did not beat `0.975`.

### D. Deeper cheap student

The first real win came from a deeper student:

- input: original honest `200`-row train split
- teachers: the two cheap `0.975` world-model-distilled BC teachers
- student:
  - `hidden_size=1024`
  - `num_layers=3`
  - `distill_loss_coef=1.0`
  - `distill_temperature=2.0`
  - `supervised_loss_coef=0.0`
  - `epochs=80`
  - `lr=5e-4`

Artifact:

- `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- teacher report:
  - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt.teacher_report.json`

Held-out result:

- `0.9875`

This matches the direct ensemble teacher while remaining a single cheap student checkpoint.

This is now the strongest honest cheap teacher artifact in the repo.

## Short Online RL Gate

Promoted artifact:

- BC init:
  - `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- teacher regularizer:
  - `/tmp/x100_v4_distill_textdistil_c020_t2_h512.pt`
  - `/tmp/x100_v4_distill_textdistil_c025_t2_h512.pt`

Experiment:

- `train_dir/rl/appo_v4_distill_ensemble_l3pure_short_b`

Config was the same conservative short APPO gate used for previous teacher-centered runs:

- `rollout-length=16`
- `reward_scale=0.005`
- `gamma=0.99`
- `gae_lambda=0.9`
- `value_loss_coeff=0.1`
- scheduled teacher loss
- scheduled teacher replay
- deterministic held-out trace monitoring

### Important bug note

An earlier run on this branch (`short_a`) was invalid for interpretation because the APPO BC warm-start path was still using the old broken two-layer assumption.

The `short_b` rerun is the trusted one.

### Trusted result for `short_b`

Checkpoint ranking:

- best learned checkpoint:
  - `checkpoint_000000003_768.pth`
  - `0.2875`
- final retained checkpoints:
  - `checkpoint_000000017_4352.pth`
  - `checkpoint_000000018_4608.pth`
  - both `0.225`

Interpretation:

- the new cheap teacher is genuinely stronger offline,
- but the current online improver still collapses almost immediately under this setup,
- so this branch does **not** earn medium promotion.

## What Held Up

- The world-model teacher-transfer line is still the strongest teacher-building path in the repo.
- Ensemble teacher information is real and compressible into a single cheap student.
- Student architecture capacity mattered more than the earlier objective tweaks suggested.
- Fixing integration bugs before trusting online runs remains essential.

## What Did Not Hold Up

- More teacher-state data did not beat the old baseline by itself.
- Two-layer students could not preserve the ensemble teacher gain.
- A better offline teacher still did not rescue the current APPO-family improver.

## Current Best Read

The repo advanced in a meaningful way:

- old best cheap teacher: `0.975`
- new best cheap teacher: `0.9875`

But the broader scientific conclusion did not change:

- the teacher pipeline improved again,
- the online improver is still the bottleneck,
- and this is now even clearer because the stronger teacher made the improver failure easier to see.

## Recommended Next Move

Do not run a medium or large APPO branch from this teacher.

Instead:

1. preserve this new `0.9875` teacher as the offline baseline,
2. keep the deeper-student and multi-teacher infrastructure,
3. treat the online failure as further evidence that the next mainline work should target the improver, not more teacher scaling on the same APPO branch.
