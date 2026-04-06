# World Model Scaling Report

## Summary

The best honest result from this line of work is now a **frozen `distilbert-base-uncased` text-conditioned world model** used as an offline feature augmenter for teacher building, followed by **teacher-logit distillation back into a cheap base-`v4` student**.

That branch produced:

- world-model downstream BC probe on original held-out traces: **`0.9625`**
- real BC teacher trained on transformed traces: **`0.9625`**
- cheap base-`v4` student distilled from that teacher: **`0.9750`**

This is the strongest offline teacher result we have seen from the world-model path.

It still does **not** justify a medium or large online RL run, because short APPO from the `0.975` student only **tied** that score early and then drifted down to **`0.875`** by the end of the run.

## Critical Fixes

Before these scaling experiments, the text-conditioned world-model branch had a serious evaluation flaw:

- trace `prompt` text included `Action: ...`
- offline world-model transforms were reading that raw prompt
- this leaked the chosen action into the world-model features

That is now fixed.

Relevant source changes:

- [src/state_encoder.py](/home/luc/rl-nethack/src/state_encoder.py)
- [rl/world_model_dataset.py](/home/luc/rl-nethack/rl/world_model_dataset.py)
- [rl/world_model_features.py](/home/luc/rl-nethack/rl/world_model_features.py)
- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)
- [rl/traces.py](/home/luc/rl-nethack/rl/traces.py)

The branch now uses **state-only text** consistently for:

- offline world-model training
- offline world-model transforms
- online env augmentation
- BC trace generation from world-model-augmented teachers

## Eval and Throughput Improvements

I also improved the world-model instrumentation and performance:

- batched world-model auxiliary encoding in [rl/world_model.py](/home/luc/rl-nethack/rl/world_model.py)
- batched trace transforms in [rl/world_model_features.py](/home/luc/rl-nethack/rl/world_model_features.py)
- class-balanced action-loss option in [rl/train_world_model.py](/home/luc/rl-nethack/rl/train_world_model.py)
- reusable BC-teacher trace relabeling in [rl/relabel_traces.py](/home/luc/rl-nethack/rl/relabel_traces.py)
- CLI wiring in [cli.py](/home/luc/rl-nethack/cli.py)

The scaffold suite is currently green:

- `uv run pytest -q tests/test_rl_scaffold.py`
- result: **`57 passed`**

## Experiments

### 1. Honest corrected `bert-tiny`, original `200/80` split

State-only text, frozen `prajjwal1/bert-tiny`, `horizon=4`.

Result:

- downstream BC probe: **`0.9375`**
- real BC teacher from transformed traces: **`0.9250`**

Interpretation:

- the corrected text-conditioned world model is real
- but this small branch only ties the older honest numeric-world-model teacher line

### 2. Synthetic data scaling with `bert-tiny`

I generated a larger `800/240` base-`v4` corpus from a strong numeric world-model teacher and retrained the text-conditioned world model on it.

Best direct held-out metrics improved a lot:

- held-out action accuracy: about **`0.69`**
- top-3 accuracy: about **`0.98`**
- feature cosine: about **`0.76`**

But downstream policy usefulness got worse:

- `concat`: **`0.9125`**
- `concat_aux`: **`0.8708`**

Class-balanced action loss did not fix this:

- `concat`: **`0.9042`**
- `concat_aux`: **`0.9083`**

Interpretation:

- more synthetic data improved prediction on that synthetic distribution
- but the generated corpus was too `north`-heavy
- the world model still collapsed minority `west/south` behavior
- synthetic scaling in this form hurts the original honest benchmark

### 3. Horizon scaling on the honest original split

Frozen `bert-tiny`, `horizon=8`, original `200/80` data.

Result:

- downstream BC probe on original held-out traces: **`0.8000`**

Interpretation:

- longer horizon on the tiny honest dataset is harmful
- the current small-data regime does not support pushing horizon higher without losing policy usefulness

### 4. Larger frozen text backbone on the honest original split

Frozen `distilbert-base-uncased`, `horizon=4`, original `200/80` data.

Result:

- downstream BC probe, `concat`: **`0.9625`**
- downstream BC probe, `concat_aux`: **`0.9500`**

This is the first honest world-model branch that clearly beats the old `0.9375` ceiling.

Then I promoted it to a real teacher:

- transformed traces:
  - `/tmp/x100_v4_train_textdistil_h4_concat.jsonl`
  - `/tmp/x100_v4_heldout_textdistil_h4_concat.jsonl`
- real BC teacher:
  - `/tmp/x100_v4_textdistil_h4_concat_bc.pt`
- teacher report:
  - `/tmp/x100_v4_textdistil_h4_concat_bc.pt.teacher_report.json`

Real teacher result:

- held-out trace match: **`0.9625`**

So this result is not just a probe artifact.

### 5. Online APPO from the `distilbert` world-model teacher

I attempted a short teacher-constrained APPO bridge from the `0.9625` teacher.

What happened:

- the rollout path effectively stalled at startup
- env frames barely advanced
- the likely cause is that rollout workers are doing transformer world-model inference on CPU

Relevant code path:

- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)

Interpretation:

- the `distilbert` world-model teacher is strong offline
- but this branch is **not an online rollout-ready representation**

### 6. Relabel-only distillation back to a cheap base-`v4` student

To remove online transformer cost, I added trace relabeling with a BC teacher:

- [rl/relabel_traces.py](/home/luc/rl-nethack/rl/relabel_traces.py)

I relabeled the original `200` base-`v4` train rows with the improved `distilbert` world-model teacher.

Result of relabeling:

- rows: `200`
- changed rows: `4`
- changed rate: **`0.02`**

Then I trained a cheap base-`v4` BC student on the relabeled traces:

- `/tmp/x100_v4_relabel_textdistil_student_bc.pt`
- report:
  - `/tmp/x100_v4_relabel_textdistil_student_bc.pt.teacher_report.json`

Held-out result:

- **`0.9500`**

Interpretation:

- the improved teacher only changes a few actions on the base training set
- relabeling base states is enough to recover the old `0.95` line
- but it is **not enough** to preserve the full `0.9625` gain

### 7. Teacher-logit distillation into a cheap base-`v4` student

Relabeling alone was too weak, so I added direct teacher-logit distillation into BC training.

Relevant source changes:

- [rl/bc_model.py](/home/luc/rl-nethack/rl/bc_model.py)
- [rl/train_bc.py](/home/luc/rl-nethack/rl/train_bc.py)
- [cli.py](/home/luc/rl-nethack/cli.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

On the original honest `200/80` split, distilling from the `distilbert` world-model teacher produced:

- `/tmp/x100_v4_distill_textdistil_c025_t2.pt`: **`0.9625`**
- `/tmp/x100_v4_distill_textdistil_c020_t2_h512.pt`: **`0.9750`**
- `/tmp/x100_v4_distill_textdistil_c025_t2_h512.pt`: **`0.9750`**

Scaling the cheap student further did **not** improve beyond that:

- `/tmp/x100_v4_distill_textdistil_c020_t2_h1024_e80.pt`: **`0.9750`**
- `/tmp/x100_v4_distill_textdistil_c015_t2_h1024_e80.pt`: **`0.9750`**

Interpretation:

- the world-model teacher can now be transferred into a cheap online-usable `v4` policy without losing the gain
- the winning recipe is **teacher logits + modest student scaling**, not relabel-only imitation
- once that transfer path works, further hidden-size scaling gives diminishing returns on this tiny honest split

### 8. Fine-tuning the `distilbert` encoder

I also tested making the text encoder trainable on the original `200/80` split:

- frozen `distilbert`: strong
- trainable `distilbert`: collapsed badly

Trainable result summary:

- action accuracy: about **`0.325`**
- feature cosine: about **`0.10`**
- model collapsed mostly to `north`

Interpretation:

- for this tiny dataset, freezing the language model is clearly better
- fine-tuning the LM is not the right scaling move here

### 9. Short APPO from the cheap `0.975` world-model-distilled student

I promoted the new cheap student to a short teacher-constrained APPO run:

- experiment:
  - `train_dir/rl/appo_v4_distill_textdistil_c025_t2_h512_short_a`
- teacher:
  - `/tmp/x100_v4_distill_textdistil_c025_t2_h512.pt`
- teacher report:
  - `/tmp/x100_v4_distill_textdistil_c025_t2_h512.pt.teacher_report.json`

Trusted result:

- best learned checkpoint:
  - `checkpoint_000000002_512.pth`
  - **`0.9750`**
- final retained checkpoints:
  - `checkpoint_000000017_4352.pth`
  - `checkpoint_000000018_4608.pth`
  - both **`0.8750`**

Interpretation:

- the stronger world-model-distilled teacher transfers cleanly into the online loop
- APPO again preserves the teacher very early
- APPO again drifts sharply by the end of the short run
- this is the same old online-improver problem, now seen from a higher teacher baseline

## What We Learned

### 1. The best world-model improvement is representational scale, not horizon scale

The strongest move was:

- keep the honest original `200/80` split
- keep `horizon=4`
- replace `bert-tiny` with frozen `distilbert-base-uncased`

Longer horizon and synthetic data scaling both underperformed that.

### 2. Direct prediction quality and policy quality are still different

This lesson remains true:

- better prediction metrics do not automatically produce a better teacher
- synthetic scaling improved direct metrics but hurt downstream BC

### 3. The best current world-model branch is an offline teacher builder plus cheap-student distiller

The `distilbert` world-model branch is currently best used for:

- building stronger offline teachers
- distilling those teachers into cheap base-`v4` students
- probing better representations
- generating improved transformed traces

It is **not** yet a good direct online feature path.

### 4. Online transformer world-model inference is the current bottleneck

Putting the `distilbert` world model directly inside rollout workers is too expensive.

So the next useful step is not:

- more APPO tuning on top of live `distilbert` world-model features

It is:

- use the strong offline teacher to produce better cheap teachers
- or distill the world-model-backed teacher into a cheaper representation

## Current Best World-Model Results

### Best offline teacher

- world model: `/tmp/x100_v4_textdistil_state_h4.pt`
- teacher: `/tmp/x100_v4_textdistil_h4_concat_bc.pt`
- held-out trace match: **`0.9625`**

### Best cheap student distilled from that teacher

- student: `/tmp/x100_v4_distill_textdistil_c025_t2_h512.pt`
- held-out trace match: **`0.9750`**

### Best short online bridge from that cheap student

- experiment: `appo_v4_distill_textdistil_c025_t2_h512_short_a`
- best learned checkpoint: **`0.9750`**
- final checkpoint: **`0.8750`**

## Recommendation

If we keep pushing this line, the next repo changes should focus on:

1. Better **teacher-to-student distillation**, not more direct online RL with the transformer world model.
2. Treat the `0.975` cheap student as the new default world-model-derived teacher artifact.
3. Focus the next online work on the improver, not on trying to put the transformer world model directly into rollout workers.
4. Only promote another RL run if a short run from the `0.975` line produces a learned checkpoint above `0.975`.

## Bottom Line

The world-model idea did pay off, but in a narrower and more useful way than “world model fixes RL.”

What it gave us is:

- the strongest honest cheap teacher so far: **`0.9750`**

What it did not yet give us is:

- a practical online rollout representation
- a teacher-beating RL branch
- a justified large-scale RL run

So the right state for the repo now is:

- **world model as teacher builder**
- **cheap student distillation as a working transfer path**
- **online RL as the remaining bottleneck, because it still ties early and then drifts**
