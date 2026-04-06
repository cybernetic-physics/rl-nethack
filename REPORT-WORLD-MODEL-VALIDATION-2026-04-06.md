# World Model Validation Report

## Summary

The world-model branch is now much better instrumented than before.

We added:

- richer direct world-model eval metrics
- support for evaluating base world models on `v4+wm_*` trace files
- a downstream BC probe inside world-model evaluation

That instrumentation found two important things:

1. The existing world model was better than it looked downstream, but its direct action prediction was very poor.
2. A retrained action-focused world model improved direct metrics dramatically, but did **not** improve the downstream teacher and did **not** help short online RL.

So the world-model story is now clearer:

- the eval/debug harness is substantially better
- the world model itself is not the current main blocker
- improving direct predictive quality alone did not produce a better RL branch

## Code Changes

Main files changed:

- [rl/world_model_features.py](/home/luc/rl-nethack/rl/world_model_features.py)
- [rl/world_model_eval.py](/home/luc/rl-nethack/rl/world_model_eval.py)
- [rl/train_world_model.py](/home/luc/rl-nethack/rl/train_world_model.py)
- [cli.py](/home/luc/rl-nethack/cli.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

New world-model eval capabilities:

- future-feature MSE and MAE
- future-feature cosine similarity
- current reconstruction MSE and cosine similarity
- action accuracy and top-3 accuracy
- reward MAE and RMSE
- latent dead-fraction / latent spread diagnostics
- per-action prediction metrics
- common action mismatch summaries
- downstream BC trace evaluation from transformed features

Important bug fixed:

- the old world-model eval could not evaluate `/tmp/x100_v4_*_wm_aux.jsonl` traces against a base `v4` world model
- it failed with a feature-dimension mismatch (`379` vs `302`)
- eval now correctly trims appended world-model features back to the base input width when needed

Validation:

- `uv run pytest -q tests/test_rl_scaffold.py` -> `55 passed`

## Baseline World Model

Model:

- `/tmp/x100_v4_world_model.pt`

Held-out direct eval on base `v4` traces:

- input: `/tmp/x100_v4_heldout_traces.jsonl`
- horizon: `4`
- feature MSE: `0.1131`
- feature cosine mean: `0.4439`
- reconstruction MSE: `0.1288`
- reconstruction cosine mean: `0.2616`
- action accuracy: `0.25`
- action top-3 accuracy: `0.8594`
- reward MAE: `1.2185`
- latent dead fraction: `0.2188`

Important failure pattern:

- the action head effectively collapsed to `north`
- common mismatches were:
  - `east -> north`
  - `south -> north`
  - `west -> north`

But the downstream BC probe was still decent:

- mode: `concat_aux`
- held-out trace match: `0.9375`

So the baseline world model was weak as a direct predictive model, but still usable as a feature augmenter.

## Retrained World Models

### Variant A: action-focused, horizon 4

Model:

- `/tmp/x100_v4_world_model_action4_h4.pt`

Training:

- horizon `4`
- epochs `40`
- hidden size `256`
- latent dim `128`
- `action_loss_coef=4.0`
- `reconstruction_loss_coef=0.5`
- `reward_loss_coef=0.25`
- `done_loss_coef=0.25`

Held-out direct eval:

- feature MSE: `0.0677`
- feature cosine mean: `0.7211`
- reconstruction MSE: `0.0905`
- reconstruction cosine mean: `0.6494`
- action accuracy: `0.875`
- action top-3 accuracy: `0.9844`
- reward MAE: `1.0475`
- latent dead fraction: `0.3438`

Interpretation:

- direct predictive quality improved a lot
- action collapse was largely removed
- held-out mismatches were now concentrated mostly in `south -> north`

Downstream BC results:

- `concat_aux`: `0.9375`
- `concat`: `0.9375`
- `replace`: `0.8875`

So this model clearly improved direct world-model quality, but did **not** improve the downstream teacher over the old baseline.

### Variant B: action-focused, horizon 2

Model:

- `/tmp/x100_v4_world_model_action4_h2.pt`

Held-out direct eval:

- feature MSE: `0.0644`
- feature cosine mean: `0.7366`
- reconstruction MSE: `0.0886`
- reconstruction cosine mean: `0.6748`
- action accuracy: `0.9583`
- action top-3 accuracy: `0.9722`
- reward MAE: `0.7608`
- latent dead fraction: `0.3047`

This is the strongest direct world-model of the two retrains.

But downstream BC was worse:

- `concat_aux`: `0.925`

So the best direct predictive model was **not** the best downstream representation model.

## New Teacher Artifacts

From the best retrained world model (`action4_h4`) I built two BC teachers:

`concat` teacher:

- model: `/tmp/x100_v4_wm_action4_h4_concat_bc.pt`
- report: [/tmp/x100_v4_wm_action4_h4_concat_bc.pt.teacher_report.json](/tmp/x100_v4_wm_action4_h4_concat_bc.pt.teacher_report.json)
- held-out trace match: `0.9375`

`concat_aux` teacher:

- model: `/tmp/x100_v4_wm_action4_h4_aux_bc.pt`
- report: [/tmp/x100_v4_wm_action4_h4_aux_bc.pt.teacher_report.json](/tmp/x100_v4_wm_action4_h4_aux_bc.pt.teacher_report.json)
- held-out trace match: `0.925`

So the correct world-model feature mode for this retrained model is still not obviously `concat_aux`. For this retrain, `concat` and `concat_aux` did not beat the old teacher, and `concat` was better than `concat_aux`.

## Short RL Validation

I ran one short scheduled-replay APPO validation using the new `action4_h4` world model with the new `concat` teacher:

- experiment: [appo_wm_action4_h4_concat_short_a](/home/luc/rl-nethack/train_dir/rl/appo_wm_action4_h4_concat_short_a)
- teacher: `/tmp/x100_v4_wm_action4_h4_concat_bc.pt`
- teacher held-out trace match: `0.9375`

Trusted result:

- best checkpoint: [checkpoint_000000018_2304.pth](/home/luc/rl-nethack/train_dir/rl/appo_wm_action4_h4_concat_short_a/checkpoint_p0/checkpoint_000000018_2304.pth)
- best trace match: `0.9125`

That is worse than:

- the teacher itself: `0.9375`
- the older best short scheduled-replay branch: `0.95`

So the better world model did **not** improve the short online RL branch.

## What We Learned

### 1. The old world-model eval was too weak

This was the biggest immediate win from the work.

Before:

- world-model quality looked like a vague MSE number
- augmented traces could crash evaluation

Now:

- we can see direct action collapse
- we can measure latent health
- we can directly test policy usefulness via downstream BC

That materially improves the repo.

### 2. Direct world-model quality and downstream teacher quality are not the same thing

This is the most important scientific result from the loop.

The `action4_h4` and `action4_h2` retrains were much better on direct predictive metrics, but they did not improve the downstream teacher beyond `0.9375`.

So for this repo:

- improving action prediction alone is not enough
- improving feature prediction alone is not enough
- the thing we care about is still downstream policy quality

### 3. The world model is not currently the main online bottleneck

The short RL run with the new world model still regressed:

- teacher: `0.9375`
- best learned checkpoint: `0.9125`

That means:

- the world model can be improved
- but the online improver is still the weaker part of the stack

## Decision

Do **not** promote the new world-model branch to a large-scale RL run yet.

Reasons:

- no downstream teacher improvement over the old baseline
- short RL still regressed below the teacher
- no trace-match gain beyond the existing best short branch

## Recommended Next Move

Keep the new evals. They are worth the code.

But for world-model work specifically, the next changes should target **representation usefulness**, not just raw prediction quality. Concretely:

1. Keep using downstream BC trace match as the promotion gate.
2. Treat direct action/reconstruction metrics as supporting diagnostics, not the final goal.
3. If we keep iterating on the world model, try objectives that are closer to policy usefulness:
   - stronger action-conditioned ranking targets
   - teacher-consistency auxiliary targets
   - short-horizon direction-choice targets
4. Do not scale RL from a new world model unless:
   - downstream BC improves
   - and a short teacher-replay APPO run does not regress below the teacher

## Bottom Line

The world-model work produced a much better **measurement system** and a better **direct predictive model**.

It did **not** yet produce a better online RL branch.

That means this was a successful debug-and-eval phase, not yet a successful promotion phase.
