# Proxy Reward Overhaul Report

## Executive Summary

The proxy-reward overhaul was a real engineering success, but not yet a learning breakthrough.

We now have:

- a full offline proxy dataset/training/eval pipeline
- a proxy model integrated into the live RL reward path
- calibrated and bounded proxy scoring that does not numerically explode during rollouts
- short APPO runs that can match the current BC teacher artifact at `0.9375`

We do **not** yet have:

- a proxy model that beats the best existing short scheduled-replay branch at `0.95`
- evidence that the learned proxy is a better online objective than the current teacher-replay baseline
- enough data diversity for the proxy heads to learn robust `search`, survival, and progression structure

So the honest conclusion is:

- the branch is **implemented and working**
- the branch is **not ready to become the mainline RL objective**
- the next step should be **improving proxy quality offline**, not scaling RL

## What Was Implemented

The plan in [PLAN-PROXY-REWARD-OVERHAUL-2026-04-06.md](/home/luc/rl-nethack/PLAN-PROXY-REWARD-OVERHAUL-2026-04-06.md) is now substantially implemented.

New modules:

- [rl/proxy_labels.py](/home/luc/rl-nethack/rl/proxy_labels.py)
- [rl/proxy_dataset.py](/home/luc/rl-nethack/rl/proxy_dataset.py)
- [rl/proxy_model.py](/home/luc/rl-nethack/rl/proxy_model.py)
- [rl/train_proxy_model.py](/home/luc/rl-nethack/rl/train_proxy_model.py)
- [rl/proxy_eval.py](/home/luc/rl-nethack/rl/proxy_eval.py)
- [rl/proxy_report.py](/home/luc/rl-nethack/rl/proxy_report.py)
- [rl/proxy_reward.py](/home/luc/rl-nethack/rl/proxy_reward.py)

Integrated modules:

- [rl/rewards.py](/home/luc/rl-nethack/rl/rewards.py)
- [rl/sf_env.py](/home/luc/rl-nethack/rl/sf_env.py)
- [rl/evaluate.py](/home/luc/rl-nethack/rl/evaluate.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [cli.py](/home/luc/rl-nethack/cli.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

Validation status:

- `uv run pytest -q tests/test_rl_scaffold.py` -> `55 passed`

## Dataset Reality

The current proxy experiments were trained on a very small trace slice:

- training rows: `200`
- held-out rows: `80`

Source traces:

- `/tmp/x100_v4_train_wm_aux.jsonl`
- `/tmp/x100_v4_heldout_wm_aux.jsonl`

Important label facts:

- the held-out action distribution is only `80` rows wide
- `search_context_positive_rate` is effectively zero on held-out data
- this means the first proxy model is being asked to learn a rich reward decomposition from a tiny, weakly supervised slice

This matters because the proxy branch did not fail only because of modeling choices. It is also starved of data.

## Offline Proxy Results

### First failure mode: bad score composition

The initial multi-head proxy produced a degenerate action ranking:

- action top-1 on held-out data was very poor
- predicted actions collapsed toward `west`

That was not because the action head was useless. The action head itself was already usable:

- ranking by `teacher_action_prob` alone gave about `0.7875` held-out top-1

The real problem was score composition:

- regression heads dominated the total score
- the proxy total was not centered or bounded
- rollout states outside the tiny training slice caused unstable totals

### Fixes that were correct

Three fixes were important and correct:

1. Add a strong `teacher_policy` term into proxy scoring.
2. Calibrate totals from the training distribution.
3. Bound the live score with `tanh` to prevent rollout explosions.

After those fixes:

- held-out proxy action top-1 reached `0.8125`
- the live proxy reward became numerically safe enough to use inside RL

Relevant artifacts:

- calibrated proxy model: `/tmp/proxy_tiny_cal.pt`
- calibrated proxy report: `/tmp/proxy_tiny_cal_report.json`

## RL Results

### Baseline teacher references

Current BC teacher artifact:

- [x100_v4_wm_aux_bc_meta.pt.teacher_report.json](/tmp/x100_v4_wm_aux_bc_meta.pt.teacher_report.json)
- held-out trace match: `0.9375`

Current best short scheduled-replay branch:

- [appo_wm_v4_schedreplay_short_a](/home/luc/rl-nethack/train_dir/rl/appo_wm_v4_schedreplay_short_a)
- best trace match: `0.95`

That `0.95` number is the real promotion gate, not `0.9375`.

### Proxy-mixed short runs

Short `mixed_proxy` runs produced:

| Experiment | Best Match |
| --- | ---: |
| `appo_proxymix_short_w002` | `0.9375` |
| `appo_proxymix_short_cal_w01` | `0.9125` |
| `appo_proxymix_short_cal_w0005` | `0.9375` |

Key artifacts:

- [appo_proxymix_short_w002 best trace](/home/luc/rl-nethack/train_dir/rl/appo_proxymix_short_w002/checkpoint_p0/best_trace_match.json)
- [appo_proxymix_short_cal_w01 best trace](/home/luc/rl-nethack/train_dir/rl/appo_proxymix_short_cal_w01/checkpoint_p0/best_trace_match.json)
- [appo_proxymix_short_cal_w0005 best trace](/home/luc/rl-nethack/train_dir/rl/appo_proxymix_short_cal_w0005/checkpoint_p0/best_trace_match.json)

Interpretation:

- the proxy branch can match the current BC teacher artifact
- the proxy branch does not beat the best short teacher-replay branch
- increasing proxy weight too much hurts
- the calibrated-and-bounded proxy is safer, but not stronger yet

## What I Think The Results Mean

### The good news

This branch removed a major ambiguity in the project:

- the repo can now learn and deploy a teacher-derived proxy reward cleanly
- the proxy path is no longer blocked by infrastructure bugs
- the failures are now about signal quality, not missing plumbing

That is real progress.

### The bad news

The learned proxy still does not add enough useful structure beyond the current teacher-replay baseline.

Specifically:

- it does not improve held-out trace match over `0.95`
- it is still trained on too little and too narrow a dataset
- the action/ranking signal is the strongest head, while the regression heads are still noisy
- `search` is effectively unlearnable from the current tiny shard because there are almost no positive labels

### My overall judgment

The proxy branch is **promising but immature**.

If the question is, “Should this become the new mainline RL objective now?” the answer is **no**.

If the question is, “Was this worth building?” the answer is **yes**.

It gave us:

- a better way to express learned reward structure than the old hand-coded scalar alone
- a new offline surface for debugging the reward itself
- a credible path toward teacher-derived short-horizon objectives

## Why It Did Not Beat The Best Branch

I think there are four main reasons.

### 1. The dataset is too small

`200` train rows and `80` held-out rows are enough for a debug loop, not enough for a trustworthy learned proxy.

### 2. The targets are too sparse and unbalanced

The proxy is supposed to learn things like:

- progression
- survival
- loop risk
- resource value
- contextual search value

but the data slice barely contains meaningful positive `search` structure.

### 3. The strongest signal is still teacher action preference

The proxy improved the most when the `teacher_policy` term was made strong. That tells us:

- the action-ranking piece is the cleanest signal
- the other heads are not yet adding enough independent value

### 4. The proxy is competing with a strong baseline

The branch is not competing against a weak hand-coded reward anymore. It is competing against a short scheduled-replay APPO branch that already reaches `0.95`.

That is a harder bar than simply “does proxy reward work at all?”

## Recommendation

Do **not** launch a large-scale proxy-driven run yet.

Instead:

1. Expand the proxy dataset well beyond the `200/80` shard.
2. Make the action/ranking target primary and keep the regression heads auxiliary.
3. Build better labels for:
   - contextual `search`
   - safe progression
   - short-horizon survival risk
4. Re-evaluate the proxy offline first.
5. Only re-enter RL once a proxy branch beats both:
   - offline action-quality gates
   - the short RL gate of `0.95`

## Final Bottom Line

The proxy-reward overhaul is a **successful implementation phase** and an **unsuccessful promotion phase**.

That is not a contradiction.

We successfully built the machinery needed for a teacher-derived learned proxy. But the first proxy is still not strong enough to justify taking over the training objective from the best teacher-replay branch.

So the right next move is:

- keep this code
- improve the proxy offline
- do not scale it prematurely
