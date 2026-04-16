# Sequence World Model Report

Date: 2026-04-16

## What changed

Implemented a new sequence world-model path alongside the legacy one-step predictor:

- [rl/sequence_world_model.py](/home/luc/rl-nethack/rl/sequence_world_model.py)
- [rl/sequence_world_model_dataset.py](/home/luc/rl-nethack/rl/sequence_world_model_dataset.py)
- [rl/train_sequence_world_model.py](/home/luc/rl-nethack/rl/train_sequence_world_model.py)
- [rl/sequence_world_model_eval.py](/home/luc/rl-nethack/rl/sequence_world_model_eval.py)
- [rl/world_model_planner.py](/home/luc/rl-nethack/rl/world_model_planner.py)
- [rl/world_model_calibration.py](/home/luc/rl-nethack/rl/world_model_calibration.py)
- [tests/test_sequence_world_model.py](/home/luc/rl-nethack/tests/test_sequence_world_model.py)

The new path adds:

- sequence-window dataset with context burn-in plus rollout supervision
- RSSM-style stochastic latent + recurrent hidden state
- open-loop rollout metrics by horizon
- simple binary temperature scaling for done calibration
- latent CEM planner with valid-action penalties

## Verification

Tests run:

- `.venv/bin/pytest -q tests/test_sequence_world_model.py`
- `.venv/bin/pytest -q tests/test_sequence_world_model.py tests/test_rl_scaffold.py -k 'world_model_train_and_eval_smoke or world_model_examples_build_k_step_windows or world_model_uses_state_only_prompt_text'`

Result:

- `3 passed`
- `3 passed, 89 deselected`

## Training loop 1

Train:

- input: `data/tracefix_v2_explore_traces.jsonl`
- config: `context_len=4`, `rollout_horizon=6`, `epochs=30`, `hidden=128`, `latent=64`, `text=hash`
- artifact: [seqwm_v1.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v1.pt)
- report: [seqwm_v1_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v1_train.json)

Held-out eval:

- input: `data/validate_v2_explore_traces.jsonl`
- feature MSE: `0.0453`
- reward MAE: `0.1259`
- horizon-6 feature MSE: `0.0469`

Observation:

- sequence rollout quality is materially better aligned with the research doc than the old one-step predictor
- planning was still weak because the planner exploited implausible actions

## Training loop 2

Change made after loop 1:

- planner now penalizes low predicted action validity and respects root `allowed_actions`
- second train increased `valid_action_loss_coef` from `0.25` to `1.0`
- second train increased `kl_loss_coef` from `0.1` to `0.2`

Train artifact:

- [seqwm_v2_valid1.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v2_valid1.pt)
- [seqwm_v2_valid1_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v2_valid1_train.json)

Held-out eval:

- feature MSE: `0.0448`
- reward MAE: `0.1407`
- horizon-6 feature MSE: `0.0463`

Observation:

- loop 2 improved held-out feature prediction slightly
- loop 2 worsened held-out reward error
- planner behavior became less pathological, but still not good enough to claim planning-grade NetHack control

## Training loop 3

Change made after loop 2:

- added bounded latent overshooting loss to regularize multi-step prior/posterior consistency
- trained on merged `tracefix_v2_explore_traces.jsonl` + `run_v2_explore_traces.jsonl`
- config:
  - `valid_action_loss_coef=1.0`
  - `kl_loss_coef=0.2`
  - `overshooting_loss_coef=0.15`
  - `overshooting_distance=3`

Train artifact:

- [seqwm_v3_overshoot_merge.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v3_overshoot_merge.pt)
- [seqwm_v3_overshoot_merge_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v3_overshoot_merge_train.json)
- merged input: [merged_v2_train.jsonl](/home/luc/rl-nethack/output/sequence_wm_runs/merged_v2_train.jsonl)

Held-out eval on `data/validate_v2_explore_traces.jsonl`:

- feature MSE: `0.0432`
- reward MAE: `0.1508`
- horizon-6 feature MSE: `0.0448`
- KL mean: `0.0042`
- overshooting KL mean: `0.0042`

Observation:

- loop 3 is the best held-out feature result so far on the external validation trace
- loop 3 improves external reward error relative to loop 2
- loop 3 greatly reduces latent KL relative to loops 1 and 2
- planner output is still too mode-collapsed to claim robust search quality, but it is less obviously nonsense than the earliest planner probes

## Training loop 4

Change made after loop 3:

- added discounted return-to-go supervision for the latent value head
- planner now uses terminal bootstrap value in addition to summed predicted rewards

Train artifact:

- [seqwm_v4_value_overshoot.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v4_value_overshoot.pt)
- [seqwm_v4_value_overshoot_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v4_value_overshoot_train.json)

Held-out eval on `data/validate_v2_explore_traces.jsonl`:

- feature MSE: `0.0430`
- reward MAE: `0.1302`
- value MAE: `0.4934`

Observation:

- loop 4 is the best external world-model run so far
- value supervision improved external reward error versus loop 3
- planner outputs became a bit more plausible, but still not robust enough to trust for actual NetHack control

## Training loop 5

Change made after loop 4:

- training now supports checkpoint selection by a planning-oriented validation proxy
- tried `selection_metric=planning_proxy` instead of `feature_mse`

Train artifact:

- [seqwm_v5_value_planselect.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v5_value_planselect.pt)
- [seqwm_v5_value_planselect_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v5_value_planselect_train.json)

External held-out eval:

- feature MSE: `0.0442`
- reward MAE: `0.2031`
- value MAE: `0.5971`

Warning carried forward:

- planning-proxy checkpoint selection improved the in-split validation score but hurt external validation badly
- this should be treated as an overfitting failure mode, not a success
- current best external model remains `seqwm_v4_value_overshoot.pt`

## Training loop 6

Change made after loop 5:

- added a planner-trace action-prior head trained directly on `planner_trace.total`
- planner root scoring now includes the learned action prior
- added replay planner evaluation:
  - [rl/sequence_planner_eval.py](/home/luc/rl-nethack/rl/sequence_planner_eval.py)

Train artifact:

- [seqwm_v6_actionprior.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v6_actionprior.pt)
- [seqwm_v6_actionprior_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v6_actionprior_train.json)

Held-out prediction eval:

- feature MSE: `0.0363` on the internal split used by training
- external prediction metrics were mixed relative to `seqwm_v4`

Replay planner diagnostic on first 10 held-out rows with cheap CEM settings:

`seqwm_v4_value_overshoot.pt`

- teacher-best recovery: `0.2`
- mean rank correlation: `0.024`
- mean teacher gap from predicted best: `0.275`

`seqwm_v6_actionprior.pt`

- teacher-best recovery: `0.4`
- mean rank correlation: `0.214`
- mean teacher gap from predicted best: `0.045`

Observation:

- direct planner-trace supervision improved offline action ranking substantially on the cheap replay diagnostic
- this is the first loop in which planner utility improved even though plain prediction metrics were not uniformly better
- the next serious loop should expand replay planner evaluation beyond the cheap 10-row diagnostic and then retune the model around that target rather than around feature reconstruction

## Training loop 7

Change made after loop 6:

- increased `planner_action_loss_coef`
- added planner-trace-aware checkpoint selection options
- added explicit random seeds to replay planner eval to make comparisons reproducible

Train artifact:

- [seqwm_v7_actionprior_select.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v7_actionprior_select.pt)
- [seqwm_v7_actionprior_select_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v7_actionprior_select_train.json)

Warning:

- stronger planner-action weighting damaged standard prediction metrics badly
- this branch is not a better world-model in the usual predictive sense

But on seeded cheap replay planner eval (`max_rows=10`, `population_size=16`, `iterations=2`, `random_seed=7`):

`seqwm_v6_actionprior.pt`

- teacher-best recovery: `0.1`
- mean rank correlation: `0.106`
- mean teacher gap from predicted best: `0.25`

`seqwm_v7_actionprior_select.pt`

- teacher-best recovery: `0.4`
- mean rank correlation: `0.295`
- mean teacher gap from predicted best: `0.23`

Observation:

- loop 7 is another split-objective result
- as a predictor, it regressed sharply
- as a seeded offline action-ranker on the cheap replay diagnostic, it improved over loop 6
- the repo now clearly needs multi-objective model selection with separate predictive and planner-facing scorecards rather than a single “best model” label

## Current conclusion

This repo now has a real sequence world-model baseline with training, evaluation, calibration, and planning hooks.

It is a meaningful step toward the long-horizon research direction, but it is not yet a SOTA NetHack solver. The current blocker is not basic rollout learning anymore; it is planning utility under sparse data and weak action-validity / uncertainty signals.

The next serious upgrade should be:

- latent overshooting or multi-step consistency beyond one-step KL
- larger and more diverse sequential training data than the current local `v2` slices
- stronger validity/risk modeling for planner scoring
- planner evaluation against actual branch ranking or receding-horizon execution, not just sampled action sequences

## Training loop 8

Change made after loop 7:

- added `rl/sequence_benchmark.py` for paired predictive + planner replay scorecards
- trained a compromise setting between `v6` and `v7` instead of pushing planner-action weighting harder
- config:
  - `planner_action_loss_coef=0.35`
  - `selection_metric=feature_mse`

Train artifact:

- [seqwm_v8_compromise.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v8_compromise.pt)
- [seqwm_v8_compromise_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v8_compromise_train.json)

External held-out eval on `data/validate_v2_explore_traces.jsonl`:

- feature MSE: `0.0445`
- reward MAE: `0.1513`
- value MAE: `0.5141`

Seeded replay planner eval on 20 held-out rows (`population_size=16`, `iterations=2`, `random_seed=7`):

- teacher-best recovery: `0.35`
- mean rank correlation: `0.176`
- mean teacher gap from predicted best: `0.2575`

Observation:

- `v8` became the best compromise checkpoint so far
- it improved seeded exact-match rate over `v6` and matched or beat earlier compromise behavior without the `v7` collapse
- but it still did not materially improve full ranking quality, which suggested the planner target needed a better loss than raw score regression

## Training loop 9

Change made after loop 8:

- added pairwise planner ranking supervision and validation metrics:
  - `planner_pairwise_loss`
  - `planner_pairwise_accuracy`
- added planner-rank-aware checkpoint selection

Train artifact:

- [seqwm_v9_pairrank.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v9_pairrank.pt)
- [seqwm_v9_pairrank_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v9_pairrank_train.json)

External held-out eval on `data/validate_v2_explore_traces.jsonl`:

- feature MSE: `0.0553`
- reward MAE: `0.0796`
- value MAE: `0.5569`

Seeded replay planner eval on 20 held-out rows:

- teacher-best recovery: `0.1`
- mean rank correlation: `0.085`
- mean teacher gap from predicted best: `0.415`

Warning carried forward:

- pairwise planner ranking overfit badly on this tiny corpus
- it improved the training split metrics it was asked to optimize and failed badly on held-out replay ranking
- this is now explicit folklore for the branch: “pairwise planner ranking is not trustworthy here without stronger regularization or more data”

## Training loop 10

Change made after loop 9:

- added listwise planner-policy supervision and validation metrics:
  - `planner_policy_ce`
  - `planner_policy_top1`
- checkpoint selection can now optimize a mixed planner-policy proxy instead of pure pairwise ranking
- this is closer to the MuZero-style idea of supervising a policy target over root actions rather than regressing raw totals

Train artifact:

- [seqwm_v10_policy.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v10_policy.pt)
- [seqwm_v10_policy_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v10_policy_train.json)

External held-out eval on `data/validate_v2_explore_traces.jsonl`:

- feature MSE: `0.0542`
- reward MAE: `0.1265`
- value MAE: `0.5122`

Seeded replay planner eval on 20 held-out rows:

- teacher-best recovery: `0.3`
- mean rank correlation: `0.299`
- mean teacher gap from predicted best: `0.2375`

Observation:

- `v10` is the best planner-ranking checkpoint so far on the held-out seeded replay benchmark
- it materially beats `v8` on ranking quality and teacher gap while avoiding the catastrophic external collapse seen in `v9`
- it is still clearly worse than `v8` as a predictive world model

## Updated conclusion

The repo now has three distinct fronts rather than one “best model”:

- best predictive checkpoint on the current external trace remains `seqwm_v4_value_overshoot.pt`
- best compromise checkpoint is `seqwm_v8_compromise.pt`
- best seeded replay planner-ranking checkpoint is `seqwm_v10_policy.pt`

Current folklore to preserve across future iterations:

- warning: planning-proxy checkpoint selection can overfit and harm external performance
- warning: pairwise planner ranking loss collapsed on held-out replay ranking under the current data scale
- useful: direct planner-trace supervision helps, but listwise planner-policy targets are more stable than pairwise ranking here

The next serious loop should bias toward:

- listwise planner-policy supervision rather than pairwise ranking
- multi-objective reporting instead of a single scalar “best”
- larger replay-eval slices so planner improvements are harder to fake
- more data before pushing planner-only losses harder

## Training loop 11

Change made after loop 10:

- added `planner_policy_target_temperature` so planner-policy supervision can be softened instead of forcing a sharp target distribution
- exposed `root_prior_coef` in [rl/sequence_planner_eval.py](/home/luc/rl-nethack/rl/sequence_planner_eval.py) so replay evaluation can measure whether stronger use of the learned policy prior actually helps
- trained a softer policy-regularized compromise run instead of another planner-heavy run

Train artifact:

- [seqwm_v11_softpolicy.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v11_softpolicy.pt)
- [seqwm_v11_softpolicy_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v11_softpolicy_train.json)

Config highlights:

- `planner_policy_loss_coef=0.1`
- `planner_policy_target_temperature=2.0`
- `selection_metric=feature_mse`

External held-out eval on `data/validate_v2_explore_traces.jsonl`:

- feature MSE: `0.0445`
- reward MAE: `0.2638`
- value MAE: `0.7368`

Seeded replay planner eval on 20 held-out rows:

`root_prior_coef=0.25`

- teacher-best recovery: `0.3`
- mean rank correlation: `0.1764`
- mean teacher gap from predicted best: `0.2625`

`root_prior_coef=0.5`

- teacher-best recovery: `0.35`
- mean rank correlation: `0.2107`
- mean teacher gap from predicted best: `0.225`

Comparison note:

- `v11` successfully recovered `v8`-level external feature MSE
- but its planner ranking stayed clearly below `v10`
- stronger root-prior usage helped `v11` materially

Planner-prior sweep folklore:

- `v11` benefits from `root_prior_coef=0.5`
- `v10` changes only slightly between `0.25` and `0.5`
- `v8` does not meaningfully improve from the stronger root prior

Updated interpretation:

- a better learned planner policy does not automatically mean “use more prior everywhere”
- the benefit of stronger prior usage depends on the checkpoint
- softened planner-policy supervision is a useful compromise tool, but it did not beat `v10` on planner ranking

## Planner tuning pass

Change made after loop 11:

- added [rl/sequence_planner_tune.py](/home/luc/rl-nethack/rl/sequence_planner_tune.py) to sweep replay-planner hyperparameters instead of fixing them by hand
- the tuner currently sweeps:
  - `bootstrap_value_coef`
  - `root_prior_coef`

Held-out replay tuning result on 20 seeded rows:

`seqwm_v8_compromise.pt`

- best tuned setting: `bootstrap_value_coef=0.5`, `root_prior_coef=0.25`
- best tuned score remained the same conclusion as before: good compromise, not best ranker

`seqwm_v10_policy.pt`

- best tuned setting: `bootstrap_value_coef=0.25`, `root_prior_coef=0.0`
- best tuned metrics:
  - teacher-best recovery: `0.3`
  - mean rank correlation: `0.3146`
  - mean teacher gap from predicted best: `0.25`

`seqwm_v11_softpolicy.pt`

- best tuned setting: `bootstrap_value_coef=0.5`, `root_prior_coef=0.5`
- best tuned metrics:
  - teacher-best recovery: `0.35`
  - mean rank correlation: `0.185`
  - mean teacher gap from predicted best: `0.225`

Important folklore:

- `v10` is strongest when planner prior usage is weaker, not stronger
- `v11` is strongest when planner prior usage is stronger
- planner hyperparameters are checkpoint-dependent and should not be treated as universal constants

## Training loop 12

Change made after tuning:

- tested whether `v10` was mostly a bad checkpoint-selection outcome rather than a bad loss recipe
- trained with the stronger listwise planner-policy loss from `v10`
- changed checkpoint selection back to `feature_mse`

Train artifact:

- [seqwm_v12_policy_featselect.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v12_policy_featselect.pt)
- [seqwm_v12_policy_featselect_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v12_policy_featselect_train.json)

External held-out eval on `data/validate_v2_explore_traces.jsonl`:

- feature MSE: `0.0440`
- reward MAE: `0.1571`
- value MAE: `0.5133`

Tuned replay planner result:

- best tuned setting: `bootstrap_value_coef=0.25`, `root_prior_coef=0.5`
- best tuned metrics:
  - teacher-best recovery: `0.3`
  - mean rank correlation: `0.1321`
  - mean teacher gap from predicted best: `0.3075`

Observation:

- `v12` is a useful negative result
- better checkpoint selection recovered external predictive quality relative to `v10`
- but it did not recover planner ranking quality
- so the planner advantage of `v10` is not just “it picked the wrong epoch”; it is genuinely tied to the planner-oriented training pressure

## Training loop 13

Change made after loop 12:

- added `planner_policy_warmup_epochs` to [rl/train_sequence_world_model.py](/home/luc/rl-nethack/rl/train_sequence_world_model.py)
- the planner-policy loss is now ramped in gradually instead of hitting at full strength from epoch 1
- this tests whether the model can learn cleaner predictive dynamics first and preserve more planner quality later

Train artifact:

- [seqwm_v13_policy_warmup.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v13_policy_warmup.pt)
- [seqwm_v13_policy_warmup_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v13_policy_warmup_train.json)

Config highlights:

- `planner_policy_loss_coef=0.2`
- `planner_policy_target_temperature=1.0`
- `planner_policy_warmup_epochs=20`
- `selection_metric=feature_mse`

External held-out eval on `data/validate_v2_explore_traces.jsonl`:

- feature MSE: `0.0439`
- reward MAE: `0.1390`
- value MAE: `0.4159`

Tuned replay planner result:

- best tuned setting: `bootstrap_value_coef=0.5`, `root_prior_coef=0.25`
- best tuned metrics:
  - teacher-best recovery: `0.3`
  - mean rank correlation: `0.2189`
  - mean teacher gap from predicted best: `0.2375`

Observation:

- `v13` is a real improvement over `v12`
- warmup recovered much better external predictive quality while retaining materially better planner ranking than `v12`
- but it still does not beat tuned `v10` on planner ranking

Updated folklore:

- planner-policy warmup helps the compromise frontier
- warmup is not enough to replace planner-oriented training when the goal is maximum replay ranking quality
- current fronts remain:
  - predictive best: `seqwm_v4_value_overshoot.pt`
  - compromise best: `seqwm_v13_policy_warmup.pt`
  - planner-ranking best: tuned `seqwm_v10_policy.pt`

## Training loop 14

Change made after loop 13:

- added optional uncertainty-style adaptive multi-task balancing in [rl/train_sequence_world_model.py](/home/luc/rl-nethack/rl/train_sequence_world_model.py)
- the trainer can now learn scalar log-variance terms for:
  - feature
  - reward
  - value
  - planner action
  - planner policy
  - done
  - valid action
- this was tested on top of the `v13` warmup recipe

Train artifact:

- [seqwm_v14_adaptive_balance.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v14_adaptive_balance.pt)
- [seqwm_v14_adaptive_balance_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v14_adaptive_balance_train.json)

External held-out eval on `data/validate_v2_explore_traces.jsonl`:

- feature MSE: `0.04394`
- reward MAE: `0.13278`
- value MAE: `0.41618`

Tuned replay planner result:

- best tuned setting: `bootstrap_value_coef=0.75`, `root_prior_coef=0.5`
- best tuned metrics:
  - teacher-best recovery: `0.25`
  - mean rank correlation: `0.19536`
  - mean teacher gap from predicted best: `0.28`

Observation:

- `v14` slightly improved held-out reward error relative to `v13`
- `v14` roughly tied `v13` on held-out feature prediction
- but `v14` regressed materially on tuned replay planner ranking

Updated folklore:

- adaptive loss balancing is not a free lunch here
- it can smooth predictive tradeoffs without preserving planner quality
- the current compromise frontier still belongs to `v13`, not `v14`

## Training loop 15

Change made after loop 14:

- added `planner_compromise_proxy` checkpoint selection to [rl/train_sequence_world_model.py](/home/luc/rl-nethack/rl/train_sequence_world_model.py)
- this proxy keeps planner-policy pressure in selection, but anchors it to predictive quality instead of optimizing planner CE alone
- extended [rl/sequence_benchmark.py](/home/luc/rl-nethack/rl/sequence_benchmark.py) so benchmark summaries can read multi-seed planner JSON directly

Verification:

- `.venv/bin/pytest -q tests/test_sequence_world_model.py`
- result: `6 passed`

Train artifact:

- [seqwm_v15_compromise_select.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v15_compromise_select.pt)
- [seqwm_v15_compromise_select_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v15_compromise_select_train.json)
- external eval: [seqwm_v15_compromise_select_eval.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v15_compromise_select_eval.json)
- multi-seed replay: [seqwm_v15_compromise_select_multiseed.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v15_compromise_select_multiseed.json)

Config highlights:

- `planner_policy_loss_coef=0.2`
- `planner_policy_warmup_epochs=20`
- `selection_metric=planner_compromise_proxy`

External held-out eval on `data/validate_v2_explore_traces.jsonl`:

- feature MSE: `0.04810`
- reward MAE: `0.09618`
- value MAE: `0.34286`
- planner action MAE: `0.36271`
- planner policy CE: `1.78304`

Tuned multi-seed replay planner summary on 20 held-out rows:

- mean exact match: `0.45`
- std exact match: `0.1080`
- mean rank correlation: `0.16369`
- std rank correlation: `0.07443`
- mean teacher gap: `0.18583`
- std teacher gap: `0.06213`

Best per-seed settings:

- seed `7`: `bootstrap_value_coef=0.75`, `root_prior_coef=0.0`, rank correlation `0.26893`, teacher gap `0.245`
- seed `17`: `bootstrap_value_coef=0.25`, `root_prior_coef=0.25`, rank correlation `0.10929`, teacher gap `0.2125`
- seed `27`: `bootstrap_value_coef=0.25`, `root_prior_coef=0.0`, rank correlation `0.11286`, teacher gap `0.10`

Observation:

- `v15` is not the new planner-ranking best by mean multi-seed rank correlation
- but it materially improves external reward and value prediction relative to the earlier compromise frontier
- and it posts the best mean multi-seed teacher gap seen so far on this benchmark

Updated frontier after the multi-seed comparison:

- predictive best by feature error: `seqwm_v4_value_overshoot.pt`
- compromise best by external reward/value plus planner robustness: `seqwm_v15_compromise_select.pt`
- planner-ranking best by mean multi-seed rank correlation: `seqwm_v10_policy.pt`
- most seed-stable compromise before `v15`: `seqwm_v13_policy_warmup.pt`

Updated folklore:

- single-seed replay scores were overstating `v10`’s edge
- `v10` still wins slightly on mean multi-seed rank correlation, but its advantage is much smaller than the seed-7 story suggested
- compromise-oriented checkpoint selection can buy much better external reward/value quality and lower teacher gap without winning the ranking metric outright

## Training loop 16

Change made after loop 15:

- added latent-uncertainty-aware planner scoring in [rl/world_model_planner.py](/home/luc/rl-nethack/rl/world_model_planner.py)
- [rl/sequence_world_model.py](/home/luc/rl-nethack/rl/sequence_world_model.py) rollout now exposes `pred_latent_uncertainty` from the RSSM prior log-variance
- replay evaluation and tuning now sweep `uncertainty_coef` through:
  - [rl/sequence_planner_eval.py](/home/luc/rl-nethack/rl/sequence_planner_eval.py)
  - [rl/sequence_planner_tune.py](/home/luc/rl-nethack/rl/sequence_planner_tune.py)
  - [rl/sequence_planner_multiseed.py](/home/luc/rl-nethack/rl/sequence_planner_multiseed.py)
- trained a higher-KL checkpoint to make the latent uncertainty signal more meaningful during planning

Verification:

- `uv run pytest -q tests/test_sequence_world_model.py`
- result: `6 passed`

Train artifact:

- [seqwm_v16_uncertainty_kl.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v16_uncertainty_kl.pt)
- [seqwm_v16_uncertainty_kl_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v16_uncertainty_kl_train.json)
- external eval: [seqwm_v16_uncertainty_kl_eval.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v16_uncertainty_kl_eval.json)
- multi-seed replay: [seqwm_v16_uncertainty_kl_multiseed.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v16_uncertainty_kl_multiseed.json)

Config highlights:

- `selection_metric=planner_compromise_proxy`
- `kl_loss_coef=0.3`
- planner replay tuning now includes `uncertainty_coef in {0.0, 0.25, 0.5, 1.0}`

External held-out eval on `data/validate_v2_explore_traces.jsonl`:

- feature MSE: `0.05053`
- reward MAE: `0.09485`
- value MAE: `0.43723`
- planner action MAE: `0.38660`
- planner policy CE: `1.78550`
- KL mean: `0.03232`
- overshooting KL mean: `0.03374`

Observation on predictive quality:

- `v16` slightly improved external reward error over `v15`
- but it clearly worsened feature and value prediction
- the stronger KL term made the latent prior much tighter, but not better overall as a predictive compromise model

Focused seed-7 uncertainty ablation:

`v15`, fixed planner setting (`bootstrap_value_coef=0.5`, `root_prior_coef=0.25`)

- `uncertainty_coef=0.0`: rank correlation `0.25786`, teacher gap `0.23`
- `uncertainty_coef=0.5`: rank correlation `0.25214`, teacher gap `0.23`

`v16`, same fixed planner setting

- `uncertainty_coef=0.0`: rank correlation `0.14857`, teacher gap `0.20`
- `uncertainty_coef=0.5`: rank correlation `0.16000`, teacher gap `0.20`

Interpretation:

- the uncertainty penalty is mostly neutral-to-harmful on `v15`
- but it helps `v16` directionally, which suggests the tighter latent prior made uncertainty-aware planning more usable

Tuned multi-seed replay summary:

- mean exact match: `0.43333`
- std exact match: `0.02357`
- mean rank correlation: `0.15095`
- std rank correlation: `0.06339`
- mean teacher gap: `0.21667`
- std teacher gap: `0.03125`

Best per-seed settings:

- seed `7`: `bootstrap_value_coef=0.25`, `root_prior_coef=0.5`, `uncertainty_coef=1.0`, rank correlation `0.24000`, teacher gap `0.1875`
- seed `17`: `bootstrap_value_coef=0.25`, `root_prior_coef=0.5`, `uncertainty_coef=0.0`, rank correlation `0.09750`, teacher gap `0.26`
- seed `27`: `bootstrap_value_coef=0.25`, `root_prior_coef=0.5`, `uncertainty_coef=1.0`, rank correlation `0.11536`, teacher gap `0.2025`

Updated folklore:

- uncertainty penalties are checkpoint-dependent, just like root-prior usage
- `v16` wants strong uncertainty penalization on two of three seeds, which is evidence that the planner-side change is real
- but the stronger-KL checkpoint did not beat `v15` as a compromise model and did not beat the earlier planner frontier either
- the branch still needs better uncertainty estimation, not just more KL pressure

## Training loop 17

Change made after loop 16:

- added stochastic rollout disagreement to [rl/sequence_world_model.py](/home/luc/rl-nethack/rl/sequence_world_model.py)
- planner scoring in [rl/world_model_planner.py](/home/luc/rl-nethack/rl/world_model_planner.py) can now penalize:
  - latent uncertainty
  - reward disagreement across sampled rollouts
  - value disagreement across sampled rollouts
- replay evaluation/tuning now accept:
  - `disagreement_coef`
  - `rollout_samples`

Verification:

- `uv run pytest -q tests/test_sequence_world_model.py`
- result: `6 passed`

Train artifact:

- [seqwm_v17_disagree_overshoot.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v17_disagree_overshoot.pt)
- [seqwm_v17_disagree_overshoot_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v17_disagree_overshoot_train.json)
- external eval: [seqwm_v17_disagree_overshoot_eval.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v17_disagree_overshoot_eval.json)

Config highlights:

- `selection_metric=planner_compromise_proxy`
- `kl_loss_coef=0.2`
- `overshooting_loss_coef=0.25`

External held-out eval on `data/validate_v2_explore_traces.jsonl`:

- feature MSE: `0.04367`
- reward MAE: `0.12742`
- value MAE: `0.41990`
- planner action MAE: `0.40522`
- planner policy CE: `1.79742`
- KL mean: `0.07234`
- overshooting KL mean: `0.07418`

Observation on predictive quality:

- `v17` recovered much stronger external feature quality than `v16`
- but it did not beat `v15` on reward/value prediction
- this is a useful rollout-consistency checkpoint, not a new best compromise checkpoint

Focused disagreement probe on 10 held-out rows (`seed=7`, `bootstrap_value_coef=0.75`, `root_prior_coef=0.0`, `uncertainty_coef=0.5`, `rollout_samples=3`):

`v15`, no disagreement penalty

- exact match `0.1`
- rank correlation `0.18643`
- teacher gap `0.275`

`v15`, `disagreement_coef=0.25`

- exact match `0.2`
- rank correlation `0.29786`
- teacher gap `0.23`

`v17`, no disagreement penalty

- exact match `0.2`
- rank correlation `0.16500`
- teacher gap `0.24`

`v17`, `disagreement_coef=0.25`

- exact match `0.2`
- rank correlation `0.15857`
- teacher gap `0.235`

Interpretation:

- disagreement-aware planning materially helps `v15` on this focused probe
- the same penalty does not help `v17`
- the planner-side change looks more valuable than the new checkpoint itself

Updated folklore:

- disagreement penalties are also checkpoint-dependent
- better rollout consistency alone did not produce a better planner-facing model
- the best next move is likely to re-evaluate the stronger `v15` frontier with disagreement-aware tuning at larger scale, rather than assume the newest checkpoint is best

## V15 Disagreement Retune

Follow-up after loop 17:

- re-ran the stronger existing `v15` frontier with disagreement-aware replay planning instead of training yet another checkpoint
- the first 10-row probe had suggested disagreement might materially help `v15`
- to validate that, a fixed-setting multiseed comparison was run on 20 held-out rows with:
  - `bootstrap_value_coef=0.75`
  - `root_prior_coef=0.0`
  - `uncertainty_coef=0.5`
  - `rollout_samples=3`
  - disagreement compared at `0.0` vs `0.25`

Artifacts:

- no disagreement: [seqwm_v15_multiseed_fixed_disagree0.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v15_multiseed_fixed_disagree0.json)
- disagreement `0.25`: [seqwm_v15_multiseed_fixed_disagree25.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_v15_multiseed_fixed_disagree25.json)

Multiseed result without disagreement:

- mean exact match: `0.36667`
- mean rank correlation: `0.11810`
- mean teacher gap: `0.25083`

Multiseed result with `disagreement_coef=0.25`:

- mean exact match: `0.31667`
- mean rank correlation: `0.09702`
- mean teacher gap: `0.29083`

Interpretation:

- the earlier 10-row disagreement win did not survive the 20-row three-seed robustness check
- at this fixed planner recipe, disagreement penalization hurt `v15` overall
- the disagreement signal is therefore not ready to be promoted to the default planner setting

Updated folklore:

- cheap probes can flatter planner tweaks that do not survive cross-seed validation
- for `v15`, disagreement-aware planning remains an interesting idea, but not a robust win yet

## Large Corpus Execution

Follow-up after the next-steps plan:

- implemented full-trajectory corpus tooling in [rl/trace_corpus.py](/home/luc/rl-nethack/rl/trace_corpus.py)
- added tests in [tests/test_trace_corpus.py](/home/luc/rl-nethack/tests/test_trace_corpus.py)
- used that tooling to audit and split the larger local pipeline trace corpus

Corpus artifacts:

- audit: [pipeline106_corpus_audit.json](/home/luc/rl-nethack/output/sequence_wm_runs/pipeline106_corpus_audit.json)
- manifest: [pipeline106_manifest.json](/home/luc/rl-nethack/output/sequence_wm_runs/pipeline106_manifest.json)
- train split: [pipeline106_train.jsonl](/home/luc/rl-nethack/output/sequence_wm_runs/pipeline106_train.jsonl)
- eval split: [pipeline106_eval.jsonl](/home/luc/rl-nethack/output/sequence_wm_runs/pipeline106_eval.jsonl)

Pipeline corpus summary:

- source rows: `2880`
- source episodes: `120`
- feature dim: `106`
- planner-trace coverage: `100%`

Episode-level split:

- train rows: `2448`
- train episodes: `102`
- held-out eval rows: `432`
- held-out eval episodes: `18`

Important note:

- this is a separate 106-dim full-trajectory branch
- it was not merged with the 160-dim `v2` branch

## Training loop pipeline106-v1

Change made after the corpus build:

- retrained the current best compromise sequence recipe on the larger 106-dim full-trajectory corpus

Verification:

- `uv run pytest -q tests/test_trace_corpus.py tests/test_sequence_world_model.py`
- result: `8 passed`

Train artifact:

- [seqwm_pipeline106_v1.pt](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_pipeline106_v1.pt)
- [seqwm_pipeline106_v1_train.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_pipeline106_v1_train.json)
- external eval: [seqwm_pipeline106_v1_eval.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_pipeline106_v1_eval.json)
- focused multiseed replay: [seqwm_pipeline106_v1_multiseed_fixed.json](/home/luc/rl-nethack/output/sequence_wm_runs/seqwm_pipeline106_v1_multiseed_fixed.json)

Training scale:

- train windows: `1215`
- internal val windows: `315`

Held-out eval on `pipeline106_eval.jsonl`:

- feature MSE: `0.03014`
- reward MAE: `0.19831`
- value MAE: `0.70052`
- planner action MAE: `0.64027`
- planner policy CE: `1.74246`
- planner pairwise accuracy: `0.80956`
- KL mean: `0.02235`

Observation on predictive quality:

- this is the first sequence world-model run in the repo with enough data that predictor metrics look genuinely stable instead of toy-sized
- feature prediction is much better than the tiny local branch
- planner teacher imitation metrics inside the held-out pipeline split are also strong

Focused multiseed replay on held-out pipeline episodes (`max_rows=20`, fixed planner recipe):

- mean exact match: `0.11667`
- mean rank correlation: `-0.02381`
- mean teacher gap: `0.49667`

Interpretation:

- scaling the full-trajectory corpus clearly improved the predictive side of the world model
- but the old planner recipe did not transfer cleanly to the larger pipeline branch
- so the branch now has a sharper split:
  - data scale fixed the “tiny-world-model” problem
  - planner scoring is still the main failure mode

Updated folklore:

- larger full-trajectory data is necessary and immediately useful for predictor stability
- stronger predictive metrics do not automatically produce a good replay planner
- after scaling data, planner transfer became the clear next bottleneck
