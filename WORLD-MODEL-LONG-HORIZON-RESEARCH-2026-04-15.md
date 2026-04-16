# Long-Horizon World Model Research Notes

Date: 2026-04-15

## Goal

Build a true non-symbolic world model for NetHack from the current predictor codebase:

- long-horizon memory over many action sequences
- calibrated latent dynamics that can be rolled forward
- usable for multi-step search and planning
- no symbolic planner, no hand-coded strategy layer

This note is based on:

- current repo code in [rl/world_model.py](/home/luc/rl-nethack/rl/world_model.py), [rl/train_world_model.py](/home/luc/rl-nethack/rl/train_world_model.py), [rl/world_model_eval.py](/home/luc/rl-nethack/rl/world_model_eval.py), [rl/world_model_dataset.py](/home/luc/rl-nethack/rl/world_model_dataset.py), [rl/world_model_features.py](/home/luc/rl-nethack/rl/world_model_features.py), [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)
- local reports [REPORT-WORLD-MODEL-VALIDATION-2026-04-06.md](/home/luc/rl-nethack/REPORT-WORLD-MODEL-VALIDATION-2026-04-06.md) and [REPORT-LLM-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack/REPORT-LLM-WORLD-MODEL-2026-04-06.md)
- primary-source papers and project pages listed in the Sources section

## Bottom Line

The current repo does **not** have a planning-grade world model.

It has:

- a short-horizon action-conditioned predictor over feature vectors
- optional frozen text conditioning
- evaluation mostly around direct prediction metrics and downstream BC usefulness

It does **not** have:

- a persistent latent belief state over long partial-observation histories
- autoregressive rollout training over long action sequences
- probabilistic uncertainty suitable for risk-aware search
- a planner that rolls the model forward and scores long trajectories
- calibration machinery strong enough to trust long open-loop rollouts

The right next step is **not** to incrementally stretch the current one-step predictor.  
The right next step is to replace it with a **sequence world model**:

- stochastic + deterministic latent state
- long-memory backbone
- multi-step latent rollout training
- uncertainty estimation and calibration
- planner over latent rollouts

## What The Current Code Actually Is

### Current model shape

[rl/world_model.py](/home/luc/rl-nethack/rl/world_model.py) defines `TraceWorldModel`:

- feedforward encoder from current feature vector to latent
- action embedding and task embedding
- transition MLP from `[latent, action, task]` to hidden
- heads for:
  - current feature reconstruction
  - action logits
  - future feature vector
  - cumulative reward over a fixed horizon
  - done-within-horizon

This is a useful representation learner, but it is still a **single-step latent transition regressor**.

### Current training target

[rl/world_model_dataset.py](/home/luc/rl-nethack/rl/world_model_dataset.py) builds examples like:

- input: row at time `t`
- action: action at `t`
- target feature vector: row at `t + horizon`
- reward target: cumulative reward across the window
- done target: whether termination happened within the window

Missing pieces for a simulator:

- no intermediate supervision for `t+1, t+2, ...`
- no learned posterior/prior latent filtering over a sequence
- no belief update from partial observations
- no training objective for open-loop rollout consistency
- no notion of uncertainty over future state

### Current eval target

[rl/world_model_eval.py](/home/luc/rl-nethack/rl/world_model_eval.py) measures:

- feature MSE/MAE/cosine
- action accuracy
- reward error
- latent spread
- downstream BC usefulness after appending latent features

This is good instrumentation for a predictor. It is not enough for planning-grade validation.

## Diagnosis

The repo’s current world-model branch is best understood as:

- a **teacher builder**
- a **feature augmenter**
- a **short-horizon predictive prior**

It is not yet:

- a long-horizon generative simulator
- a calibrated latent dynamics model for search
- a belief-state model for partial observability

That matches the local reports:

- [REPORT-WORLD-MODEL-VALIDATION-2026-04-06.md](/home/luc/rl-nethack/REPORT-WORLD-MODEL-VALIDATION-2026-04-06.md): direct predictive quality improved, but downstream online RL did not
- [REPORT-LLM-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack/REPORT-LLM-WORLD-MODEL-2026-04-06.md): text conditioning improved the teacher, but did not solve online drift or planning

## Literature Threads That Matter

### 1. RSSM-style latent state-space models are the correct baseline

The PlaNet and Dreamer line establishes the core recipe:

- latent state should combine:
  - deterministic memory
  - stochastic latent state
- the model should learn:
  - posterior latent from observation history
  - prior latent from previous latent + action
  - reward and continuation heads
- training should directly support **open-loop imagination**

Key sources:

- PlaNet: https://proceedings.mlr.press/v97/hafner19a.html
- DreamerV3 PDF: https://arxiv.org/pdf/2301.04104.pdf

Important takeaways:

- DreamerV3 explicitly uses an RSSM, with encoder -> stochastic `z_t`, recurrent memory `h_t`, and reward / continuation / reconstruction heads.
- DreamerV3 shows open-loop prediction far beyond the immediate next step.
- PlaNet introduced the core idea that planning should happen in latent space, not in raw observation space.

Why this matters for this repo:

- your current `TraceWorldModel` has no recurrent memory state
- it cannot maintain a latent belief state over partial observations
- it cannot naturally support long imagined trajectories

### 2. Long-horizon memory is a backbone problem, not just a loss problem

World models fail on long horizons partly because the backbone forgets.

Key sources:

- R2I / Recall to Imagine: https://openreview.net/forum?id=1vDArHJ68h
- TransDreamer: https://openreview.net/forum?id=s3K0arSRl4d
- Structured SSMs in RL: https://openreview.net/forum?id=4W9FVg1j6I
- Emerging 2026 hierarchical planning note: https://huggingface.co/papers/2604.03208

Important takeaways:

- R2I argues current MBRL agents struggle with long-term dependencies and credit assignment.
- R2I replaces the recurrent backbone with a structured state-space model and reports strong gains on memory-heavy tasks.
- TransDreamer shows transformer world models can help on tasks requiring long-range memory access.
- Structured state space models offer a compelling middle ground:
  - better long-range memory than GRU/LSTM
  - cheaper inference than long-context transformers
  - recurrent inference mode is good for rollouts

Why this matters for this repo:

- NetHack is partially observed
- planning needs memory of old observations, not just the current feature vector
- if the model forgets stairs, hazards, food context, inventory changes, or loop structure, planning will fail even with a perfect one-step loss

My conclusion:

- the world model backbone should be either:
  - RSSM + SSM backbone
  - RSSM + transformer memory if compute allows
- for this repo, **RSSM + SSM** is the best first serious target

### 3. Planning can happen in latent space without decoding everything

MuZero, EfficientZero, and TD-MPC2 show that planning-grade world models do not need to reconstruct full observations at every step.

Key sources:

- MuZero (Nature): https://www.nature.com/articles/s41586-020-03051-4
- EfficientZero: https://papers.nips.cc/paper/2021/hash/d5eca8dc3820cad9fe56a3bafda65ca1-Abstract.html
- TD-MPC2 OpenReview: https://openreview.net/forum?id=Oxh5CstDJU
- TD-MPC2 project page: https://www.tdmpc2.com/

Important takeaways:

- MuZero learns a latent model that predicts quantities relevant to planning rather than reconstructing the full environment.
- EfficientZero adds self-supervised consistency to make latent dynamics more sample efficient.
- TD-MPC2 uses decoder-free latent dynamics with planning directly in latent space.

Why this matters for this repo:

- NetHack observations are huge and messy
- a planning-grade model does not need perfect full-state text generation first
- it needs a latent state that preserves:
  - controllable state changes
  - reward-relevant consequences
  - termination risk
  - validity of future actions
  - enough memory to support long rollout branching

My conclusion:

- do **not** define success as “generates perfect next prompt text”
- define success as “supports accurate latent rollouts for search and return estimation”

### 4. Long-horizon planning fails if uncertainty is not modeled and calibrated

The planning literature is very clear:

- long rollouts amplify model error
- uncertainty estimates from neural nets are often miscalibrated
- if uncertainty is bad, planning exploits model errors

Key sources:

- PETS / probabilistic ensembles: https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html
- MBPO: https://papers.nips.cc/paper/9416-when-to-trust-your-model-model-based-policy-optimization
- Calibrated MBRL PDF: https://proceedings.mlr.press/v97/malik19a/malik19a.pdf
- Emerging 2026 uncertainty paper: https://openreview.net/forum?id=pZuZWRuPyi

Important takeaways:

- PETS separates aleatoric and epistemic uncertainty through probabilistic ensembles and trajectory sampling.
- MBPO’s main lesson is that long model rollouts are dangerous; short branched rollouts are often safer.
- Calibrated MBRL argues that uncertainty must be calibrated, not just present.
- Horizon-calibrated uncertainty is an increasingly explicit theme in newer work.

Why this matters for this repo:

- NetHack is stochastic and partially observed
- rollout error compounds badly over long horizons
- a planner without calibrated uncertainty will overtrust imagined futures

My conclusion:

- the first planning-grade model should be an **ensemble latent world model**
- the planner should be **risk-aware**
- the evaluation must include horizon-wise calibration, not just MSE

### 5. Text-game world-model work supports using text as context, but warns against naive LM fine-tuning

Key sources:

- PLM-based World Models for Text-based Games: https://aclanthology.org/2022.emnlp-main.86/
- Paper PDF: https://aclanthology.org/2022.emnlp-main.86.pdf
- On the Effects of Fine-tuning Language Models for Text-Based Reinforcement Learning: https://aclanthology.org/2025.coling-main.445/
- IBM summary page: https://research.ibm.com/publications/on-the-effects-of-fine-tuning-language-models-for-text-based-reinforcement-learning

Important takeaways:

- pretrained language models help for world modeling in text environments
- useful tasks include:
  - future valid action prediction
  - graph/state change prediction
- naive fine-tuning of LMs inside RL can cause semantic degeneration

Why this matters for this repo:

- your current text-conditioned world-model result already points the same way
- text should remain a conditioning signal or auxiliary channel
- the core dynamics model should not become “an LLM that autoregresses prompt text”

My conclusion:

- keep text conditioning
- use text for latent belief formation and auxiliary prediction
- do not make raw text generation the primary simulator objective

### 6. NetHack specifically needs scale, long trajectories, and partial-observation training

Key sources:

- Dungeons and Data / NLD PDF: https://proceedings.neurips.cc/paper_files/paper/2022/file/9d9258fd703057246cb341e615426e2d-Paper-Datasets_and_Benchmarks.pdf
- NetHack Challenge supplement / AutoAscend appendix: https://proceedings.mlr.press/v176/hambro22a/hambro22a-supp.pdf
- Challenge report: https://proceedings.mlr.press/v176/hambro22a.html

Important takeaways:

- NLD contains:
  - 10B state transitions from human play
  - 3B state-action-score transitions from AutoAscend
- the paper explicitly says NLE is:
  - partially observed
  - stochastic
  - procedurally generated
  - long-horizon
- the same paper also says symbolic bots beat deep RL by about 5x in the challenge

Why this matters for this repo:

- you need longer sequential training data than current short trace slices
- you should eventually train on longer unrolls from large trajectory corpora
- the model should be judged on generalization to new seeds, not replay-only fit

## What The New World Model Should Look Like

## Design target

The target should be a **calibrated sequence world model with latent planning**, not a bigger version of `TraceWorldModel`.

### Core latent state

At each step:

- observation encoder:
  - current numeric features
  - local map patch
  - inventory / message text features
- posterior latent `z_t`
- deterministic memory state `h_t`
- combined belief state `b_t = [h_t, z_t]`

This is the planning state.

### Backbone

Best first serious candidate:

- **RSSM with SSM backbone**

Concretely:

- prior transition:
  - `h_t = f_ssm(h_{t-1}, z_{t-1}, a_{t-1})`
  - `p(z_t | h_t)`
- posterior correction:
  - `q(z_t | h_t, o_t, text_t)`

Why:

- keeps Dreamer/PlaNet-style latent filtering
- adds longer memory capacity than a simple GRU
- stays cheaper than long-context transformer rollouts
- naturally supports recurrent inference for planning

### Prediction heads

The model should predict:

- immediate reward
- continuation / done
- next-step valid action mask
- compact next-observation features
- optional text-conditioned auxiliary targets:
  - message changes
  - inventory deltas
  - tile-discovery deltas
  - visible monster/item summaries

The model does **not** need to decode raw NetHack state exhaustively on day 1.

### Training objective

Must move from single target at `t+h` to **sequence training**.

Recommended losses:

- posterior/prior KL losses
- immediate reward prediction
- continuation prediction
- next-step feature prediction
- action-validity prediction
- multi-step latent consistency
- open-loop rollout consistency at horizons:
  - 1, 2, 4, 8, 16, 32, 64
- optional contrastive latent loss for state identity / loop closure
- uncertainty loss:
  - probabilistic ensemble NLL
  - or distributional heads

Important:

- train with teacher forcing on observed sequences
- also evaluate with **open-loop rollout**, where only the initial context is given and the model is rolled forward using the action sequence

### Uncertainty

This is mandatory for planning.

Recommended first version:

- ensemble of `K=5` world models
- each predicts:
  - reward distribution
  - done probability
  - latent next-state distribution
- planner uses:
  - mean reward/value
  - variance penalty
  - trajectory rejection when uncertainty exceeds threshold

Calibration:

- maintain train / val / calibration splits
- fit post-hoc calibration for:
  - done probability
  - reward intervals
  - action-validity probabilities
- track reliability curves and expected calibration error by rollout horizon

### Planning layer

No symbolic controller.

Use latent-space planning only:

- cross-entropy method (CEM) over action sequences
- beam search over discrete actions
- optionally MCTS once the latent model is strong enough

Planner input:

- belief state from recent observation history

Planner objective:

- sum of predicted reward
- terminal risk penalty
- epistemic uncertainty penalty
- optional value bootstrap from a learned latent value head

The first planner should search over short-to-medium horizons with receding horizon control:

- e.g. plan 16 to 64 steps
- execute 1 to 4 steps
- reobserve and replan

This is more realistic than expecting one open-loop 500-step plan to work.

### Long-horizon memory without symbolic abstraction

Because symbolic methods are off the table, long-horizon abstraction should be **learned**, not hand-coded.

The most plausible path:

- multi-scale latent dynamics
- chunked action modeling
- learned temporal abstraction

Concretely:

- fine-scale model:
  - predicts step-level transitions
- coarse-scale model:
  - predicts over action chunks or latent summaries every `k` steps
- planner can search coarsely first, then refine

This matches the direction of recent hierarchical planning with latent world models, but without introducing symbolic skills.

## What To Build In This Repo

### Phase 1: replace predictor with sequence belief model

Add a new model alongside the old one:

- `SequenceWorldModel`

Suggested files:

- `rl/sequence_world_model.py`
- `rl/sequence_world_model_dataset.py`
- `rl/train_sequence_world_model.py`
- `rl/sequence_world_model_eval.py`
- `rl/world_model_planner.py`
- `rl/world_model_calibration.py`

Keep the old `TraceWorldModel` only as a baseline.

### Phase 2: upgrade data pipeline from point examples to sequence windows

Current dataset builder emits one `(x_t, a_t, x_{t+h})` style target.

Replace with windows that emit:

- `o_{t-L+1:t}` context
- `a_{t:t+H-1}` future actions
- `r_{t:t+H-1}`
- `done_{t:t+H-1}`
- optional text prompts / messages over the sequence

Use:

- variable context length
- variable rollout horizon
- role / seed / episode metadata

### Phase 3: improve observation representation for latent state tracking

The current `v4` feature vector is useful but too compressed for planning-grade world modeling.

Keep it, but add:

- larger local spatial patch around agent
- map-memory token features
- inventory summary embedding
- message text embedding
- recent action / recent position history

Do not depend only on a single 302-dim vector if the goal is a simulator.

### Phase 4: planning-grade evaluation

New eval suite should include:

- one-step prediction accuracy
- open-loop rollout error by horizon
- latent belief recall tests
- uncertainty calibration by horizon
- planner utility:
  - does search outperform policy-only baseline on held-out seeds?
- branch ranking metrics:
  - does the model correctly rank candidate action sequences?
- loop-closure tests:
  - after long paths, does the latent state still remember previously seen structures?

For NetHack, I would add explicit held-out tests for:

- finding downstairs after delayed cue
- avoiding starvation after long detour
- preserving map-memory for backtracking
- handling long action sequences with no immediate reward

### Phase 5: only then connect to RL improvement

Do not use the planner to train policy improvement until the world model passes rollout and calibration gates.

Otherwise:

- the policy will exploit model errors
- search quality will look good offline and fail online

## Recommended Architecture Choice

If we had to choose one concrete plan now:

### Recommended first serious architecture

- observation encoder:
  - numeric `v4` features
  - learned local patch encoder
  - frozen text encoder for message / prompt / inventory text
- latent state:
  - posterior stochastic `z_t`
  - deterministic memory `h_t`
- sequence backbone:
  - SSM / S4-style memory backbone
- dynamics:
  - prior `p(z_t | h_t)`
  - reward / continue / validity / feature heads
- uncertainty:
  - 5-model ensemble
- planning:
  - CEM or beam search in latent space
- training:
  - sequence ELBO-style objective
  - multi-step open-loop rollout losses
  - horizon-wise calibration reporting

This is the strongest fit to:

- your current codebase
- NetHack’s partial observability
- the need for long memory
- the desire for non-symbolic planning

## Things We Should Not Do

### 1. Do not just increase `horizon` in the current dataset builder

That still leaves:

- no sequential latent state
- no posterior/prior filtering
- no intermediate rollout supervision

### 2. Do not make raw text generation the main simulator objective

Raw text is too lossy and too unconstrained as the primary planning target.

Use text as:

- context
- auxiliary supervision
- semantic regularizer

### 3. Do not trust a single deterministic model for long-horizon search

Need:

- ensemble uncertainty
- calibration curves
- risk-aware planning

### 4. Do not use downstream BC score as the main measure of world-model quality

That metric is useful, but it can hide:

- rollout drift
- forgetting
- uncertainty failure
- planner exploitation

### 5. Do not fine-tune a full LM in the online RL loop as the default plan

The text-RL literature warns this can degrade semantics.

## Immediate Repo Plan

## Step 1

Freeze the current `TraceWorldModel` as baseline.

## Step 2

Implement a sequence dataset:

- context length `L`
- rollout horizon `H`
- outputs per time step

## Step 3

Implement `SequenceWorldModel` with:

- posterior encoder
- prior transition
- recurrent / SSM memory
- reward / done / validity / feature heads

## Step 4

Implement open-loop rollout eval:

- roll forward for full action sequence
- compare predicted vs actual at horizons 1/2/4/8/16/32/64

## Step 5

Implement uncertainty:

- ensemble checkpoints
- calibration set
- horizon reliability plots

## Step 6

Implement planner:

- beam search or CEM over discrete action sequences
- receding horizon execution

## Step 7

Only after passing rollout gates, compare:

- policy-only BC / APPO
- planner-only with world model
- planner-guided policy

## My Current Recommendation

If the objective is:

> a calibrated generative simulator we can reliably roll forward and use for long-horizon planning

then the repo should pivot from:

- feature augmentation for BC

to:

- sequence latent belief modeling
- planning-focused rollouts
- uncertainty calibration

The most defensible first build is:

**Dreamer/PlaNet-style latent state-space model, upgraded with an SSM memory backbone, ensemble uncertainty, and receding-horizon latent planning.**

That is the cleanest bridge from what exists now to a real world model.

## Sources

### Local repo

- [rl/world_model.py](/home/luc/rl-nethack/rl/world_model.py)
- [rl/train_world_model.py](/home/luc/rl-nethack/rl/train_world_model.py)
- [rl/world_model_eval.py](/home/luc/rl-nethack/rl/world_model_eval.py)
- [rl/world_model_dataset.py](/home/luc/rl-nethack/rl/world_model_dataset.py)
- [rl/world_model_features.py](/home/luc/rl-nethack/rl/world_model_features.py)
- [rl/feature_encoder.py](/home/luc/rl-nethack/rl/feature_encoder.py)
- [REPORT-WORLD-MODEL-VALIDATION-2026-04-06.md](/home/luc/rl-nethack/REPORT-WORLD-MODEL-VALIDATION-2026-04-06.md)
- [REPORT-LLM-WORLD-MODEL-2026-04-06.md](/home/luc/rl-nethack/REPORT-LLM-WORLD-MODEL-2026-04-06.md)

### Primary-source external references

- PlaNet / Learning Latent Dynamics for Planning from Pixels: https://proceedings.mlr.press/v97/hafner19a.html
- PlaNet project page: https://planetrl.github.io/
- DreamerV3 PDF: https://arxiv.org/pdf/2301.04104.pdf
- MuZero: https://www.nature.com/articles/s41586-020-03051-4
- EfficientZero: https://papers.nips.cc/paper/2021/hash/d5eca8dc3820cad9fe56a3bafda65ca1-Abstract.html
- TD-MPC2 OpenReview: https://openreview.net/forum?id=Oxh5CstDJU
- TD-MPC2 project page: https://www.tdmpc2.com/
- R2I / Mastering Memory Tasks with World Models: https://openreview.net/forum?id=1vDArHJ68h
- TransDreamer: https://openreview.net/forum?id=s3K0arSRl4d
- Structured State Space Models for In-Context RL: https://openreview.net/forum?id=4W9FVg1j6I
- Hierarchical Planning with Latent World Models: https://huggingface.co/papers/2604.03208
- PETS: https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html
- MBPO: https://papers.nips.cc/paper/9416-when-to-trust-your-model-model-based-policy-optimization
- Calibrated Model-Based Deep RL PDF: https://proceedings.mlr.press/v97/malik19a/malik19a.pdf
- Horizon-Calibrated Uncertainty World Model: https://openreview.net/forum?id=pZuZWRuPyi
- PLM-based World Models for Text-based Games: https://aclanthology.org/2022.emnlp-main.86/
- PLM-based World Models PDF: https://aclanthology.org/2022.emnlp-main.86.pdf
- On the Effects of Fine-tuning Language Models for Text-Based RL: https://aclanthology.org/2025.coling-main.445/
- IBM page for the same paper: https://research.ibm.com/publications/on-the-effects-of-fine-tuning-language-models-for-text-based-reinforcement-learning
- Dungeons and Data / NLD PDF: https://proceedings.neurips.cc/paper_files/paper/2022/file/9d9258fd703057246cb341e615426e2d-Paper-Datasets_and_Benchmarks.pdf
- NetHack Challenge report: https://proceedings.mlr.press/v176/hambro22a.html
- NetHack Challenge supplement / AutoAscend appendix: https://proceedings.mlr.press/v176/hambro22a/hambro22a-supp.pdf
