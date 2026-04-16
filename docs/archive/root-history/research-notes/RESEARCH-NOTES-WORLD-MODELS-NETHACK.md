# Research Notes: World Models for NetHack and This Repo

## Goal

Answer ten concrete questions about how world models could help improve the current teacher-aligned RL pipeline in this repository.

This note is grounded in:

- the current repo state
- our current failure mode: teacher-constrained RL can tie the teacher, but drifts instead of improving past it
- relevant world-model, text-game, and NetHack-adjacent literature

## Current Repo Context

Right now the strongest line in this repo is:

- offline teacher / behavior-regularized model on `v4` features
- teacher-constrained APPO online improvement
- deterministic held-out trace match as the trusted metric

The current bottleneck is:

- online RL drifts after correct initialization
- the policy does not reliably improve beyond the teacher

So the key question is not “can a world model solve NetHack outright?”

It is:

- can a world model improve representation, planning, or alignment enough to help this pipeline beat the teacher?

## Main Literature Reviewed

### NetHack / offline pretraining

- **Accelerating exploration and representation learning with offline pre-training**
  - shows that offline pretraining improves NetHack representation learning and sample efficiency
  - very relevant to this repo because we already have teacher traces and Dungeons and Data is available
  - local copy: [nethack_offline_pretrain_arxiv2023.pdf](/home/luc/rl-nethack/references/papers/nethack_offline_pretrain_arxiv2023.pdf)
  - arXiv: https://arxiv.org/abs/2304.00046

### Generic world-model RL

- **DreamerV3**
  - strong generic latent world-model RL baseline
  - important as a reference for what model-based RL can do well
  - local copy: [dreamerv3_arxiv2023.pdf](/home/luc/rl-nethack/references/papers/dreamerv3_arxiv2023.pdf)
  - arXiv: https://arxiv.org/abs/2301.04104

- **TD-MPC2**
  - model-predictive control with latent planning and strong sample efficiency
  - useful for “short-horizon planning inside a learned latent space”
  - arXiv: https://arxiv.org/abs/2310.16828

- **MuZero**
  - planning over a learned latent dynamics model rather than raw future observations
  - useful conceptually for “value-relevant abstract dynamics”
  - arXiv: https://arxiv.org/abs/1911.08265

### Text-game / language-world-model work

- **EMNLP 2022 text-game world-model paper**
  - world model for text-based games using pretrained language modeling components
  - focuses on predictive state structure and action consequences in text-like environments
  - local copy: [text_game_world_models_emnlp2022.pdf](/home/luc/rl-nethack/references/papers/text_game_world_models_emnlp2022.pdf)
  - PDF: https://aclanthology.org/2022.emnlp-main.86.pdf

### Long-horizon abstract planning over skills

- **Long-Horizon Planning with Predictable Skills**
  - advocates abstract world models over temporally extended skills
  - especially relevant for long-horizon, partially observable control
  - PDF: https://openreview.net/pdf/bf6fafd191fb8d7ac2b6d26a8420b15744d5b005.pdf

### Offline model-based scaling / generalization

- **JOWA**
  - jointly optimized world-action model for offline RL
  - useful for the idea that world-model pretraining and action modeling can share a backbone
  - arXiv: https://arxiv.org/abs/2410.00564

- **WHALE**
  - emphasizes generalizability and uncertainty for world models under distribution shift
  - useful because our policy drifts off the teacher distribution
  - arXiv: https://arxiv.org/abs/2411.05619

## Answers to the Ten Questions

### 1. Can a world model give us a better state representation than the current hand-built `v4` features?

Yes, very likely.

This is the most credible use of world models for this repo.

The NetHack offline pretraining paper is the strongest evidence here: even without a full model-based controller, offline predictive pretraining improves representation quality and sample efficiency on NetHack-like learning. That is directly aligned with our current problem, where `v4` features were a major improvement over earlier hand-built representations.

Conclusion:

- a learned latent state is more promising than more feature engineering alone
- the first world-model win for this repo is probably representation quality, not full model-based control

### 2. Should the model predict one-step transitions, or is NetHack better served by a k-step / skill-level outcome model?

NetHack is better served by a **k-step or skill-level outcome model**.

Reason:

- one-step prediction is too local
- our current failure mode is long-horizon objective drift, not inability to predict immediate movement
- the most relevant planning paper for this setting argues for **abstract world models over skills**, not raw one-step rollouts

Conclusion:

- do not center the next iteration on raw one-step next-state prediction
- prefer:
  - `k`-step returns
  - skill-conditioned outcomes
  - termination and cumulative reward prediction

### 3. Would a world model trained on Dungeons and Data let us initialize the policy with latent features that preserve teacher behavior better than plain BC alone?

Probably yes.

Dungeons and Data has:

- massive scale
- competent AutoAscend behavior
- long horizons
- broad early-to-mid game coverage

That makes it ideal for latent pretraining.

Instead of only initializing the policy head from BC, we could initialize:

- encoder / trunk from predictive pretraining
- policy head from BC / behavior regularization

This should help preserve teacher behavior while giving the policy richer state abstraction than hand-built features alone.

Conclusion:

- Dungeons and Data is a strong candidate for latent-state pretraining

### 4. Can we use the world model to predict teacher-aligned quantities like frontier gain, revisit risk, or expected trace-match improvement, instead of only raw next-state changes?

Yes, and this is likely more useful than raw reconstruction.

A good world model for this repo should not just predict:

- next observation

It should also predict auxiliary quantities that matter for the teacher-aligned objective:

- frontier gain
- revisit / loop risk
- local dead-end probability
- short-horizon teacher return proxy
- skill termination success

This follows the MuZero-style lesson:

- model only the parts of dynamics that matter for planning and value

Conclusion:

- the right predictive targets are value-relevant and teacher-relevant, not purely reconstructive

### 5. Should the world model be skill-conditioned (`explore`, `descend`, `search`, `combat`) so planning happens over options instead of primitive actions?

Yes.

This is one of the clearest conclusions from the literature.

NetHack is long-horizon, partially observable, and option-rich. The abstract-skill planning paper and the broader NetHack hierarchy work both point in the same direction:

- planning over options is more tractable than planning over every primitive action step-by-step

Conclusion:

- if we add a world model, it should probably be **skill-conditioned**
- the world model should predict outcomes of a skill executed for `K` steps, not only one-step primitive dynamics

### 6. Can we use the world model as an auxiliary loss during APPO to reduce drift?

Yes, this is a strong candidate.

This is one of the lowest-risk ways to integrate a world model into the current stack.

Instead of asking the world model to control directly, use it to regularize the policy representation:

- predict next latent state
- predict k-step summary
- predict revisit/frontier signals
- predict skill success/termination

This can help keep the actor’s internal representation grounded in environment dynamics, which may reduce degenerate reward hacking and drift.

Conclusion:

- auxiliary world-model losses during APPO are a better first integration path than replacing APPO

### 7. Would short-horizon imagination help rank candidate actions online, replacing some of the current weak scalar proxy with predicted multi-step return?

Yes, possibly a lot.

This is especially promising for the current `explore` setting.

We already know:

- one-step shaping is not sufficient
- the current scalar proxy is too weak

A short-horizon imagination module could score candidate actions by predicted:

- unique tile gain over the next `k` steps
- loop risk
- chance of finding stairs/frontier
- skill-specific return

This is closer to model-predictive control than to full Dreamer-style latent rollouts.

Conclusion:

- short-horizon imagination is promising
- full free-running long imagination is probably too brittle as a first step

### 8. Can we train a latent value model on top of world-model rollouts so RL improves beyond the teacher without immediately leaving the teacher manifold?

Potentially yes, but only if the model stays close to teacher-supported regions.

This is where recent world-model papers emphasizing distribution shift and uncertainty matter. Model-based optimization tends to fail when imagination drifts too far from the data distribution.

For this repo, that means:

- keep imagined rollouts short
- keep them near teacher / Dungeons-and-Data support
- use uncertainty or conservative filtering

Conclusion:

- a latent value model could help
- but it should be conservative and likely stay short-horizon initially

### 9. How much partial observability can the world model actually absorb without recurrence or explicit memory?

Not enough.

NetHack is strongly partially observable. Several relevant papers and our own repo experience both point to the same conclusion:

- visible state alone is not enough
- memory matters

A world model here should have access to:

- action history
- short latent history
- memory summaries
- possibly explicit map-memory state

Conclusion:

- a useful NetHack world model should be memory-augmented
- a memoryless one-step predictor is unlikely to be enough

### 10. What is the cheapest high-signal experiment?

Best cheap experiment:

1. train a latent predictive encoder on teacher / Dungeons-and-Data traces
2. freeze or partially freeze it
3. plug it into BC / behavior-reg
4. measure held-out trace match on the current benchmark

Second-best:

- add a world-model auxiliary loss to APPO on top of the same encoder

The least efficient first experiment would be:

- full Dreamer-like replacement of the current control stack

Conclusion:

- first use world models for representation and auxiliary prediction
- then use short-horizon imagined ranking or skill planning

## Overall Conclusion

World models can help this repo, but the most promising path is:

1. **representation learning from long teacher traces**
2. **skill-conditioned short-horizon predictive modeling**
3. **auxiliary losses for BC / APPO**
4. **short-horizon imagined ranking of actions or options**

The least promising next step is:

- trying to replace the current teacher-constrained RL loop with a full end-to-end model-based controller immediately

## Concrete Recommendation for This Repo

### Best first world-model project

Build a **skill-conditioned latent world model** that predicts:

- next latent state after `K` steps
- cumulative skill reward / frontier gain
- revisit / loop risk
- skill termination flag

Train it on:

- current teacher traces
- eventually Dungeons and Data / AutoAscend trajectories

Use it first for:

- encoder pretraining
- BC / behavior-reg feature extraction
- auxiliary loss during APPO

### Best second world-model project

Use the same model for **short-horizon action or option ranking**:

- candidate action -> imagine `K` steps -> score predicted teacher-relevant outcome

### What not to do first

- full Dreamer-style control replacement
- long free-running imagination over weakly grounded latents
- world-model-only RL without teacher constraints

## Repo-Shaping Implications

The repo should evolve toward:

- **teacher data / Dungeons and Data -> latent pretraining**
- **BC / behavior-reg -> teacher-constrained RL**
- **world model auxiliary losses**
- **skill-level short-horizon planning**

This is much more likely to improve the trusted trace metric than blindly scaling APPO or replacing the whole stack with generic model-based RL.

## Sources

- [nethack_offline_pretrain_arxiv2023.pdf](/home/luc/rl-nethack/references/papers/nethack_offline_pretrain_arxiv2023.pdf)
- [dreamerv3_arxiv2023.pdf](/home/luc/rl-nethack/references/papers/dreamerv3_arxiv2023.pdf)
- [text_game_world_models_emnlp2022.pdf](/home/luc/rl-nethack/references/papers/text_game_world_models_emnlp2022.pdf)
- [Long-Horizon Planning with Predictable Skills](https://openreview.net/pdf/bf6fafd191fb8d7ac2b6d26a8420b15744d5b005.pdf)
- [TD-MPC2](https://arxiv.org/abs/2310.16828)
- [MuZero](https://arxiv.org/abs/1911.08265)
- [JOWA](https://arxiv.org/abs/2410.00564)
- [WHALE](https://arxiv.org/abs/2411.05619)
