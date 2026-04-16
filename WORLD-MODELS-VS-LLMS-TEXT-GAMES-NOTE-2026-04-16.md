# World Models vs LLMs for Text-Based Games

Date: 2026-04-16

## Short answer

A world model is usually built to model:

- `state or belief + action -> next state or belief + reward + termination`

An LLM is usually built to model:

- `text prefix -> next token`

So the main difference is usually:

- objective
- interface
- outputs

not always the raw backbone.

A transformer can be used as either one, but the surrounding training target and prediction heads make the role different.

## Core architectural difference

### LLM

A standard LLM usually has:

- token embeddings
- transformer blocks with causal self-attention
- a next-token prediction head

Primary target:

- maximize next-token likelihood over text

Typical output:

- a continuation of text

### World model

A standard world model usually has:

- an observation encoder
- an action-conditioned transition model
- a latent or belief state
- heads for reward, termination, value, policy, or next-state features
- sometimes a planner on top

Primary target:

- predict what happens after an action

Typical output:

- next latent state
- next observation or features
- reward
- done
- sometimes policy or value

## Why RL world models often look different

Planning needs properties that raw LLMs usually do not expose directly:

- explicit action-conditioned transitions
- compact latent state
- cheap repeated rollout over many branches
- reward prediction
- done prediction
- uncertainty or disagreement estimates
- search compatibility

A raw autoregressive LLM can sometimes simulate a world, but it is usually expensive and awkward for branchy planning.

A latent world model is usually designed to make imagined rollout cheap.

## What the literature says

### DreamerV3

DreamerV3 is a canonical modern world-model RL system. It learns a world model and improves behavior by imagining futures in latent space rather than directly modeling raw token sequences.

That is a very different use pattern from a standard LLM.

Source:

- <https://arxiv.org/abs/2301.04104>

### MuZero

MuZero explicitly learns a model oriented around quantities relevant to planning rather than raw observation reconstruction. The important outputs are reward, value, and policy under recurrent imagined rollout.

That is again world-model behavior, not plain next-token modeling.

Source:

- <https://www.nature.com/articles/s41586-020-03051-4>

### PLM-based World Models for Text-based Games

This paper is important because it blurs the line.

The authors use pretrained language models as the basis for an actionable world model in text games. They add world-model-like structure around the PLM, including constrained decoding and predictions like:

- future valid actions
- graph changes

So this is not “just an LLM,” but also not a classic small RSSM. It is a language-model backbone adapted into a world-model role.

Source:

- <https://aclanthology.org/2022.emnlp-main.86/>

### Can Language Models Serve as Text-Based World Simulators?

This paper directly asks whether strong language models can serve as world simulators in text settings. The key lesson is that strong LLMs can do surprisingly well, but they are still unreliable as faithful simulators without further work.

That matters because “good at text” is not the same as “reliable transition model.”

Source:

- <https://aclanthology.org/2024.acl-short.1/>

## Clean comparison

### If the model behaves mainly like an LLM

It is doing:

- text continuation
- token-level prediction
- implicit state tracking in hidden activations

Its strengths are usually:

- broad prior knowledge
- strong language understanding
- long-range sequence modeling
- better initial sample efficiency if pretrained well

Its weaknesses for planning are usually:

- expensive branching rollout
- no explicit transition interface by default
- no explicit reward or done heads by default
- no compact planner state unless added manually

### If the model behaves mainly like a world model

It is doing:

- belief or latent-state update
- action-conditioned transition prediction
- explicit reward, done, and often value prediction
- imagined rollout for control or search

Its strengths are usually:

- planner-friendly latent state
- cheap counterfactual rollout
- explicit control-relevant predictions
- easier integration with model-based RL

Its weaknesses are usually:

- often weaker prior if trained from scratch
- more data-hungry
- weaker open-world text understanding unless backed by a pretrained language model

## What this means for NetHack

For NetHack, a raw small scratch-trained world model has clear advantages:

- cheap rollout
- explicit planning signals
- controllable architecture

But it also has obvious disadvantages:

- weak prior
- poor data efficiency
- limited long-context understanding

A pretrained long-context LLM has the opposite tradeoff:

- much stronger prior
- much better long-range sequence modeling
- likely better abstraction over text and state history

But it is not ideal as a direct planner core:

- expensive for tree search
- not naturally action-conditioned in the RL sense
- not naturally equipped with reward/value/done heads unless we add them

## Best hybrid view

For this project, the best architecture is probably not:

- “LLM instead of world model”

It is more likely:

- `LLM as sequence encoder / belief model`
- plus `world-model-style auxiliary heads`
- and later possibly `distillation into a smaller planner-friendly latent model`

That gives:

- strong representation learning from the pretrained backbone
- long-context memory
- structured supervision from reward/value/done/legal actions
- a path to cheap planning later

## Practical design for this repo

The likely best long-run design is:

1. Train a long-context Qwen model on full NetHack trajectories.
2. Feed both:
   - serialized game history
   - structured side features such as `feature_vector` and legal actions
3. Train multi-task heads for:
   - next action
   - reward
   - done
   - value
   - legal actions
   - optional planner-trace policy targets
4. Evaluate on robust held-out long-horizon slices.
5. Distill into a smaller latent planner model if we need cheap online search.

## Bottom line

The architectural difference is real, but it is mostly about what the model is trained to do.

- An LLM is usually a next-token model.
- A world model is usually an action-conditioned transition model.

In text-based games, the line can blur when a pretrained language model is wrapped with world-model heads and constraints.

For NetHack, that hybrid approach is likely the strongest direction:

- pretrained long-context LLM for representation and memory
- world-model-style supervision for control relevance
- smaller latent planner later if needed

## Sources

- DreamerV3: <https://arxiv.org/abs/2301.04104>
- MuZero: <https://www.nature.com/articles/s41586-020-03051-4>
- PLM-based World Models for Text-based Games: <https://aclanthology.org/2022.emnlp-main.86/>
- Can Language Models Serve as Text-Based World Simulators?: <https://aclanthology.org/2024.acl-short.1/>
