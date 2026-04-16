# Input Context Notes (2026-04-16)

## What The Repo Currently Feeds Models

There are several different input styles in this repo. They are not all used by the same training path.

### 1. Compact current-state text

Used by the simple forward-model SFT path.

State contains:

- HP / max HP
- AC, Str, Dex
- position
- gold, depth, turn
- 4 adjacent tiles
- visible monsters
- visible items

Then the prompt appends:

- `Action: <action>`

Relevant code:

- [src/state_encoder.py](/home/luc/rl-nethack-worktree-20260416/src/state_encoder.py:264)
- [src/state_encoder.py](/home/luc/rl-nethack-worktree-20260416/src/state_encoder.py:292)

Approximate size per turn:

- about `50-100` tokens


### 2. Enriched memory + viewport prompt

Used in Andrew's `scripts/generate_training_data.py` path.

Prompt contains:

- memory summary
- current viewport as ASCII
- stats
- last message
- proposed action

Relevant code:

- [src/memory_tracker.py](/home/luc/rl-nethack-worktree-20260416/src/memory_tracker.py:343)
- [scripts/generate_training_data.py](/home/luc/rl-nethack-worktree-20260416/scripts/generate_training_data.py:396)

Measured from local `data/training_pairs_5k.jsonl` sample:

- first 500 prompts:
- mean `523.9` chars
- median `499` chars
- p95 `835` chars
- mean `48.2` words

Approximate size per turn:

- about `90-220` tokens typical
- around `130` tokens average is a reasonable estimate


### 3. RL / BC / world-model numeric state

The main RL stack often does not use text as the primary state at all.

It uses a numeric `feature_vector`.

Observation dimensions:

- `v1 = 106`
- `v2 = 160`
- `v3 = 244`
- `v4 = 302`

Relevant code:

- [rl/feature_encoder.py](/home/luc/rl-nethack-worktree-20260416/rl/feature_encoder.py:51)

This means token count is effectively:

- `0` text tokens for the main numeric state input


## How Much Past The System Sees

### Main current BC / world-model / RL stack

Mostly current state only.

- BC usually sees the current `feature_vector`
- prompt-conditioned BC/world-model can also see compact `state_prompt`
- this is not a real multi-turn raw history window


### Simple SFT data path

Mostly current state only.

- one example is `current state + proposed action -> next-state delta`


### Counterfactual data path

Also mostly current state only.

- one example is `current state + candidate action -> delta`


### Andrew's memory-augmented policy/data path

This is the one path that sees meaningful past context.

It includes:

- last 4 explicit `(state, action)` turns in chat history
- summarized long-term memory via `MemoryTracker`

The summary includes:

- explored area
- discovered rooms
- remembered floor items
- recently seen monsters

Relevant code:

- [scripts/generate_training_data.py](/home/luc/rl-nethack-worktree-20260416/scripts/generate_training_data.py:202)
- [src/memory_tracker.py](/home/luc/rl-nethack-worktree-20260416/src/memory_tracker.py:293)


## What "Delta" Means

`Delta` means the change caused by one action, not the whole next state.

It includes:

- position change
- HP / gold / depth / turn changes
- newly revealed tiles
- resulting game message
- survival flag

Relevant code:

- [src/state_encoder.py](/home/luc/rl-nethack-worktree-20260416/src/state_encoder.py:206)


## Current Training Data Usage

### Are we training on `counterfactual_training_pairs.jsonl` now?

Not in the main documented current path.

The current documented SFT run uses:

- `cli.py generate` to build `data/pipeline_train.jsonl`
- then `train.py` on that dataset

Relevant docs:

- [RL-APPO-HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/RL-APPO-HANDOFF.md:611)
- [RL-APPO-HANDOFF.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/handoffs/RL-APPO-HANDOFF.md:634)

`counterfactual_training_pairs.jsonl` still exists as a generator path, but it is not the main current training dataset in the documented runs.


## Teacher / BC / World-Model Terms

### Proposed action

The action being conditioned on in the prompt.

Example:

- "here is the current state"
- "assume the action is `east`"
- predict the resulting delta


### BC model

BC means behavior cloning.

It is a policy model trained to imitate actions from trace rows.

- input: state features (and optionally prompt text)
- target: action label

Relevant code:

- [rl/train_bc.py](/home/luc/rl-nethack-worktree-20260416/rl/train_bc.py:133)
- [rl/bc_model.py](/home/luc/rl-nethack-worktree-20260416/rl/bc_model.py:44)


### World-model transformed traces

Each trace row's original `feature_vector` is run through the world-model encoder.

Then the trace row is rewritten in one of these modes:

- `replace`: latent only
- `concat`: original features + latent
- `concat_aux`: original features + latent + world-model action logits

Relevant code:

- [rl/world_model_features.py](/home/luc/rl-nethack-worktree-20260416/rl/world_model_features.py:80)


### Teacher

Usually a stronger BC checkpoint trusted more than the student being trained.


### Distill from teacher

Run the teacher on each row and train the student to match the teacher's action distribution, not just the hard action label.

Relevant code:

- [rl/train_bc.py](/home/luc/rl-nethack-worktree-20260416/rl/train_bc.py:193)
- [rl/train_bc.py](/home/luc/rl-nethack-worktree-20260416/rl/train_bc.py:215)


### Relabel by teacher

Replace the trace row's stored `action` with the teacher-predicted action.

The original label is preserved as `original_action`.

Relevant code:

- [rl/relabel_traces.py](/home/luc/rl-nethack-worktree-20260416/rl/relabel_traces.py:13)


## Comparison To Raw ASCII Board Or Image Input

Approximate per-turn token costs for LLM-style input:

- compact current-state text: about `50-100` tokens
- enriched memory + viewport prompt: about `90-220` tokens typical
- full visible ASCII board every turn: about `500-1200` tokens
- image frame through a vision model: highly model-dependent, but roughly `300-1500+` image-token-equivalent is a reasonable ballpark

Takeaway:

- compact structured text is much cheaper than raw ASCII board dumps
- the enriched prompt is still fairly cheap compared with full ASCII
- image input is expensive and model-dependent


## 256k Context Window Estimate

Assume:

- total context: `256,000` tokens
- reserve `8,000` tokens for model output
- usable input budget: about `248,000` tokens

Estimated number of prior turns that fit:

### Compact structured state

- `~70` tokens/turn: about `3,540` turns
- `~100` tokens/turn: about `2,480` turns

If you also keep past action outputs in context:

- `~75-115` tokens/turn
- about `2,150-3,300` turns


### Enriched memory + viewport prompt

- `~130` tokens/turn: about `1,900` turns
- `~200` tokens/turn: about `1,240` turns

With action outputs included:

- `~135-215` tokens/turn
- about `1,150-1,830` turns


### Full ASCII board each turn

- `~800` tokens/turn: about `310` turns
- `~1,200` tokens/turn: about `206` turns

With action outputs included:

- `~805-1,215` tokens/turn
- about `204-308` turns


## Bottom-Line Assessment

The current setup is:

- good for short-horizon supervised learning
- decent for local state/action imitation
- weak for long-horizon sequential reasoning

Biggest limitation:

- most current paths do not feed the model a real multi-turn raw history window

The main representational gap is probably not token budget. It is lack of explicit sequence history.
