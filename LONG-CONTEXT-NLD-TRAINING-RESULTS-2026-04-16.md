## Long-Context NLD Training Results

Date: 2026-04-16

This note records the first serious long-game training run on the imported `NLD-AA`
long-sequence corpus, the failed `64k` attempt, the successful `32k` fallback, and
the matched online gameplay comparisons against both the raw base model and the
earlier medium bootstrap adapter.

### Goal

Move beyond the small bootstrap long-history runs and train `Qwen/Qwen2.5-14B-Instruct-1M`
on much longer real NetHack games from the imported `NLD-AA` corpus, then evaluate
whether that larger long-game training improves online closed-loop gameplay.

### Data Used

Primary imported corpus:

- [data/nld-aa_long_sequences_smoke.jsonl](/home/luc/rl-nethack-worktree-20260416/data/nld-aa_long_sequences_smoke.jsonl)

Derived training slice:

- [data/nld_large_run/train.jsonl](/home/luc/rl-nethack-worktree-20260416/data/nld_large_run/train.jsonl)
- [data/nld_large_run/eval_tail_1024.jsonl](/home/luc/rl-nethack-worktree-20260416/data/nld_large_run/eval_tail_1024.jsonl)

Important properties of the imported NLD long-sequence data:

- examples are long-history next-action rows, not delta-prediction rows
- prompt format is the repo’s long-sequence chat format
- rows are long:
  - sample mean `context_tokens_estimate`: about `57k`
  - sample mean `history_steps_included`: about `239`
- the sampled source episode was a deep real game:
  - `maxlvl = 18`
  - `outcome = loss`
  - `game_phase = late`

The first extracted `train.jsonl` had one trailing corrupted partial JSON row due to
an interrupted write. That was repaired in place by truncating at the first bad line.

Final repaired counts:

- train rows: `34,157`
- eval rows: `1,024`

### First Large Attempt: 64k Context

Training target:

- model: `Qwen/Qwen2.5-14B-Instruct-1M`
- data: NLD long-game slice
- context: `65536`
- GPUs: `4x H200`
- LoRA rank: `32`
- LoRA alpha: `64`

Command intent:

- use the real imported NLD long-game corpus
- keep the long-history setup
- push the trainer into the `64k` regime

What happened:

- model loading succeeded across all `4` GPUs
- dataset preprocessing succeeded
- train/eval set materialization succeeded
- the run reached the first actual training step
- the first backward pass failed with `CUDA out of memory`

Observed failure mode:

- OOM on the cross-entropy backward path
- attempted allocation was roughly `32-37 GiB`
- live allocated memory was already around `116 GiB` per GPU

Conclusion:

- `64k` context does not fit in the current trainer stack and configuration, even on
  `4x H200`
- the failure was not data corruption or load failure at that point
- the hard blocker was backward-pass memory

### Successful Large Run: 32k Context

Successful training output:

- [output/qwen14b_nld_long_32k_run](/home/luc/rl-nethack-worktree-20260416/output/qwen14b_nld_long_32k_run)
- [output/qwen14b_nld_long_32k_run/training_meta.json](/home/luc/rl-nethack-worktree-20260416/output/qwen14b_nld_long_32k_run/training_meta.json)

Configuration:

- model: `Qwen/Qwen2.5-14B-Instruct-1M`
- data: [train.jsonl](/home/luc/rl-nethack-worktree-20260416/data/nld_large_run/train.jsonl)
- eval: [eval_tail_1024.jsonl](/home/luc/rl-nethack-worktree-20260416/data/nld_large_run/eval_tail_1024.jsonl)
- max sequence length: `32768`
- LoRA rank: `32`
- LoRA alpha: `64`
- GPUs: `4`
- max train examples: `2048`
- max eval examples: `256`
- max steps: `30`
- gradient accumulation: `2`

Observed runtime behavior:

- preprocessing completed successfully
- training started successfully
- all four GPUs stayed near full memory utilization during training
- approximate steady memory: `~118 GiB` per GPU
- step time after startup: about `22-23s` per optimizer step

Final training result:

- final loss: `0.0440`
- steps: `30`
- runtime: `687.3s`
- adapter hash: `b291d3bacf3d37ddb3dfdf6dd9d71324e42a7261d83c746f1783c105e7af6723`

Interpretation:

- a materially larger real long-game run is now proven in this repo
- the `32k` regime is viable on `4x H200`
- the `64k` regime is not yet viable with the current trainer path

### Earlier Medium Adapter Baseline

Prior medium bootstrap-trained adapter:

- [output/qwen14b_long_medium_50step](/home/luc/rl-nethack-worktree-20260416/output/qwen14b_long_medium_50step)
- [output/qwen14b_long_medium_50step/training_meta.json](/home/luc/rl-nethack-worktree-20260416/output/qwen14b_long_medium_50step/training_meta.json)

That earlier run was:

- trained on the smaller bootstrap long-history corpus
- much shorter and cheaper than the NLD run
- already shown to improve behavior over the raw base model by preventing
  `wait`-collapse in short online rollouts

### Online Gameplay Evaluation Setup

The online eval used the same matched live harness throughout:

- same seeds
- same horizon
- same long-sequence prompt format used in training
- direct next-action generation in closed loop

Compared artifacts:

- raw base model prior:
  - [output/qwen14b_long_medium_live_eval_4x32.json](/home/luc/rl-nethack-worktree-20260416/output/qwen14b_long_medium_live_eval_4x32.json)
- new NLD `32k` adapter:
  - [output/qwen14b_nld_long_32k_live_eval_4x32.json](/home/luc/rl-nethack-worktree-20260416/output/qwen14b_nld_long_32k_live_eval_4x32.json)

Evaluation regime:

- seeds: `42, 43, 44, 45`
- horizon: `32` steps per seed

### Base Model Online Result

Raw base model behavior in the matched `4 x 32` online run:

- mean reward: `0.0`
- mean max depth: `1`
- mean final HP: `13.25`
- mean min HP: `13`

Behavior pattern:

- mostly `wait`
- some `pickup`
- effectively degenerate local behavior

### Medium Bootstrap Adapter Online Result

Earlier medium adapter online result:

- mean reward: `0.0`
- mean max depth: `1`
- mean final HP: `14.0`
- mean min HP: `14.0`

Behavior pattern:

- active movement
- mostly `west`, `north`, `east`, `south`
- no invalid actions

Interpretation:

- clearly better than the raw base model in action style
- still no actual progress signal on reward or dungeon depth in this short harness

### New NLD 32k Adapter Online Result

New larger NLD-trained adapter online result:

- mean reward: `0.0`
- mean max depth: `1`
- mean final HP: `14.0`
- mean min HP: `14.0`

Behavior pattern shifted relative to the medium adapter:

- more `east`
- more `search`
- occasional `wait`
- occasional `throw`
- occasional `fire`

No invalid actions were produced.

### Comparison Summary

Against raw base:

- the new NLD adapter is still clearly better behaviorally
- it avoids the base model’s `wait/pickup` collapse
- it preserves the same HP advantage the medium adapter had

Against the earlier medium bootstrap adapter:

- no improvement on:
  - mean reward
  - mean max depth
  - mean final HP
  - mean min HP
- the main difference is action distribution, not outcome

Direct deltas from the saved eval report:

- vs medium adapter:
  - `mean_reward`: `0.0`
  - `mean_max_depth`: `0`
  - `mean_final_hp`: `0`
  - `mean_min_hp`: `0`
- vs base:
  - `mean_reward`: `0.0`
  - `mean_max_depth`: `0`
  - `mean_final_hp`: `+0.75`
  - `mean_min_hp`: `+1.0`

### What We Learned

1. The imported NLD long-game path is now operational.
2. The repo can train a real long-context NLD adapter on `4x H200`.
3. `64k` context is currently too large for the backward pass in this trainer.
4. `32k` context is viable and stable.
5. The larger NLD-trained adapter does not yet beat the earlier medium adapter on the
   current short online gameplay harness.
6. The current online harness is still weak as a success metric for long-horizon
   competence:
   - no depth progress
   - no reward separation
   - only short horizon

### Current Bottom Line

The big result is not that we achieved better NetHack play. The big result is:

- we successfully moved from tiny bootstrap long-history training
- to a real imported long-game human-like corpus
- and trained a `Qwen 14B` long-context adapter on it at `32k` context on `4x H200`

But the current online evaluation still says:

- better than raw base collapse
- not yet better than the prior medium adapter on actual outcome

### Failure Analysis: Why The New Adapter Behaved This Way

After the larger NLD run and the online gameplay re-evaluation, I inspected both:

- the actual online eval behavior traces
- the long-sequence NLD training rows in [data/nld_large_run/train.jsonl](/home/luc/rl-nethack-worktree-20260416/data/nld_large_run/train.jsonl)

The result is that the observed behavior is explainable. The main issue was not
that training “failed” in an abstract sense. The main issue is that the model
learned exactly the wrong distribution for the current online harness.

#### 1. Prompt-format mismatch between training and online evaluation

The online evaluation prompt used the repo’s compact tokenized board rendering:

- `BoardMode: tokenized`
- RLE board rows from [src/board_view.py](/home/luc/rl-nethack-worktree-20260416/src/board_view.py)
- compact `Stats:` line plus compact `Message:`

The imported NLD training rows were not in that format. They were in:

- `BoardMode: external_text`
- ttyrec-style full screen text
- menu and modal text exactly as shown in the original session

So the model was trained on one state serialization and tested on another. That
is a real train/eval distribution mismatch, and it likely hurt online transfer.

#### 2. The NLD training slice is dominated by UI/menu/meta actions

The target-action distribution in the repaired NLD training slice was:

- `space`: `7177`
- `search`: `4100`
- `look`: `2436`
- `east`: `2020`
- `west`: `1918`
- `read`: `1428`
- `whatis`: `1425`
- `apply`: `1305`
- `southwest`: `1328`
- `num_5`: `950`
- `throw`: `782`
- `esc`: `657`
- `more`: `618`
- `fire`: `445`

This is a poor direct-action policy distribution for the current bare online
rollout harness. The model learned lots of:

- menu continuation
- inspection commands
- modal interaction commands
- inventory / special combat commands

That maps directly onto the online behavior shift seen in evaluation:

- the medium bootstrap adapter mostly produced movement
- the new NLD adapter started producing much more `search`
- and occasionally `throw` / `fire`

#### 3. The histories themselves are full of menu and modal states

Looking through concrete rows in [data/nld_large_run/train.jsonl](/home/luc/rl-nethack-worktree-20260416/data/nld_large_run/train.jsonl), many histories include text like:

- `View which?`
- `(end)`
- `--More--`
- bag interaction menus
- attribute screens
- end-of-game / epitaph / score screens

So even when the final action on a row is a normal game action, the surrounding
history often contains UI-navigation behavior. That biases the model toward
meta-actions that are specific to ttyrec-style session flow, not to the simple
online rollout harness.

#### 4. The “large” training slice was only two episodes

The repaired NLD training slice contained only:

- `nld-aa-local-4467`: `18,083` rows
- `nld-aa-local-60556`: `16,074` rows

So although the row count was large, the episode diversity was extremely low.
This likely caused the model to overfit narrow local habits from only two long
games, including their menu-navigation patterns.

#### 5. The online results match these failure modes

The saved online comparison in [output/qwen14b_nld_long_32k_live_eval_4x32.json](/home/luc/rl-nethack-worktree-20260416/output/qwen14b_nld_long_32k_live_eval_4x32.json) showed:

- base model:
  - mostly `wait` / `pickup`
- medium bootstrap adapter:
  - mostly directional movement
- new NLD adapter:
  - much more `east`
  - much more `search`
  - occasional `wait`
  - occasional `throw`
  - occasional `fire`

So the new adapter did not become worse randomly. It became more like the
action distribution it was trained on.

#### 6. The online evaluation harness is still too weak

Even with these clear behavioral differences, the current short online harness
still showed:

- mean reward: `0.0`
- mean max depth: `1`

for both the medium adapter and the NLD adapter.

So the harness can detect gross collapse, like:

- base model waiting forever

but it is still weak at separating:

- useful long-horizon behavior
- from noisy active behavior

over short horizons like `4 seeds x 32 steps`.

### Refined Diagnosis

The new adapter underperformed relative to hopes because we trained on the wrong
policy slice for this online use case:

- wrong state serialization for eval transfer
- too many menu / meta / UI actions
- too few distinct episodes
- no inference-time action masking or sanitization

The model then learned a policy distribution that is plausible for raw ttyrec
session imitation, but poorly matched to the current stripped-down online
gameplay harness.

### Next Recommended Moves

1. Run a longer online evaluation on the new adapter:
   - `8-16` seeds
   - `128+` steps
2. Add inference-time action masking or action sanitization:
   - the NLD-trained adapter emits actions like `throw` and `fire`
   - these may be valid in corpus terms but unhelpful in the current bare harness
3. Improve the online harness to measure actual exploration/progression:
   - tiles explored
   - rooms found
   - branch/depth progress
   - not just reward / HP / depth after `32` steps
4. Revisit memory efficiency for `64k+`:
   - smaller batch / accumulation geometry
   - alternate attention kernels
   - stronger checkpointing or packing changes
5. Train a second `32k` NLD run with a tighter filtered subset:
   - stronger long games
   - fewer noisy menu / terminal / `--More--` rows
   - explicitly remove rows dominated by:
     - `space`
     - `esc`
     - `look`
     - `whatis`
     - `more`
     - bag / menu / `View which?` screens
   - ensure many more distinct episodes instead of just two huge ones
