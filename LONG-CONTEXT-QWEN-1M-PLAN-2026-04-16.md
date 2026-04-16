# Long-Context Qwen 1M Plan (2026-04-16)

## Decision

We are switching the main text-policy direction to:

- `Qwen/Qwen2.5-14B-Instruct-1M`

and we are changing the training target from:

- `state + action -> delta`

to:

- `long rolling game history -> next action`

The repo is currently centered on short-context Qwen models:

- `train.py` defaults to `Qwen/Qwen2.5-3B-Instruct` and a default `--max-seq-length 1024` ([train.py](/home/luc/rl-nethack-worktree-20260416/train.py:87), [train.py](/home/luc/rl-nethack-worktree-20260416/train.py:104))
- `scripts/generate_training_data.py` still defaults to short local serving settings like `--vllm-max-model-len 2048` ([scripts/generate_training_data.py](/home/luc/rl-nethack-worktree-20260416/scripts/generate_training_data.py:637))

That is no longer the main direction.


## Live Checklist

### Done

- [x] Decide on `Qwen/Qwen2.5-14B-Instruct-1M` as the main long-context model.
- [x] Add a long-history next-action dataset builder.
- [x] Use the reusable whole-board serializer in the long-history path.
- [x] Add token-budget-based history trimming with context-bucket labels.
- [x] Add a CLI command to generate long-sequence datasets directly from NLE rollouts.
- [x] Add a CLI command to convert episode-style JSONL corpora into the long-sequence format.
- [x] Add mixed-context dataset generation support so one corpus can emit multiple target context buckets.
- [x] Add metadata-aware training-data filtering so curriculum stages can target specific context buckets.
- [x] Add a staged curriculum launch script for long-context LoRA training.
- [x] Add a subset-selection tool for building focused shards such as `gold_wins`.
- [x] Add a ranked `gold_wins` shard builder that groups and scores episodes.
- [x] Add a long-sequence evaluation path with slice metrics by context bucket, outcome, and game phase.
- [x] Add a selective positive/negative dataset builder for losing-segment mining.
- [x] Add a weighted-SFT training path for mined positive/negative long-sequence rows.
- [x] Add a pairwise preference training path for teacher-preferred losing-segment alternatives.
- [x] Add a KTO-style training path for labeled positive/negative long-sequence rows without introducing new trainer dependencies.
- [x] Add an evaluation-comparison harness for weighted SFT vs pairwise vs KTO on long-sequence benchmarks.
- [x] Add repo-side NLD / ttyrec import tooling for long-sequence conversion.
- [x] Add deterministic long-sequence benchmark-shard tooling.
- [x] Add a Qwen 1M vLLM launch script and config template.
- [x] Add a long-context LoRA launch script.
- [x] Add a weighted preference LoRA launch script.
- [x] Add a pairwise preference LoRA launch script.
- [x] Add a KTO LoRA launch script.
- [x] Add a repo-native fallback LoRA training path that works without `trl` / `unsloth`.
- [x] Add a preference-method comparison launcher.
- [x] Add a native single-run curriculum launcher.
- [x] Add an NLD gold-wins import launcher.
- [x] Add a long-sequence benchmark builder launcher.
- [x] Add a token-budgeted mixed-corpus builder for long-sequence shards with full-episode and strided long-window sampling.
- [x] Add tests for board serialization, long-sequence packing, converter path, and metadata heuristics.

### In Progress

- [~] Keep the checklist below synchronized with the actual codebase.
- [~] Support both compact and exact board views cleanly in long-sequence data products.
  Current status: long-sequence generation can now train on one prompt board mode while optionally persisting both exact ASCII and tokenized board views per example when raw observations are available.
- [~] Add richer episode metadata for outcome and game phase across all corpora, not just flexible external JSONL conversion.
  Current status: flexible external JSONL conversion already carries outcome/phase metadata, and live NLE-generated episodes now promote final episode outcome (`win`/`loss`/`truncated`) plus final-phase labels back onto all generated steps. Remaining work is validating the same richness on real imported corpora such as NLD.
- [~] Build the mixed-length curriculum trainer path inside training, not only dataset generation.
  Current status: dataset-side multi-budget generation, metadata filtering, staged shell launchers, and native single-run curriculum assembly inside `train.py` are implemented. Remaining work is tuning the actual schedule on real long-context runs.

### Not Done

- [~] Validate real `4x H200` `Qwen/Qwen2.5-14B-Instruct-1M` serving.
  Current status: the host does have `4x H200` available and the repo now has the Qwen 1M launch/config path, but the actual vLLM `Qwen/Qwen2.5-14B-Instruct-1M` server has not been brought up yet.
- [~] Mine / import long winning traces.
  Current status: repo-side NLD / ttyrec registration, ranking, and long-sequence import tooling is implemented, including a `gold_wins`-oriented launcher; actual dataset-root registration and import on this host is still pending.
- [~] Build a win-heavy mixed corpus toward the `1B` token target.
  Current status: the repo now has a token-budgeted corpus compiler that can mix full episodes plus very-long / long / medium strided windows from one or more long-sequence shards while emitting a manifest with projected token totals. The remaining blocker is feeding it real win-heavy shards such as NLD ascensions or locally generated AutoAscend full runs.
- [x] Add actual model training on losing-segment negatives.
  Current status: the repo now supports weighted-SFT training on mined positive/negative long-sequence rows, pairwise-preference training on teacher-preferred losing-segment alternatives, and a repo-native KTO-style training path for labeled positives/negatives without requiring `trl`.
- [~] Validate long-horizon evaluation against a real served model on meaningful held-out data.
  Current status: deterministic benchmark building and report comparison are implemented and a local one-step LoRA smoke train has completed on a small shard, but served-model evaluation on meaningful held-out long-horizon data is still pending.


## Why This Change

The current repo already documents the main representational problem:

- most paths only see the current state or a very short history
- the one path with meaningful history only keeps the last 4 explicit turns plus summarized memory

See:

- [INPUT-CONTEXT-NOTES-2026-04-16.md](/home/luc/rl-nethack-worktree-20260416/docs/archive/root-history/research-notes/INPUT-CONTEXT-NOTES-2026-04-16.md)

For NetHack, that is too weak for:

- long-span exploration
- inventory planning
- backtracking
- delayed consequences
- partial observability
- remembering why a local board state is dangerous or promising

So the primary model should learn from as much real trajectory history as we can fit, not from one-step deltas.


## Model Choice

`Qwen/Qwen2.5-14B-Instruct-1M` is the mainline target because:

- the official model card says it supports up to about `1,010,000` tokens of context
- the official deployment guidance explicitly targets Hopper/Ampere GPUs
- the official VRAM guidance says the `14B` 1M model needs about `320GB` total VRAM for 1M-token processing
- we have `4x H200`, which makes this realistic

Sources:

- Qwen model card: <https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M>
- Relevant lines:
  - context length and generation limits: lines `52-64`
  - Hopper recommendation and VRAM guidance: lines `114-126`
  - custom vLLM branch / launch settings: lines `128-204`

This is also the least disruptive model change because the repo is already Qwen-centric.


## Training Objective

### Main objective

Train a sequence policy on long histories:

- input: a long serialized game transcript ending at timestep `t`
- target: the expert / teacher action at timestep `t`

The sequence should include, per step:

- full board view
- message text
- bottom-line stats
- compact structured side data
- previous action

We should use the new reusable whole-board serializer:

- [src/board_view.py](/home/luc/rl-nethack-worktree-20260416/src/board_view.py:1)

That gives us:

- exact full-board ASCII
- a reversible token-aware board encoding

### What we are not centering anymore

Not the mainline:

- `state + action -> delta`
- single-step counterfactual deltas
- ultra-short prompt-only BC with no real history

Those may still remain as auxiliary tasks, but not the main training objective.


## History Packing Strategy

We do want to move directly to a `1M`-capable dataset, but the rollout should still be incremental.

That means:

1. The dataset format should be built for `1M` context from the start.
2. The trainer should support windows at:
   - `128k`
   - `256k`
   - `512k`
   - `1M`
3. Early training should mix shorter and longer windows instead of only using `1M` immediately for every batch.
4. We should increase the fraction of `1M` windows as packing, throughput, and stability improve.

So the incremental part is:

- not a different dataset format
- not a different model
- just a staged packing / sampling schedule on top of a `1M`-ready corpus

This avoids building the wrong pipeline twice.


## Data Strategy

We need long-span winning traces first. If we cannot get enough, we fall back to large-scale AutoAscend traces and keep training the long sequential policy there.

### Priority order

1. `NLD-NAO` winning human traces
2. direct public-server ttyrecs / xlog-backed ascensions
3. local AutoAscend traces we generate ourselves
4. existing large AutoAscend-derived corpora as supplemental long-horizon pretraining


## Best External Sources For Long Winning Games

### 1. NLD-NAO

This is the best known source for real long-span winning games.

The official NLD supplemental reports:

- `NLD-NAO` has `1.46%` ascensions
- `NLD-AA` has `0.00%` ascensions
- `NLD-NAO` has much better late-game coverage than `NLD-AA`

Source:

- NLD supplemental: <https://proceedings.neurips.cc/paper_files/paper/2022/file/9d9258fd703057246cb341e615426e2d-Supplemental-Datasets_and_Benchmarks.pdf>
- Relevant lines: `335-364`

The official NLE repo also says:

- NLD support ships with `NLE`
- users can load both `nld-aa` and `nld-nao`

Source:

- NLE repo: <https://github.com/facebookresearch/nle>
- Relevant lines: `365-380`

This should be our first attempt for winning long-span supervision.

### 2. Public ttyrecs / xlog-backed human ascensions

The same NLD supplemental explains how NAO games are stitched from ttyrec fragments using:

- username
- start / end time
- xlogfile alignment

Source:

- NLD supplemental: lines `327-373`

So if needed, we can reproduce the same approach on public server archives and explicitly mine:

- long games
- ascensions
- late-game runs that reach Gehennom / Amulet / Astral

This is attractive because even if we do not recover a massive dataset immediately, the wins we do recover are exactly the kind of horizon we care about.

### 3. AutoAscend

AutoAscend is still the strongest fallback source.

Official repo:

- AutoAscend is the `1st place NetHack agent` for the NeurIPS 2021 challenge

Source:

- <https://github.com/maciej-sypetkowski/autoascend>
- Relevant lines: `234-244`

This is not the same as winning human trajectories, but it is still a very strong long-horizon expert process and it is already close to the repo’s current instrumentation path.

### 4. HiHack and LangHack

These are useful references and possible auxiliary corpora, but they are not the main source of winning traces.

HiHack:

- around `3 billion` keypresses
- around `100k` AutoAscend games
- each keypress has a full game view plus strategy labels

Source:

- <https://upiterbarg.github.io/hihack-demo/>
- Relevant lines: `3-6`

LangHack:

- built from AutoAscend games
- converted into text
- but sampled into contiguous chunks of `64` timesteps

Source:

- <https://huggingface.co/datasets/upiter/LangHack/blob/main/README.md>
- Relevant lines: `107-109`

LangHack is closer to our new direction than the current repo, but its fixed `64`-step chunks are still shorter than our target regime.


## Winning Traces vs Losing Traces

### Winning traces

Winning traces should be the highest-value supervised corpus because they contain:

- actual late-game competence
- inventory discipline
- route planning
- recovery from dangerous states
- endgame action chains

### Losing traces

Using losing traces negatively can make sense, but only if we do it carefully.

What does make sense:

- KTO / preference-style negative signal on clearly bad actions
- penalizing short bad segments near death or obvious blunders
- pairing teacher / winner actions against bad alternatives on the same or similar states

What does not make sense:

- treating every action in every losing game as globally negative

Reason:

- many long losing runs still contain lots of correct local actions
- a game can be losing because of one late mistake, bad luck, or cumulative drift

So the rule should be:

- `winning traces = high-confidence positive`
- `losing traces = selective negative`, mostly segment-level or action-level, not blanket episode-level rejection


## Planned Dataset Format

Each training example should represent a long prefix of a real game:

- `state_0`
- `action_0`
- `state_1`
- `action_1`
- `...`
- `state_t`
- predict `action_t`

Each `state_i` should include:

- board from [src/board_view.py](/home/luc/rl-nethack-worktree-20260416/src/board_view.py:1)
- message line
- bottom-line stats
- inventory summary if available
- event markers such as stair transitions, item pickup, key discoveries

Each sequence should also store metadata:

- episode id
- turn index
- source dataset (`nld-nao`, `autoascend-local`, etc.)
- whether the episode won
- whether this slice is from a high-value late-game segment
- packing length bucket (`128k`, `256k`, `512k`, `1M`)


## Immediate Blockers

1. The current trainer is still a short-context Unsloth SFT path.
   - default `--max-seq-length` is `1024` in [train.py](/home/luc/rl-nethack-worktree-20260416/train.py:104)
2. The current policy-generation path is also short-context by default.
   - `--vllm-max-model-len` defaults to `2048` in [scripts/generate_training_data.py](/home/luc/rl-nethack-worktree-20260416/scripts/generate_training_data.py:637)
3. Qwen’s official `1M` guidance recommends a custom vLLM branch for best long-context behavior.
4. We do not yet have a canonical long-history dataset builder in this repo.
5. We do not yet have a mined winning-trace corpus.


## Todo

### Phase 0: Commit the decision

- [x] Treat `Qwen/Qwen2.5-14B-Instruct-1M` as the default long-context sequence model.
- [ ] Keep current short-context Qwen paths only as baselines, not roadmap center.

### Phase 1: Infrastructure

- [x] Stand up the repo-side Qwen-recommended long-context serving stack definition for `Qwen/Qwen2.5-14B-Instruct-1M`.
- [ ] Actually launch and validate the serving stack on this host.
- [ ] Verify `4x H200` deployment with long context and stable throughput.
- [x] Add a repo script for launching the `1M` model with the required vLLM settings.
- [x] Add a repo config template for the `1M` launch settings.
- [ ] Record real max context that is stable in our environment: `128k`, `256k`, `512k`, `1M`.

### Phase 2: Sequence dataset builder

- [x] Add a new dataset builder for `long history -> next action`.
- [x] Use [src/board_view.py](/home/luc/rl-nethack-worktree-20260416/src/board_view.py:1) as the board serializer.
- [~] Store both exact and compact board forms, but train primarily on the compact reversible serialization.
  Current status: the builder supports both `ascii` and `tokenized` board modes, and external corpora can still be converted even when only text state is available, but examples currently store one selected board view per training sample rather than persisting both views together.
- [x] Add packing code for `128k`, `256k`, `512k`, and `1M` windows.
  Current status: the builder trims by an explicit token budget and labels the resulting context bucket.
- [x] Expose when a current state alone exceeds a requested synthetic context budget.
- [~] Add episode metadata fields for win/loss, phase of game, and source corpus.
  Current status: source and context metadata are implemented; win/loss and game-phase heuristics are implemented for flexible external corpora conversion, but live NLE-generated examples still only carry provisional outcome labels.
- [x] Add a CLI path to generate long-sequence datasets.
- [x] Add a reusable episode-to-example conversion path so external corpora can be converted later.
- [x] Add a CLI path to convert episode-style JSONL corpora into the long-sequence format.

### Phase 3: Winning-trace acquisition

- [~] Attempt to load / mine `NLD-NAO` first.
  Current status: repo-side import tooling now supports registering NLD-compatible roots, ranking candidate games from the sqlite metadata DB, selecting wins/deep runs, and converting them into long-sequence JSONL. Actual dataset registration and execution against a real NLD root is still pending.
- [~] Extract all ascensions and other deep-game episodes.
  Current status: the importer supports `wins_only`, `min_turns`, `min_maxlvl`, and bounded game counts; this still needs to be run on a real corpus.
- [ ] Rank episodes by:
  - win status
  - total turns
  - late-game achievements
  - sequence length
- [~] Build a `gold_wins` shard from the highest-value long winning games.
  Current status: repo-side episode ranking and shard-building tools exist; actual winning-game corpora still need to be imported and processed.
- [x] Add repo-side tooling to filter and materialize focused long-sequence subsets once the right metadata is present.

### Phase 4: Public-server mining if NLD-NAO is insufficient

- [ ] Reproduce the xlog + ttyrec stitching logic described in the NLD supplemental.
- [ ] Mine public-server ascensions and long deep-game runs.
- [ ] Convert them into the same long-sequence training format.

### Phase 5: AutoAscend fallback / scale path

- [ ] If winning human traces are too scarce, generate local AutoAscend traces at scale.
- [ ] Prefer long uninterrupted games and end-to-end full episodes.
- [ ] Keep source labels so we can distinguish:
  - human winning traces
  - human non-winning traces
  - AutoAscend traces

### Phase 6: Training plan

- [x] Start with a mixed-length curriculum on the `1M`-ready dataset.
  Current status: mixed-context dataset generation, metadata filtering, staged curriculum shell launchers, and native single-run curriculum scheduling inside `train.py` are implemented.
- [ ] Early schedule:
  - mostly `128k` and `256k`
  - some `512k`
  - a small but nonzero fraction of `1M`
- [ ] Later schedule:
  - increase `512k`
  - increase `1M`
  - monitor quality and throughput
- [x] Train next-action prediction first.
- [x] Add a repo launch script for long-context LoRA training.
- [x] Add a staged curriculum launcher that trains context buckets in sequence.
- [~] Add a selective negative objective on losing segments only after the plain policy path is stable.
  Current status: selective weighted-SFT is now supported for mined losing segments, pairwise-preference dataset export and training are supported when teacher alternatives exist, and a repo-native KTO-style LoRA training path is now implemented. Remaining work is empirical comparison rather than missing training code.

### Phase 7: Losing-trace negative training

- [ ] Do not mark whole losing episodes as uniformly bad.
- [~] Build action- or segment-level negatives:
  Current status: there is now a selective positive/negative dataset builder that mines repeated-action runs, repeated weak-action runs, low-HP stall actions, near-death mistakes under dangerous messages, and optional teacher-disagreement rows from losing episodes while keeping winning actions as positives. It can also export pairwise teacher-preferred alternatives when teacher actions are present.
  - near-death mistakes
  - teacher-dispreferred alternatives
  - loops / repeated useless actions
- [x] Add actual training support for those negatives via weighted SFT.
- [x] Add actual training support for teacher-preferred alternatives via pairwise preference training.
- [x] Add actual training support for labeled positive/negative rows via KTO-style training.
- [~] Evaluate whether KTO helps over simple weighted SFT or pairwise preference ranking.
  Current status: the repo now has weighted-SFT, pairwise-preference, and repo-native KTO-style training paths plus a comparison harness for evaluation reports on deterministic long-sequence benchmarks. Remaining work is running the actual experiments and comparing the resulting reports.

### Phase 8: Evaluation

- [~] Add held-out long-horizon sequence eval, not just short local action match.
  Current status: there is now a long-sequence evaluation module and CLI command with exact-action metrics sliced by context bucket, outcome, game phase, turn-depth bucket, action family, dangerous-message slices, post-danger recovery windows, and focused behavior slices for search/inventory/stairs/wait. Real served-model validation is still pending.
- [ ] Measure:
  - action accuracy by turn depth
  - action accuracy on late-game states
  - recovery after dangerous messages
  - inventory / search / stair behavior over long slices
- [x] Keep deterministic benchmark slices so regressions are easy to catch.


## Working Hypothesis

The strongest near-term recipe is:

- `Qwen/Qwen2.5-14B-Instruct-1M`
- long rolling history
- whole-board serialized state
- next-action training
- gold winning traces when available
- AutoAscend long traces for scale
- selective negative signal on losing segments, not blanket punishment

If we can get enough long winning games, those should dominate the highest-weight training shard.

If we cannot, the fallback is still coherent:

- long sequential training on AutoAscend traces we generate locally


## Success Criteria

- We have a `1M`-ready dataset format in-repo.
- We can train `Qwen/Qwen2.5-14B-Instruct-1M` on long NetHack sequences on `4x H200`.
- We have a mined `gold_wins` corpus or a documented proof that it is too scarce.
- We have a long-horizon eval that actually measures whether added context is helping.
- The repo is no longer centered on short one-step delta prediction.


## Immediate Execution Todo

- [x] Generate a small long-sequence train/eval shard from live NLE rollouts.
  Current status: completed with `data/smoke/long_sequence_train.jsonl` and `data/smoke/long_sequence_eval.jsonl`.
- [x] Build a deterministic long-sequence benchmark shard from that eval split.
  Current status: completed with `data/smoke/long_sequence_benchmark.jsonl`.
- [x] Run one small weighted-SFT smoke training pass on the shard.
  Current status: completed as a one-step LoRA smoke run with `Qwen/Qwen2.5-0.5B-Instruct` and the repo-native `transformers+peft` fallback trainer, producing `output/smoke_qwen_0_5b_longseq`.
- [ ] Run one small pairwise-preference smoke training pass if teacher-preference rows are available.
- [ ] Run one small KTO-style smoke training pass if labeled positive/negative rows are available.
- [ ] Launch the Qwen 1M vLLM stack on the `4x H200` host and verify health.
- [ ] Run held-out long-sequence eval against a served model on the deterministic benchmark.
- [ ] Compare weighted SFT vs pairwise vs KTO reports with the comparison harness.
- [ ] Attempt real NLD / winning-trace import and build the first `gold_wins` shard.
- [ ] Promote the best-performing method and context schedule into the default operator path.


## Source Links

- Qwen 1M model card: <https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M>
- NLE repo / NLD support: <https://github.com/facebookresearch/nle>
- Dungeons and Data paper: <https://proceedings.neurips.cc/paper_files/paper/2022/file/9d9258fd703057246cb341e615426e2d-Paper-Datasets_and_Benchmarks.pdf>
- Dungeons and Data supplemental: <https://proceedings.neurips.cc/paper_files/paper/2022/file/9d9258fd703057246cb341e615426e2d-Supplemental-Datasets_and_Benchmarks.pdf>
- AutoAscend repo: <https://github.com/maciej-sypetkowski/autoascend>
- HiHack dataset page: <https://upiterbarg.github.io/hihack-demo/>
- LangHack dataset card: <https://huggingface.co/datasets/upiter/LangHack/blob/main/README.md>
