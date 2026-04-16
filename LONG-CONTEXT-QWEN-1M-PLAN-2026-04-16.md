# Long-Context Qwen 1M Plan

Date: 2026-04-16

## Decision

We are adopting a new mainline direction for the repo:

- primary backbone: `Qwen/Qwen2.5-14B-Instruct-1M`
- primary training target: `long rolling full-game history -> next action`
- primary data unit: full trajectories, not disconnected state/action pairs

This replaces the old assumption that the main route should be:

- short-context behavior cloning
- `state + action -> delta`
- tiny-window world-model-only training

Those paths remain useful as baselines and auxiliary supervision, but they are not the center anymore.

## Why This Change

The current repo has already taught us several useful lessons.

### Lesson 1: tiny local datasets are not enough

The early sequence world-model branch worked as an implementation prototype, but the original `v2` corpus was tiny:

- `273` merged training rows
- about `121` train windows for recent sequence runs

That was enough to implement RSSM-style training and planner probes, but not enough to build a serious long-horizon NetHack model.

### Lesson 2: full trajectories matter

We confirmed that training windows should come from complete episodes, not isolated fragments.

Why:

- long-horizon dependencies matter
- NetHack is partially observable
- delayed consequences matter
- deaths and terminal segments matter
- loops, backtracking, and revisitation matter

Short windows are still valid training units, but the corpus itself should be made of full trajectories.

### Lesson 3: plain predictor quality and planner quality split apart

The sequence world-model branch did not converge to one universally best checkpoint.

We ended up with different fronts:

- predictive best
- planner-ranking best
- compromise best

That means we should stop pretending one scalar metric defines progress.

### Lesson 4: tiny replay probes lie

Several planner improvements looked good on tiny or single-seed probes and then weakened or reversed on wider replay checks.

So:

- single-seed evaluation is not enough
- tiny-slice evaluation is not enough
- model selection must be robust by default

### Lesson 5: more data helped the predictor side immediately

When we moved to the larger local full-trajectory `pipeline106` corpus:

- train rows increased to `2448`
- held-out rows increased to `432`
- train windows increased to `1215`

Predictive feature error improved materially, which confirms that scale is a real bottleneck.

### Lesson 6: the current small RSSM is useful, but probably not the best final backbone

The RSSM branch is planner-friendly and cheap, but it has a weak prior and is data-hungry.

Given:

- `4x H200`
- willingness to train large models
- the need for long-span memory and abstraction

the best mainline is likely a pretrained long-context model with NetHack-specific finetuning, not a small scratch-trained latent model alone.

## Main Hypothesis

The strongest near-term path is:

- use a long-context pretrained Qwen as the main sequence model
- train it on full NetHack trajectories
- predict the next action from long rolling history
- inject structured NetHack features as side information
- add world-model-style auxiliary heads for reward, value, termination, and optionally next-feature prediction
- use large-scale mixed-quality trajectories for representation learning
- upweight wins and late-game traces for action supervision
- evaluate with robust long-horizon held-out slices, not flattering toy probes

## Model Choice

Primary target:

- `Qwen/Qwen2.5-14B-Instruct-1M`

Reason:

- the model is explicitly built for very long context
- the repo is already Qwen-oriented
- `4x H200` makes this a realistic large-model finetuning target
- it is a much stronger prior for long-sequence reasoning than the small in-repo RSSM

Source:

- Qwen model card: <https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M>

## Training Objective

### Main objective

Train a long-context action model:

- input: a long prefix of a real game up to time `t`
- target: the action at time `t`

Each timestep in the prefix should include:

- board representation
- message text
- bottom-line stats
- previous action
- compact structured features
- optional inventory / event markers

### Auxiliary objectives

We should carry forward the useful parts of the world-model branch as auxiliary supervision, not as the mainline objective.

Candidate auxiliary heads:

- reward prediction
- done prediction
- value / return-to-go prediction
- legal-action prediction
- next-feature prediction
- planner-trace policy target

This lets the LLM learn:

- action selection from long context
- latent belief quality
- useful planning-related signals

without forcing the whole system to be a tiny latent-only world model.

## Data Strategy

The world model lesson is clear: volume and coverage matter.

For this new direction, we should train on a lot of data, and that data does not need to be winning-only.

### Core rule

For world understanding:

- transition coverage matters more than win rate

For action quality:

- wins and strong trajectories matter more

So the corpus should be mixed, not pure.

### Priority order

1. `NLD-NAO` human trajectories
2. public-server ttyrecs / xlog-backed games
3. local AutoAscend full games
4. HiHack / LangHack style converted corpora
5. local existing corpora already in this repo

### Why mixed-quality trajectories still matter

Even losing games contain valuable supervision for:

- local navigation
- combat transitions
- item interactions
- partial-observation memory
- board-state progression
- message interpretation

What losing games are bad for:

- naive blanket “all actions here are bad” training

So the rule should be:

- use broad trajectories for representation and auxiliary modeling
- use stronger trajectories for high-weight action imitation
- use losing trajectories selectively for negative training later

## Full-Trajectory Requirement

We should train from complete episodes.

That means:

- store full games end-to-end
- keep exact order
- preserve episode boundaries
- keep terminal endings
- keep long prefixes available for packing

The training batches can still be windows, but those windows must be sampled from full episodes.

Training unit:

- long packed prefix ending at step `t`

Data unit:

- full game trajectory

## Input Representation

We should not depend on text alone when we already have structured information.

Each timestep should support both:

- natural-language / serialized game text
- structured numeric side inputs

### Textual sequence payload

Per step:

- full board serialization
- message line
- bottom-line stats
- action just taken
- compact event tags

The existing board serialization work should be reused where applicable.

### Structured side channels

Per step:

- `feature_vector`
- allowed actions mask
- optional planner trace summaries
- optional compact inventory features

Best implementation pattern:

- project structured features through an MLP
- inject as soft tokens or fused side embeddings
- train end-to-end with the language backbone

## Two-Track Architecture

We should not throw away the existing world-model work.

The best architecture is probably two-track:

### Track A: main model

Long-context Qwen policy / belief model:

- long history in
- next action out
- auxiliary reward/value/done heads

### Track B: fast planner model

Optional smaller latent model:

- distilled from the main model later
- used for cheap branching rollout or replanning

This is likely better than asking the LLM itself to do expensive token-by-token search at runtime.

## Dataset Format

We need one canonical format that works for both:

- long-context LLM training
- auxiliary world-model supervision

Minimum per-step fields:

- `episode_id`
- `step`
- `action`
- `allowed_actions`
- `reward`
- `done`
- `prompt`
- `feature_vector`
- `observation_version`

Useful optional fields:

- `planner_trace`
- `value_target`
- `source_corpus`
- `won`
- `seed`
- `phase`
- `obs_hash`
- `inventory_summary`
- `board_ascii`
- `board_compact`

Rules:

- never silently mix feature dimensions
- never silently mix incompatible observation versions
- split at episode level
- keep source labels for every corpus

## Packing Strategy

The dataset format should be `1M`-ready from the start, even if training ramps up progressively.

Target buckets:

- `128k`
- `256k`
- `512k`
- `1M`

Recommended curriculum:

1. start mostly with `128k` and `256k`
2. add a meaningful fraction of `512k`
3. introduce `1M` windows early at low frequency
4. increase `1M` share as throughput and stability permit

The important part is:

- do not build a short-format dataset now and rebuild everything later

## Loss Design

### Primary loss

- next-action cross-entropy

### Secondary losses

- value regression
- reward regression
- done prediction
- legal-action prediction
- planner-policy imitation when available
- optional next-feature prediction

### What not to do

- do not let a planner proxy become the sole checkpoint selector
- do not trust tiny single-seed ranking improvements
- do not overweight pairwise ranking losses just because they spike on a cheap probe

The world-model branch already showed those failure modes.

## Evaluation

This part has to be stricter than the earlier prototype branch.

### Required scorecards

1. Action prediction scorecard
   - top-1 accuracy
   - top-k accuracy
   - accuracy by episode depth
   - accuracy by source corpus
   - accuracy on late-game slices

2. Auxiliary prediction scorecard
   - reward MAE
   - value MAE
   - done calibration
   - legal-action accuracy

3. Long-context sensitivity scorecard
   - same eval slice under truncated contexts
   - measure gain from `4k`, `32k`, `128k`, `256k`, `512k`, `1M`

4. Robustness scorecard
   - multi-seed decoding or planner probes
   - fixed deterministic benchmark slices
   - held-out episode-level splits

### What counts as real progress

Real progress is:

- better long-horizon action quality on held-out episodes
- better late-game decision quality
- context-length scaling that actually helps
- stable gains across seeds and eval slices

Not real progress:

- one cherry-picked seed
- one tiny replay slice
- one internal-only metric

## Negative Training on Losing Traces

We should use losing data carefully.

Valid later-stage uses:

- segment-level negatives near obvious blunders
- preference targets against stronger alternatives
- loop / stall penalties
- selective KTO or pairwise preference training

Invalid use:

- marking every action in every losing episode as bad

## Immediate Blockers

1. The current repo is still centered around smaller short-context paths.
2. We do not yet have the full external long-trajectory corpus in canonical format.
3. We do not yet have the long-context Qwen finetuning path as the repo’s mainline trainer.
4. We do not yet have the long-context eval suite that will keep us honest.

## Execution Plan

### Phase 0: lock the direction

- [x] Decide that the mainline is long-context Qwen, not tiny delta prediction
- [x] Keep the existing world-model branch as supporting infrastructure, not the final backbone
- [ ] Update repo defaults so short-context paths are clearly labeled baseline-only

### Phase 1: dataset acquisition

- [ ] Acquire the largest usable external full-trajectory corpora
- [ ] Start with `NLD-NAO` and `NLD-AA`
- [ ] Audit public ttyrec / xlog-backed sources
- [ ] Pull in AutoAscend full games at scale if external winning corpora are sparse
- [ ] Preserve source labels and win/loss metadata

### Phase 2: canonical corpus builder

- [ ] Build or extend one canonical full-trajectory converter
- [ ] Emit one row per step with stable schema
- [ ] Store board text plus structured side features
- [ ] Add shard manifests and episode-level splits
- [ ] Produce corpus audit artifacts before every train

### Phase 3: long-context training stack

- [ ] Add the main long-context Qwen trainer path
- [ ] Support structured feature fusion
- [ ] Support auxiliary heads for reward/value/done/legal actions
- [ ] Support mixed-length packing up to `1M`
- [ ] Validate stable training on `4x H200`

### Phase 4: baseline long-context finetune

- [ ] Train plain next-action prediction first
- [ ] Measure gains from longer context lengths
- [ ] Establish a strong held-out long-context baseline before adding negatives

### Phase 5: auxiliary supervision

- [ ] Add value / reward / done heads
- [ ] Add legal-action head
- [ ] Add planner-trace policy targets where available
- [ ] Compare against the plain action-only baseline

### Phase 6: selective negative training

- [ ] Add segment-level negative or preference training only after the baseline is stable
- [ ] Avoid blanket losing-episode penalties
- [ ] Measure whether negatives help late-game action quality without hurting broad behavior

### Phase 7: compression / planning follow-up

- [ ] Distill the long-context model into a smaller planner-friendly latent model if needed
- [ ] Reuse the existing sequence world-model path as the likely student path
- [ ] Benchmark whether distillation preserves action ranking and value quality

## Working Folklore

These are the warnings worth preserving.

- Tiny datasets produce unstable conclusions.
- Full trajectories matter more than curated fragments.
- Single-seed and tiny-slice probes are easy to fool.
- One “best checkpoint” is often a false simplification.
- Better predictive metrics do not guarantee better planner behavior.
- Planner-proxy checkpoint selection can overfit badly.
- Pairwise ranking losses can look exciting and still be a trap.
- More data helped immediately; architecture churn is not the only lever.
- If we can afford a stronger pretrained backbone, we should use it.

## Success Criteria

- We have a canonical large-scale full-trajectory corpus.
- We can finetune a long-context Qwen model on `4x H200`.
- The model uses both text history and structured side features.
- Long-context action quality beats short-context baselines on held-out episodes.
- Auxiliary heads improve useful planning signals without degrading core action quality.
- Evaluation is robust enough that a “win” survives wider held-out checks.

## Source Links

- Qwen 1M model card: <https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M>
- NLE repo / NLD support: <https://github.com/facebookresearch/nle>
- Dungeons and Data paper: <https://proceedings.neurips.cc/paper_files/paper/2022/file/9d9258fd703057246cb341e615426e2d-Paper-Datasets_and_Benchmarks.pdf>
- Dungeons and Data supplemental: <https://proceedings.neurips.cc/paper_files/paper/2022/file/9d9258fd703057246cb341e615426e2d-Supplemental-Datasets_and_Benchmarks.pdf>
- AutoAscend repo: <https://github.com/maciej-sypetkowski/autoascend>
- HiHack dataset page: <https://upiterbarg.github.io/hihack-demo/>
- LangHack dataset card: <https://huggingface.co/datasets/upiter/LangHack/blob/main/README.md>
