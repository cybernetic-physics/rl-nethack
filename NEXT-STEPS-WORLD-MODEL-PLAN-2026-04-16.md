# Next Steps World Model Plan

Date: 2026-04-16

## Goal

Move the current sequence world-model branch from a tiny local prototype to a serious long-horizon NetHack world model trained on large numbers of full trajectories.

This plan assumes:

- the current sequence world-model code path remains the base implementation
- full-trajectory data matters more than curated “good” traces
- the immediate bottleneck is dataset scale and evaluation robustness, not missing small loss terms

## Current state

The repo already has:

- a sequence-window RSSM-style world model
- rollout supervision with context burn-in
- planner-facing heads and replay evaluation
- uncertainty-aware and disagreement-aware planner scoring hooks

But the current `v2` training corpus is still tiny:

- `273` rows in the merged `v2` train file
- about `121` train windows for recent sequence runs

That is enough for implementation and smoke validation, not enough for a planning-grade NetHack world model.

## What we need to do next

### 1. Build a large full-trajectory dataset path

We need a data ingestion pipeline for large external NetHack corpora.

Target sources:

- NetHack Learning Dataset (NLD)
- NAO / alt.org ttyrec archives
- any compatible full-game NetHack trajectory dump with complete temporal ordering

Requirements:

- preserve full episode boundaries
- preserve exact step order
- keep raw action sequence
- keep terminal states and death endings
- support long games, not just filtered fragments

Deliverables:

- a new ingestion script under `rl/` or `scripts/`
- a normalized episode schema
- a documented output format for local training

### 2. Define one canonical normalized trace schema

We need one schema that every dataset converter writes.

Minimum fields:

- `episode_id`
- `step`
- `action`
- `allowed_actions`
- `reward`
- `done`
- `prompt` or text observation proxy
- `feature_vector`
- `planner_trace` when available
- `observation_version`

Optional but useful:

- `seed`
- `policy`
- `obs_hash`
- `next_obs_hash`
- auxiliary metadata like inventory/message hashes

Rules:

- one observation version per corpus split
- one feature dimensionality per trained checkpoint
- no silent mixing of incompatible feature spaces

### 3. Support large-scale episode storage and streaming

We should not depend on loading everything into memory as Python lists for serious dataset scale.

We need:

- streaming JSONL reading or shard iteration
- optional shard manifests
- episode-aware sampling
- train/val/test split at the episode level

If needed later:

- compressed shards
- precomputed window index files

### 4. Convert full trajectories into training windows

Training should still use windows, but those windows must come from full episodes.

We need:

- overlapping window extraction from complete games
- configurable context length
- configurable rollout horizon
- sampling across early/mid/late game
- explicit inclusion of terminal windows

We should avoid:

- only sampling “interesting” states
- excluding failures and deaths
- breaking episode order

### 5. Create two training tracks

We should maintain two separate tracks instead of forcing everything into one mixed feature space.

Track A:

- current `v2` 160-dim world-model branch
- used for continuity with current reports

Track B:

- larger full-trajectory corpus branch, likely on the older 106-dim pipeline features or a new canonical feature extractor
- used to actually scale the world model

The key rule is:

- do not silently merge 106-dim and 160-dim traces into one model

### 6. Retrain the current best recipe on the larger corpus

Once the larger corpus exists, retrain the current strongest compromise recipe first before inventing new architecture changes.

Starting recipe:

- `v15`-style compromise checkpoint selection
- sequence RSSM backbone
- planner-policy warmup
- overshooting enabled

Then compare:

- plain predictor metrics
- replay planner metrics
- multi-seed robustness

### 7. Upgrade evaluation from small probes to robust benchmarks

We need evaluation that is hard to fake.

Required:

- larger replay slices than the cheap 10-row probe
- multi-seed replay reporting by default
- separate predictive and planner scorecards

Add next:

- trajectory-level receding-horizon evaluation
- closed-loop replanning on held-out episodes
- evaluation by episode depth band

### 8. Keep planner changes behind robust validation

Recent work showed:

- single-seed replay can overstate wins
- disagreement penalties can look good on tiny probes and fail on wider checks
- uncertainty penalties are checkpoint-dependent

So every planner change should be judged by:

- multi-seed replay
- wider held-out slices
- clear comparison against the current frontier

### 9. Add dataset quality reports

Before training on any new corpus, produce a dataset report.

It should include:

- row count
- episode count
- average / median episode length
- action distribution
- terminal fraction
- reward distribution
- planner-trace coverage
- feature dimensionality
- observation versions

This should be saved as a JSON or markdown artifact per corpus.

### 10. Preserve working folklore in docs

The branch has already accumulated useful negative results.

We should keep updating the report with:

- what failed
- what improved only on small probes
- what survived wider replay checks

This avoids repeating dead ends when the branch gets compressed later.

## Recommended execution order

1. Implement large-corpus ingestion for full NetHack trajectories.
2. Define and document the canonical normalized trace schema.
3. Build a dataset audit report command.
4. Create the first large merged full-trajectory corpus.
5. Train the `v15`-style sequence world model on that larger corpus.
6. Run held-out prediction eval.
7. Run larger multi-seed replay planner eval.
8. Only after that, revisit uncertainty/disagreement planner penalties.

## Immediate concrete tasks

### Checklist Status

- [x] Task 1: create a dataset ingestion / corpus tooling module for full-trajectory traces
- [x] Task 2: create a dataset audit command
- [x] Task 3: build a first larger corpus manifest and merged/split outputs
- [x] Task 4: train the current best compromise world-model recipe on that larger corpus
- [x] Task 5: run held-out predictive eval and a multi-seed replay planner eval artifact

### Task 1

Create a dataset ingestion module for full NetHack trajectories.

Output:

- normalized JSONL episodes
- one row per step

### Task 2

Create a dataset audit command that summarizes:

- rows
- episodes
- feature dim
- observation version
- planner-trace coverage

### Task 3

Build a first large corpus manifest and merged output.

### Task 4

Train the current best compromise world-model recipe on that corpus.

### Task 5

Run:

- held-out predictive eval
- multi-seed replay planner eval
- report update

## Success criteria

We should consider the next phase successful only if:

- training data scale increases by at least an order of magnitude
- full episodes, not just fragments, are used as the corpus source
- evaluation moves beyond tiny replay probes
- the new model improves on either:
  - predictive quality without planner regression
  - planner quality without collapse
  - ideally both

## Non-goals

For the immediate next phase, we should not:

- mix incompatible feature spaces into one checkpoint
- spend more time on tiny coefficient sweeps over the tiny local corpus
- treat single-seed replay gains as real wins
- optimize for winning trajectories only

## Bottom line

The next step is not another clever loss.  
The next step is a real full-trajectory dataset pipeline and retraining at much larger scale.
