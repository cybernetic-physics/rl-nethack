# TODO

## P0: Canonicalize long-context policy training and runtime format

## Problem

The current long-context policy path still has a train/runtime mismatch.

What the code already does:

- [src/long_sequence_dataset.py](/home/luc/rl-nethack-worktree-20260416/src/long_sequence_dataset.py) builds rolling-history next-action rows using the repo’s own board/state rendering path
- [src/board_view.py](/home/luc/rl-nethack-worktree-20260416/src/board_view.py) already provides the reusable exact and tokenized full-board serializers
- [src/long_sequence_eval.py](/home/luc/rl-nethack-worktree-20260416/src/long_sequence_eval.py) evaluates long-sequence datasets by normalizing model outputs with `parse_action(...)`
- [src/nld_long_sequence_import.py](/home/luc/rl-nethack-worktree-20260416/src/nld_long_sequence_import.py) imports NLD/ttyrec data, but currently emits episode rows from raw tty text via `state_text`
- [nle_agent/agent_http.py](/home/luc/rl-nethack-worktree-20260416/nle_agent/agent_http.py) defines the runtime action map and action parsing logic used by online inference

What is still wrong:

- imported NLD policy data can still train on raw tty-style `external_text` screens rather than the canonical runtime prompt surface
- action labels from imported data can include modal/UI actions that do not match the intended online control loop
- runtime action parsing and offline dataset normalization are not yet one explicit shared policy-spec layer
- the repo has long-sequence data builders and eval, but not one canonical replay pipeline that rebuilds every corpus into the exact runtime prompt family

This issue is now important enough to treat as the main data-quality blocker for the long-context branch.

## Source Anchors

Use these as the implementation and audit anchors:

- docs:
  - [docs/consolidated-2026-04/05-blockers-and-next-steps.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/05-blockers-and-next-steps.md)
  - [docs/consolidated-2026-04/07-operator-quickstart.md](/home/luc/rl-nethack-worktree-20260416/docs/consolidated-2026-04/07-operator-quickstart.md)
  - [LONG-CONTEXT-QWEN-1M-PLAN-2026-04-16.md](/home/luc/rl-nethack-worktree-20260416/LONG-CONTEXT-QWEN-1M-PLAN-2026-04-16.md)
  - [LONG-CONTEXT-NLD-TRAINING-RESULTS-2026-04-16.md](/home/luc/rl-nethack-worktree-20260416/LONG-CONTEXT-NLD-TRAINING-RESULTS-2026-04-16.md)
- long-context code:
  - [src/long_sequence_dataset.py](/home/luc/rl-nethack-worktree-20260416/src/long_sequence_dataset.py)
  - [src/long_sequence_eval.py](/home/luc/rl-nethack-worktree-20260416/src/long_sequence_eval.py)
  - [src/long_sequence_benchmark.py](/home/luc/rl-nethack-worktree-20260416/src/long_sequence_benchmark.py)
  - [src/board_view.py](/home/luc/rl-nethack-worktree-20260416/src/board_view.py)
  - [src/nld_long_sequence_import.py](/home/luc/rl-nethack-worktree-20260416/src/nld_long_sequence_import.py)
- runtime/action code:
  - [nle_agent/agent_http.py](/home/luc/rl-nethack-worktree-20260416/nle_agent/agent_http.py)
- operator surfaces:
  - [cli.py](/home/luc/rl-nethack-worktree-20260416/cli.py)
  - [README.md](/home/luc/rl-nethack-worktree-20260416/README.md)
- tests to extend:
  - [tests/test_long_sequence_dataset.py](/home/luc/rl-nethack-worktree-20260416/tests/test_long_sequence_dataset.py)
  - [tests/test_long_sequence_eval.py](/home/luc/rl-nethack-worktree-20260416/tests/test_long_sequence_eval.py)
  - [tests/test_nld_long_sequence_import.py](/home/luc/rl-nethack-worktree-20260416/tests/test_nld_long_sequence_import.py)
  - [tests/test_policy_generation.py](/home/luc/rl-nethack-worktree-20260416/tests/test_policy_generation.py)

## Desired End State

We want one canonical policy interface shared by:

- long-sequence training
- NLD import
- external-episode conversion
- long-sequence evaluation
- live runtime inference

That means:

- one canonical state renderer
- one canonical action vocabulary and normalization layer
- one canonical history packing scheme
- one canonical row-replay path for rebuilding recorded episodes into training rows
- one documented policy spec that the code follows

## Execution Checklist

### 1. Write the canonical policy spec

- [x] Add a short design note under `docs/` that defines the canonical long-context policy interface.
- [x] Define the primary board representation:
  - use [src/board_view.py](/home/luc/rl-nethack-worktree-20260416/src/board_view.py) as the source of truth
  - decide whether `tokenized_board` or `ascii_board` is the canonical training/runtime default
  - keep dual-view persistence only as debug metadata, not as a separate policy family
- [x] Define the canonical prompt fields and ordering:
  - episode identifier
  - turn/step index
  - message text
  - stats line
  - board block
  - rolling prior turns
- [x] Define the canonical history structure:
  - exact separator text
  - whether assistant turns contain only one action token
  - deterministic oldest-first truncation rules
- [x] Define the canonical action vocabulary:
  - movement
  - stairs
  - interaction
  - inventory/gameplay actions
  - modal-only actions, if any
- [x] Define the policy for modal/menu states:
  - drop
  - retain as modal-control examples
  - or convert into canonical fallback examples only when unavoidable

Done when:

- there is one doc the importer, dataset builder, eval path, and runtime can all point at
- the doc names the exact modules that implement the spec

### 2. Audit existing sources against the spec

- [x] Build a compatibility table inside the design note or a sibling audit note.
- [x] Cover these sources explicitly:
  - local long-sequence NLE generation from [src/long_sequence_dataset.py](/home/luc/rl-nethack-worktree-20260416/src/long_sequence_dataset.py)
  - NLD import from [src/nld_long_sequence_import.py](/home/luc/rl-nethack-worktree-20260416/src/nld_long_sequence_import.py)
  - external JSONL conversion through `convert_episode_jsonl_to_long_sequence_dataset(...)`
  - any preference/KTO datasets derived from long-sequence corpora
  - future AutoAscend conversion path
- [x] For each source, record:
  - state renderer used today
  - action labels used today
  - modal/UI row exposure
  - metadata fields available
  - whether it already matches the canonical interface
  - exact transforms needed

Done when:

- every data source has a clear “usable as-is / needs replay / needs filtering” decision

### 3. Extract a shared policy rendering layer

- [x] Add one shared renderer module for canonical policy text rather than letting importers and dataset builders assemble format ad hoc.
- [x] Move or wrap the relevant logic from:
  - [src/long_sequence_dataset.py](/home/luc/rl-nethack-worktree-20260416/src/long_sequence_dataset.py)
  - [src/board_view.py](/home/luc/rl-nethack-worktree-20260416/src/board_view.py)
- [x] Ensure the shared renderer supports:
  - deterministic formatting
  - stable field ordering
  - token counting
  - primary board view selection
  - optional persisted debug views
- [x] Make local NLE generation call the shared renderer.
- [x] Make NLD import call the shared renderer.
- [x] Make external episode conversion call the shared renderer.

Done when:

- there is one importable function or module used by every policy-data builder

### 4. Extract a shared action normalization layer

- [x] Add one shared action-normalization module instead of splitting logic across import and eval code.
- [x] Start from existing runtime behavior in [nle_agent/agent_http.py](/home/luc/rl-nethack-worktree-20260416/nle_agent/agent_http.py):
  - `_build_action_map()`
  - `parse_action(...)`
- [x] Reuse or replace the importer-specific keypress conversion in [src/nld_long_sequence_import.py](/home/luc/rl-nethack-worktree-20260416/src/nld_long_sequence_import.py) so all paths flow through the same canonical names.
- [x] Classify actions into:
  - canonical gameplay actions
  - canonical modal actions
  - dropped/disallowed actions
- [x] Add explicit decisions for:
  - `more`
  - `space`
  - `esc`
  - `look`
  - `whatis`
  - menu navigation
  - death/end-of-run screens

Done when:

- training labels, eval normalization, and runtime parsing all depend on one action-spec module

### 5. Add modal/UI-state detection

- [x] Add row-level heuristics for common tty/UI states:
  - `--More--`
  - inventory/menu screens
  - yes/no prompts
  - “what do you want to ...” prompts
  - inspection/look screens
  - death/grave/end screens
- [x] Attach modal metadata to imported and replayed rows.
- [x] Define a row disposition field:
  - `keep_gameplay`
  - `keep_modal`
  - `drop_modal_noise`
  - `drop_terminal_screen`
- [x] Ensure these flags can flow into dataset manifests and validation reports.

Done when:

- the corpus builder can explain why a row was kept or dropped

### 6. Build dataset replay into canonical rows

- [x] Implement a shared episode replay function that rebuilds recorded step data into the canonical long-context prompt family.
- [x] Inputs it must support:
  - raw observations when available
  - recorded tty/screen text when raw observations are unavailable
  - chronological step/action episode rows from external JSONL
- [x] Outputs it must produce:
  - canonical `conversations` or `messages`
  - canonical assistant completion
  - normalized action label
  - step metadata
  - modal metadata
  - provenance metadata such as `render_source`
- [x] Reuse the same history trimming logic as the long-sequence builder:
  - token budget
  - context bucket labels
  - deterministic oldest-first truncation

Done when:

- replayed corpora no longer depend on free-form `external_text` as the default policy-training mode

### 7. Rework NLD import around replay

- [x] Change [src/nld_long_sequence_import.py](/home/luc/rl-nethack-worktree-20260416/src/nld_long_sequence_import.py) so the default policy-training path is:
  - ingest ttyrec minibatches
  - build chronological episode rows
  - replay into canonical prompt rows
- [x] Add importer switches for:
  - gameplay-only rows
  - keep/drop modal rows
  - optional modal-control export
  - fallback-text rendering when parsing into canonical layout fails
- [x] Ensure imported rows carry:
  - normalized action names
  - modal flags
  - outcome and game-phase metadata
  - render provenance metadata

Done when:

- imported NLD corpora can be used for policy training without silently mixing runtime-incompatible prompt surfaces

### 8. Rebuild existing corpora through replay

- [x] Regenerate the current NLD long-sequence corpus through the canonical replay path.
- [x] Regenerate the long bootstrap corpus through the same path.
- [x] Regenerate preference/KTO corpora derived from these sources so they no longer inherit stale prompt formats.
- [x] Preserve side-by-side manifests so old and rebuilt corpora can be compared.

Notes:

- `data/rebuilt/long_bootstrap/{train,eval,benchmark}.jsonl` were rebuilt through the canonical replay/generation path.
- `data/rebuilt/preferences/` now contains rebuilt bootstrap and rebuilt NLD-derived KTO/weighted preference corpora.
- A valid public raw NLD taster shard was also fetched to `data/nld_hf_taster/` from `Howuhh/nld-aa-taster`:
  - `data/nld_hf_taster/data-cav-gno-neu-any.hdf5`
  - `data/nld_hf_taster/metadata-cav-gno-neu-any.json`
- A real raw-source canonical NLD smoke corpus was rebuilt from that HDF5 shard through the shared replay path:
  - `data/rebuilt/nld_hf_taster/long_sequences_smoke_canonical.jsonl`
  - `data/rebuilt/nld_hf_taster/long_sequences_smoke_canonical.jsonl.validation.json`
  - `data/rebuilt/nld_hf_taster/benchmark.jsonl`
  - `data/rebuilt/preferences/nld_hf_taster_kto_train.jsonl`
  - `data/rebuilt/preferences/nld_hf_taster_weighted_train.jsonl`
  - `data/rebuilt/preferences/nld_hf_taster_pairwise_train.jsonl`
- The local `data/nld/nld-aa-taster.zip` artifact was not a usable zip payload; it was an `AccessDenied` XML response. To avoid blocking the replay work entirely, the NLD rebuild was executed against a sampled shard back-converted from the locally cached historical long-sequence corpus:
  - `data/rebuilt/nld_taster/nld_old_sample_longseq.jsonl`
  - `data/rebuilt/nld_taster/episodes_from_old_longseq_sample.jsonl`
  - `data/rebuilt/nld_taster/nld_canonical_rebuilt_sample.jsonl`
  - `data/rebuilt/nld_taster/nld_canonical_benchmark_sample.jsonl`
- The older sampled back-conversion artifacts are still useful as comparison data, but the current canonical smoke rebuild no longer depends on them.

Done when:

- there are no “mainline” training corpora left in mixed prompt formats

### 9. Add replay validation and reporting

- [x] Add validation utilities that report:
  - rows before replay
  - rows after replay
  - rows dropped by reason
  - modal vs non-modal counts
  - action histogram before normalization
  - action histogram after normalization
  - render-source split
  - context-bucket split
- [x] Add row-sampling checks to confirm:
  - prompt format matches runtime spec
  - history is in the correct order
  - action labels are normalized
  - modal filtering is behaving as intended
- [x] Write validation outputs to JSON so they can be diffed between corpus rebuilds.

Done when:

- every rebuilt corpus has a machine-readable validation report

### 10. Extend tests before scaling data or training

- [x] Add tests for the canonical renderer.
- [x] Add tests for action normalization and alias handling.
- [x] Add tests for modal-row classification.
- [x] Add tests for replayed history assembly and deterministic truncation.
- [x] Add importer tests covering:
  - gameplay rows
  - modal rows
  - dropped rows
  - fallback render-source paths
- [x] Add eval tests proving the same action normalization is used offline and at runtime.

Done when:

- the replay/canonicalization path is covered in `tests/test_long_sequence_dataset.py`, `tests/test_nld_long_sequence_import.py`, `tests/test_long_sequence_eval.py`, and `tests/test_policy_generation.py`

### 11. Re-run the trusted short loop on rebuilt corpora

- [x] Build a deterministic held-out benchmark shard from the rebuilt corpus.
- [x] Run a small long-context LoRA smoke train on rebuilt data.
- [x] Evaluate with `evaluate-long-sequences` on the rebuilt benchmark.
- [x] Run the current online long-context harness on a small fixed seed set.
- [x] Compare against the pre-rebuild baseline:
  - exact match
  - action-family distribution
  - modal-action leakage
  - online invalid/odd action rate

Notes:

- Rebuilt bootstrap smoke train completed at `output/rebuilt/long_bootstrap_qwen_0_5b_smoke256`.
- Offline eval reports:
  - `output/rebuilt/old_bootstrap_eval_report.json` -> exact match `0.34375`
  - `output/rebuilt/rebuilt_bootstrap_eval_report.json` -> exact match `0.15625`
  - `output/rebuilt/rebuilt_nld_sample_eval_report.json` -> exact match `0.0`
- Comparison report written to `output/rebuilt/compare_eval_reports.json`.
- Live online probe written to `output/rebuilt/live_long_eval.json`:
  - seeds `42,43,44`
  - `48` total steps
  - invalid/odd action rate `0.4166666666666667`
- The first short-loop comparison above was run before the replay-history preservation bug was fixed. After fixing canonical-history packing and regenerating the rebuilt bootstrap corpus, a matched v2 smoke comparison was rerun with both old-format and rebuilt-format adapters trained under the same settings:
  - `output/rebuilt/old_model_on_old_benchmark_v2.json` -> exact match `0.0`
  - `output/rebuilt/old_model_on_rebuilt_benchmark_v2.json` -> exact match `0.0078125`
  - `output/rebuilt/rebuilt_model_on_old_benchmark_v2.json` -> exact match `0.0`
  - `output/rebuilt/rebuilt_model_on_rebuilt_benchmark_v2.json` -> exact match `0.0390625`
- On the same rebuilt benchmark, the rebuilt-format smoke adapter now outperforms the old-format smoke adapter.
- A second live online probe with the fixed rebuilt adapter was written to `output/rebuilt/live_long_eval_v2.json`:
  - seeds `42,43,44`
  - `48` total steps
  - invalid/odd action rate `0.7708333333333334`
- The short-loop benchmark criterion is now satisfied in the matched v2 comparison, but online action quality still needs follow-up tuning.

Done when:

- we know whether canonical replay reduced the current train/runtime mismatch in practice

## Recommended Order

Do this in order:

1. write the policy spec
2. extract shared renderer and action normalization
3. add modal detection
4. build replay path
5. rework NLD import to use replay
6. add validation and tests
7. rebuild corpora
8. run short-loop training and eval

## Stop Conditions

Do not scale to larger long-context runs until these are true:

- one canonical policy spec exists
- one shared renderer exists
- one shared action-normalization layer exists
- NLD import uses replay into canonical rows
- rebuilt corpora have validation reports
- the rebuilt short-loop benchmark is at least as good as the current mixed-format baseline

Current status:

- All core code-path items above are implemented.
- The benchmark stop condition is satisfied by the matched v2 rebuilt-vs-old comparison on the rebuilt benchmark.
- The remaining open issue is the partially rebuilt raw NLD corpus from a very large source artifact, plus the still-poor online odd-action rate.
