# TODO

## P0: Fix training/runtime format mismatch for long-context policy training

### Why this is a breaking issue

We currently have a major disconnect between:

- the state/action format used to train the long-context policy, and
- the state/action format we expect the model to handle at runtime.

In practice this means:

- training data, especially NLD-derived data, often uses `external_text` screen dumps taken from ttyrec-like terminal states,
- those states include menu screens, modal prompts, `--More--`, inventory UIs, inspection screens, graves, and other non-core gameplay interfaces,
- the action labels in that data include many UI/meta actions like `space`, `esc`, `look`, `whatis`, `more`, `throw`, `fire`, and similar commands,
- but runtime inference currently uses our own structured long-sequence prompt format built from the tokenized board view, stats, and messages.

So the model is not actually being trained on the same interface that we ask it to act on online.

This is not a cosmetic issue. It causes:

- representation mismatch: train prompt surface form differs from inference prompt surface form,
- action-space mismatch: train targets include many actions that are undesirable or invalid for the runtime harness,
- context mismatch: train histories are polluted by modal/UI states that rarely appear in the intended online control loop,
- eval mismatch: online eval can understate or misread policy quality because the trained model is optimized for a different prompt/action distribution.

This likely explains why:

- the NLD-trained adapter stopped collapsing into `wait`,
- but started producing `search`, `throw`, and `fire`,
- and did not convert long-context training into meaningful gameplay improvement on the current online harness.

### Desired end state

We need a single canonical policy interface:

- one canonical state serialization,
- one canonical action vocabulary,
- one canonical history packing scheme,
- one canonical training/eval/runtime prompt family.

All policy training data, including NLD, AutoAscend, and locally generated long-sequence data, should be converted into that canonical interface before training.

The online runtime harness and offline eval harness must consume the exact same interface.

### Canonical policy format decisions to make

- Decide whether the canonical board view is:
  - tokenized board view from `src/board_view.py`,
  - exact ASCII board view,
  - or dual-view with one fixed primary representation.
- Decide which side channels are always present:
  - stats,
  - message log,
  - inventory summary,
  - depth/turn,
  - task/phase markers,
  - danger indicators.
- Decide the canonical history structure:
  - `state, action, state, action, ...`
  - exact separators,
  - whether assistant turns contain only the chosen action or richer structured action metadata.
- Decide the canonical action vocabulary:
  - core movement and gameplay actions,
  - inventory and menu actions allowed only when the state explicitly requires them,
  - normalized aliases and disallowed actions.

### Subtasks

#### 1. Write the canonical runtime/training spec

- Create a short design note that defines the canonical policy prompt format precisely.
- Include:
  - board serialization format,
  - metadata fields,
  - history packing order,
  - action normalization rules,
  - allowed action vocabulary,
  - rules for modal/menu states.
- Make this spec the reference used by:
  - dataset builders,
  - importers,
  - training code,
  - online eval harness,
  - live gameplay inference.

#### 2. Audit all existing policy data paths against the canonical spec

- Enumerate current sources:
  - local long-sequence generation,
  - NLD import,
  - AutoAscend traces,
  - bootstrap/wagmi corpora,
  - any preference or weighted-SFT datasets.
- For each source, document:
  - current board/state representation,
  - current action labels,
  - current metadata fields,
  - whether it already matches the canonical format,
  - exact mismatches.
- Produce a compatibility table:
  - `source -> current format -> required transforms -> usable for policy training?`

#### 3. Build a canonical state renderer for all training paths

- Ensure there is one shared renderer for policy state text.
- It should support:
  - full-board canonical rendering,
  - deterministic formatting,
  - stable field ordering,
  - optional dual-view persistence for debugging,
  - token-count accounting.
- NLD import must not bypass this renderer.
- AutoAscend conversion must not bypass this renderer.
- Local NLE-generated long-sequence data must use the same renderer.

#### 4. Build an action normalization and filtering layer

- Create a single action-normalization module.
- Map variant/imported action strings into canonical actions.
- Mark actions as:
  - allowed gameplay actions,
  - allowed modal actions,
  - disallowed/meta noise.
- Define explicit handling for:
  - `space`,
  - `esc`,
  - `more`,
  - `look`,
  - `whatis`,
  - menu navigation,
  - inventory submenus,
  - death/end screens.
- Ensure training and eval use the same action normalization.

#### 5. Detect and classify modal/UI states

- Add row-level classifiers or heuristics for:
  - `--More--`,
  - inventory menus,
  - “what do you want to ...” prompts,
  - viewing screens,
  - death/grave/end screens,
  - yes/no or selection prompts.
- Attach modal-state metadata to imported rows.
- Decide whether each row should:
  - be removed,
  - be retained with a modal action label,
  - or be converted into a canonical modal state/action example.

#### 6. Clean the NLD importer

- Change the NLD path so it emits the canonical runtime-format state serialization rather than raw `external_text` training rows for policy learning.
- Add switches to:
  - drop modal/menu rows entirely,
  - keep only gameplay rows,
  - optionally retain a separate modal-control dataset for later use.
- Ensure the importer outputs:
  - canonical `messages`,
  - canonical `completion`,
  - normalized `action`,
  - modal-state flags,
  - outcome/phase metadata.

#### 6a. Rebuild datasets by replaying recorded step histories into the canonical prompt format

- The required replay is dataset-level replay, not emulator-level replay.
- We do not need to replay the full NetHack engine state to fix this issue.
- We do need to replay each recorded episode step-by-step through our serializer so that the final training rows match runtime usage.

What this means in practice:

- take recorded per-step state/action data,
- walk the episode in chronological order,
- rebuild the history window at each step,
- render each step with the canonical state formatter,
- emit a new long-sequence training row whose prompt exactly matches the runtime prompt family.

Sources that should support this replay path:

- NLD-derived step data,
- local long-sequence NLE-generated episodes,
- AutoAscend traces once available,
- any other imported episode JSONL that has step-ordered state/action rows.

Required implementation tasks:

- Build a shared `replay_episode_to_canonical_examples(...)` path that:
  - takes chronological step rows,
  - reconstructs the rolling history,
  - applies the canonical state renderer,
  - applies action normalization,
  - emits canonical `messages` plus next-action `completion`.
- Ensure this shared replay path is used by:
  - NLD import,
  - generic external JSONL conversion,
  - local trace conversion,
  - future AutoAscend conversion.
- Make the replay deterministic:
  - stable row ordering,
  - stable history trimming,
  - stable action normalization,
  - stable prompt text.

#### 6b. Define exactly what per-step state gets replayed from recorded data

- For sources with raw observations:
  - replay from `obs` and render the canonical board/state text directly.
- For sources with only recorded screen text:
  - replay from the recorded screen text after converting it into the canonical board/state layout.
- Explicitly avoid keeping a separate `external_text` policy-training mode as the default path.

Needed decisions:

- whether recorded tty text is parsed into:
  - exact canonical board block,
  - a lossy but standardized board/message/state block,
  - or a fallback canonical text screen block only when parsing fails.
- how to preserve source provenance so we can audit:
  - `render_source=obs`,
  - `render_source=tty_text`,
  - `render_source=fallback_text`.

#### 6c. Add replay-time history reconstruction controls

- The replay builder should support:
  - full-episode replay,
  - strided replay,
  - late-game-only replay,
  - danger-window replay,
  - modal-row drop/keep policies.
- History reconstruction should use the same budget logic as runtime-oriented training:
  - token budget,
  - turn budget,
  - context bucket labels,
  - deterministic truncation from the oldest turns first.

#### 6d. Add replay validation

- For every rebuilt corpus, sample rows and verify:
  - the prompt format matches runtime format,
  - the action label is normalized,
  - removed rows are correctly labeled as modal/UI drops,
  - history windows are assembled in the correct order.
- Add validation reports showing:
  - row counts before replay,
  - row counts after replay,
  - counts dropped by reason,
  - action histogram before/after normalization,
  - modal/non-modal split,
  - rendering-source split.

#### 6e. Rebuild all existing long-sequence policy corpora using replay

- Rebuild the current canonical corpora through the replay path rather than leaving old rows in mixed formats.
- At minimum regenerate:
  - the NLD policy corpus,
  - the current long bootstrap corpus,
  - any preference datasets derived from these corpora.
- Mark older pre-replay corpora as legacy and do not use them for new policy training runs.

#### 7. Rebuild the NLD training corpus

- Regenerate the NLD-based training dataset using the cleaned importer.
- Require:
  - canonical prompt format,
  - normalized actions,
  - filtered UI/meta rows,
  - broader episode coverage than the prior two-episode-heavy slice.
- Build explicit shards:
  - training shard,
  - eval shard,
  - benchmark shard,
  - optional modal shard.
- Record action distributions before and after cleanup.

#### 8. Align runtime inference to the same canonical prompt

- Confirm the online gameplay harness uses the exact same prompt structure as the cleaned training corpus.
- Remove any formatting drift between:
  - long-sequence dataset builder,
  - eval harness prompt builder,
  - live inference path.
- If different prompt builders exist, consolidate them.

#### 9. Add action masking / inference sanitization

- Runtime should not freely emit actions that are impossible or undesired in the current state.
- Add inference-time sanitization for:
  - impossible commands,
  - disallowed modal actions in normal gameplay states,
  - out-of-vocabulary outputs.
- If the state is non-modal, suppress pure menu/meta actions.
- If the state is modal, restrict to the appropriate modal action subset.

#### 10. Add dataset quality reports

- For every policy corpus build, emit a report with:
  - action histogram,
  - modal vs non-modal row counts,
  - outcome distribution,
  - game-phase distribution,
  - episode count,
  - mean/median context length,
  - number of rows removed by cleanup,
  - top suspicious actions and screens.
- Store this report beside the dataset and push it to HF with the corpus.

#### 11. Re-run the medium and large training experiments on the cleaned corpus

- Re-run a medium smoke run first:
  - same model family,
  - same trainer,
  - cleaned canonical-format corpus.
- Then re-run the larger NLD-based training run.
- Compare against prior runs on:
  - train loss,
  - eval exact action match,
  - late-game slices,
  - post-danger recovery,
  - online gameplay metrics.

#### 12. Tighten online evaluation

- Expand online eval beyond the current shallow harness.
- Use:
  - more seeds,
  - longer horizons,
  - clearer success metrics,
  - runtime-valid action masks.
- Track:
  - movement entropy,
  - invalid action rate,
  - modal action rate in non-modal states,
  - reward,
  - depth,
  - survival,
  - exploration progress.

#### 13. Keep raw-screen training only as a separate research branch

- If we still want to experiment with raw ttyrec text or UI-rich screen training, keep it separate.
- Do not mix it silently into the main policy corpus.
- Label it explicitly as:
  - raw-screen branch,
  - modal-control branch,
  - or auxiliary imitation branch.

### Immediate next actions

- Write the canonical policy prompt/action spec.
- Implement shared action normalization.
- Implement modal/UI row classification.
- Modify NLD import to emit canonical runtime-format examples.
- Rebuild a cleaned NLD training shard.
- Re-run the medium training/eval loop on the cleaned shard before any new large run.

### Success criteria

We can say this issue is fixed only when all of the following are true:

- training and runtime use the same canonical prompt format,
- training and runtime use the same normalized action vocabulary,
- NLD-derived policy rows are filtered or normalized for modal/UI states,
- cleaned corpora show sane action distributions,
- online eval no longer shows obvious train/runtime prompt mismatch effects,
- the retrained model improves not just action style but actual gameplay metrics.
