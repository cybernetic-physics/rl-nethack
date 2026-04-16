# Canonical Policy Spec (2026-04)

## Purpose

This note defines the canonical long-context policy interface used by:

- long-sequence dataset generation
- external episode conversion
- NLD import
- long-sequence evaluation
- runtime action normalization

Primary implementation anchors:

- [src/policy_replay.py](/home/luc/rl-nethack-worktree-20260416/src/policy_replay.py)
- [src/policy_actions.py](/home/luc/rl-nethack-worktree-20260416/src/policy_actions.py)
- [src/long_sequence_dataset.py](/home/luc/rl-nethack-worktree-20260416/src/long_sequence_dataset.py)
- [src/nld_long_sequence_import.py](/home/luc/rl-nethack-worktree-20260416/src/nld_long_sequence_import.py)
- [src/long_sequence_eval.py](/home/luc/rl-nethack-worktree-20260416/src/long_sequence_eval.py)

## Canonical Prompt Family

System prompt:

- one fixed instruction telling the model to choose exactly one NetHack action

User prompt layout:

1. `EpisodeId`
2. `TargetStep`
3. task line
4. `HistoryTurns`
5. `CurrentTurn`
6. `NextAction:`

Each rendered turn uses a canonical state block with:

- `TurnIndex`
- `RenderSource`
- `Message`
- `BoardMode`
- `BoardShape`
- optional `StatusLines`
- `Board`

Assistant target:

- exactly one canonical action string

## Canonical Board Representation

Primary policy board mode:

- `tokenized`

Debug-only alternate:

- exact ASCII board persistence

Implementation:

- observation-backed rendering uses [src/board_view.py](/home/luc/rl-nethack-worktree-20260416/src/board_view.py)
- tty/screen-text rendering is converted into the same board block format through row-based board serialization

## Canonical Action Vocabulary

Runtime/gameplay actions come from the runtime action map in:

- [nle_agent/agent_http.py](/home/luc/rl-nethack-worktree-20260416/nle_agent/agent_http.py)

Shared normalization lives in:

- [src/policy_actions.py](/home/luc/rl-nethack-worktree-20260416/src/policy_actions.py)

Action classes:

- movement
- stairs
- interaction
- inventory/gameplay
- wait
- search
- modal
- dropped

Modal aliases currently recognized:

- `space`
- `esc`
- `more`
- `look`
- `whatis`
- `inventory_menu`
- `yes`
- `no`

Default training/import policy:

- keep gameplay actions
- drop modal actions unless explicitly requested
- drop unrecognized actions instead of silently mapping them to `wait`

## Modal And UI Policy

Screen-text rows are classified by heuristics in:

- [src/policy_replay.py](/home/luc/rl-nethack-worktree-20260416/src/policy_replay.py)

Current modal types:

- `more_prompt`
- `command_prompt`
- `paged_view`
- `inventory_menu`
- `yes_no_prompt`
- `death_screen`
- `none`

Default replay policy:

- drop modal rows
- drop terminal/death screens
- keep gameplay rows

## Replay Policy

Recorded episodes are replayed chronologically through one shared builder.

Replay responsibilities:

- normalize actions
- classify modal rows
- render state blocks into the canonical prompt family
- preserve provenance such as `render_source`
- emit validation counts

Current render sources:

- `obs`
- `tty_text`
- `fallback_text`

## Validation Requirements

Every rebuilt corpus should emit a machine-readable validation report including:

- rows before replay
- rows after replay
- dropped rows by reason
- action histogram before normalization
- action histogram after normalization
- modal split
- render-source split

Current conversion path writes:

- `<output>.validation.json`

## Source Compatibility Table

| Source | Current representation | Canonical path | Main transforms | Usable for policy training? |
|---|---|---|---|---|
| Local long-sequence NLE generation | raw `obs` | replay/render from `obs` | shared renderer, shared action normalization | Yes |
| External episode JSONL with `state_prompt` / `state_text` | free-form text | replay/render from tty/screen text | modal classification, canonical screen layout, action normalization | Yes, after replay |
| NLD import | tty chars and keypresses | tty chars -> episode rows -> replay -> canonical rows | keypress normalization, modal filtering, canonical board block, validation report | Yes, after replay |
| Preference/KTO derivatives | inherited from source corpus | rebuild from canonical corpora | regenerate after corpus rebuild | Not until rebuilt |
| AutoAscend traces | not yet wired into canonical replay path | future episode replay path | source adapter + shared renderer/action normalization | Not yet |
