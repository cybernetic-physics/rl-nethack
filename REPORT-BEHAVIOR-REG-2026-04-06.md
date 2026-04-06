# Behavior-Reg Results 2026-04-06

## Summary

The first behavior-regularized branch is real, but it did not beat the current best teacher.

Current best trusted baseline remains:

- `v4 + wm_concat_aux`
- best held-out trace match: `0.95`
- source branch: `appo_wm_v4_schedreplay_short_a`

## What We Ran

### Offline behavior-reg training

Train split:

- `/tmp/x100_v4_train_wm_aux.jsonl`

Held-out split:

- `/tmp/x100_v4_heldout_wm_aux.jsonl`

Runs:

1. Default behavior-reg
2. Balanced behavior-reg with weak-action boost
3. Balanced behavior-reg after masked-prior fix

### Online continuation from behavior-reg

Online fine-tune:

- `appo_from_breg_v4_balanced_short_a`
- `appo_from_breg_v4_balanced_fix_short_a`

## Results

### Offline behavior-reg

Default behavior-reg:

- held-out trace match: `0.9000`

Balanced/boosted behavior-reg:

- held-out trace match: `0.9250`

Balanced/boosted behavior-reg after masked-prior fix:

- held-out trace match: `0.9125`

This means the best offline behavior-reg teacher was still below the current teacher line of `0.9500`.

### Online continuation from behavior-reg teacher

Experiment:

- `train_dir/rl/appo_from_breg_v4_balanced_short_a`

Trusted ranking:

- best checkpoint: `0.9375`
- later checkpoints: `0.9250`

So online APPO recovered some performance from the weaker behavior-reg teacher, but still did not beat the current best branch.

### Online continuation from fixed behavior-reg teacher

Experiment:

- `train_dir/rl/appo_from_breg_v4_balanced_fix_short_a`

Trusted ranking:

- best checkpoint: `0.9250`
- later checkpoints: `0.9250`

So the post-fix behavior-reg teacher did not improve online. It simply held its offline score.

## Latest Rerun

I reran the short online continuation after the masked-prior fix to confirm the branch end to end.

Rerun experiment:

- `train_dir/rl/appo_from_breg_v4_balanced_fix_short_a`

Rerun trusted ranking:

- best checkpoint: `0.9250`
- later checkpoints: `0.9250`

So the rerun confirmed the same result:

- the fixed behavior-reg teacher is stable enough to fine-tune
- but the online learner does not improve it on the held-out trace benchmark

## What We Learned

1. The behavior-reg code path trains and evaluates correctly.
2. The branch is not broken operationally.
3. On the current benchmark, behavior-reg is still weaker than the best existing teacher.
4. Online APPO fine-tuning from that teacher still does not beat the best branch.
5. After the masked-prior fix, this branch now looks more like:
   - stable but weaker
   - not unstable and promising

## Bug Found In Behavior-Reg

The original behavior-reg loss had a real design flaw:

- the KL regularizer used a global action prior
- but it did not respect per-row action masks
- this meant the model could be penalized relative to actions that were unavailable in that row

That is not a valid regularization target.

The trainer has now been fixed to:

- mask the behavior prior by row
- renormalize over allowed actions
- fall back to a uniform distribution over allowed actions when needed

## Interpretation

This fix makes the behavior-reg trainer more coherent, but it does not by itself prove that behavior-reg will beat the current teacher.

What it does mean is:

- the old implementation had a real conceptual flaw
- but fixing that flaw did not improve the current benchmark result
- so the main limitation of this branch is now the objective itself, not just a trainer bug

## Next Step

The next correct step is:

1. do not replace the current mainline with behavior-reg yet
2. if this branch continues, change the objective more substantially than a masked-prior KL
3. keep using the held-out deterministic trace benchmark as the gate
4. do not run a larger behavior-reg online continuation until a short run beats `0.9250`
