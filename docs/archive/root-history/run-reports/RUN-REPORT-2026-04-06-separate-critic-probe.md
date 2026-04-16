# Separate Critic Probe Report

## Purpose

Test a precise hypothesis from the new `0.9875` world-model-distilled teacher line:

- maybe the first APPO updates are degrading the policy because actor and critic still share the same encoder,
- so value updates may be perturbing the actor backbone even when the warm-start is exact.

The concrete intervention was:

- use separate actor and critic encoders,
- warm-start both encoders from the BC teacher,
- keep the short-gate schedule otherwise identical.

## Code Paths Touched

- [rl/config.py](/home/luc/rl-nethack/rl/config.py)
- [rl/model.py](/home/luc/rl-nethack/rl/model.py)
- [rl/train_appo.py](/home/luc/rl-nethack/rl/train_appo.py)
- [rl/trainer.py](/home/luc/rl-nethack/rl/trainer.py)
- [rl/teacher_reg.py](/home/luc/rl-nethack/rl/teacher_reg.py)
- [cli.py](/home/luc/rl-nethack/cli.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

## Implementation

Added support for:

- `--separate-actor-critic`
- `ModelConfig.actor_critic_share_weights`
- APPO argv emission of `--actor_critic_share_weights=False`
- BC warm-start copying into:
  - `actor_encoder.*`
  - `critic_encoder.*`
  - plus the actor action head

I also generalized actor parameter anchoring to enumerate all actor hidden linear layers instead of hard-coding the old two-layer shared-encoder keys.

## Validation

Commands:

```bash
uv run pytest -q tests/test_rl_scaffold.py -k 'trainer_scaffold_includes_teacher_reg_args or build_appo_config_infers_hidden_size_from_bc_checkpoint or bc_warmstart_uses_final_linear_for_deeper_teacher or bc_warmstart_copies_teacher_to_separate_actor_and_critic_encoders'
uv run pytest -q tests/test_rl_scaffold.py
```

Result:

- `68 passed`

## Experiment

Experiment name:

- `train_dir/rl/appo_v4_distill_ensemble_l3pure_probe_sepcritic_a`

Teacher artifact:

- `/tmp/x100_v4_distill_ensemble_l3_pure_h1024.pt`
- held-out trace match: `0.9875`

Teacher regularizers:

- `/tmp/x100_v4_distill_textdistil_c020_t2_h512.pt`
- `/tmp/x100_v4_distill_textdistil_c025_t2_h512.pt`

Short-gate config matched the previous faithful-bridge probe except for separate actor/critic weights:

- `model_hidden_size=1024`
- `model_num_layers=3`
- `separate_actor_critic=True`
- `learning_rate=1e-4`
- `reward_scale=0.005`
- `gamma=0.99`
- `gae_lambda=0.9`
- `value_loss_coeff=0.1`
- scheduled teacher loss and replay
- deterministic held-out trace monitoring

## Results

### Step 0

Warm-start metadata:

- [warmstart_trace_match.json](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_probe_sepcritic_a/checkpoint_p0/warmstart_trace_match.json)

Trusted result:

- env step `0`: `0.9875`

So the separate-critic branch preserves the teacher exactly at initialization, just like the repaired shared-backbone branch.

### Learned checkpoints

Checkpoint ranking:

- [best_trace_match.json](/home/luc/rl-nethack/train_dir/rl/appo_v4_distill_ensemble_l3pure_probe_sepcritic_a/checkpoint_p0/best_trace_match.json)

Ranked results:

- env step `512`: `0.9375`
- env step `1024`: `0.8875`
- env step `1536`: `0.8125`

### Comparison to the shared-backbone probe

Shared-backbone probe:

- warm-start `0.9875`
- best learned `0.975` at env step `256`
- later `0.925`, `0.9125`

Separate-critic probe:

- warm-start `0.9875`
- best learned `0.9375` at env step `512`
- later `0.8875`, `0.8125`

So this intervention is clearly worse on the trusted metric.

## Interpretation

This hypothesis did not hold up.

What we learned:

- critic interference through a shared encoder is not the main explanation for the early APPO degradation on this branch
- separating actor and critic did not make the first learned checkpoint better
- in fact it made short-run policy quality materially worse

The most plausible reasons are:

- the shared-backbone inductive bias was actually helping in this small-data regime
- critic sharing was not the dominant source of drift
- the main problem is still the actor update objective relative to the trusted teacher objective

## Recommendation

Do not promote the separate-critic APPO branch.

Keep the infrastructure because it is useful and now tested, but treat the result as negative evidence.

The next online-improver hypothesis should target:

- the actor update rule itself,
- teacher-state regularization strength,
- or a more explicitly behavior-constrained improver,

not just the actor/critic sharing topology.
