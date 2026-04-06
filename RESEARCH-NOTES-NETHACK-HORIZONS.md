# Research Notes: NetHack Horizons and Step Budgets

## Question

Are the current rollout lengths, episode horizons, and teacher trace lengths in this repo too short for NetHack?

Short answer:

- `rollout-length` is probably not the main problem.
- episode horizon and teacher trace horizon are still short relative to serious NetHack work.
- total RL training budget is also still small by NetHack standards.

## Current Repo State

From the current codebase:

- APPO update unroll:
  - default `--rollout-length 64` in [cli.py](/home/luc/rl-nethack/cli.py)
  - many recent runs used `32`
- RL episode horizon:
  - default `env_max_episode_steps = 5000` in [rl/config.py](/home/luc/rl-nethack/rl/config.py)
- teacher / trace horizons:
  - many defaults are `20`, `30`, or `50` steps in [cli.py](/home/luc/rl-nethack/cli.py)
  - forward-model and trace generation often use `50` in [scripts/generate_training_data.py](/home/luc/rl-nethack/scripts/generate_training_data.py)

This means:

- APPO learns from short update chunks of `32-64` steps
- the actual RL episode is usually capped at `5000` steps
- many teacher traces are only `20-50` steps long

## What The Literature Uses

### NLE bundled agent

The NLE repo includes a TorchBeast example:

- `--unroll_length 80`
- `--total_steps 1000000000`

Source:

- [facebookresearch/nle](https://github.com/facebookresearch/nle)

Takeaway:

- chunk lengths around `80` are normal
- total training budgets are enormous compared to our current runs

### Sample Factory NetHack

Sample Factory’s NetHack integration uses:

- `--rollout=32`
- published model trained for `2B` environment steps

Source:

- [Sample Factory NetHack docs](https://www.samplefactory.dev/09-environment-integrations/nethack/)

Takeaway:

- our current `32` rollout is not unusual
- our total step budget is still tiny

### nle-language-wrapper

The language-wrapper NetHack agent reports results after roughly:

- `700M` frames

Source:

- [ngoodger/nle-language-wrapper](https://github.com/ngoodger/nle-language-wrapper)

### Motif

Motif trains NetHack APPO agents with:

- `2,000,000,000` environment steps

Source:

- [facebookresearch/motif](https://github.com/facebookresearch/motif)

### HiHack / NetHack is Hard to Hack

HiHack reports:

- `env max episode steps = 100000`

Source:

- [HiHack paper PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/764ba7236fb63743014fafbd87dd4f0e-Paper-Conference.pdf)

Takeaway:

- our current episode cap of `200` is extremely small relative to challenge-style or hierarchy-style NetHack work

### NetHack Challenge

Challenge evaluation uses:

- termination at `50,000` steps

Source:

- [AIcrowd NetHack Challenge rules](https://www.aicrowd.com/challenges/neurips-2021-the-nethack-challenge/challenge_rules)

## Interpretation

### 1. Rollout length is probably not the main issue

Our current:

- `rollout-length = 32`

Literature:

- `32`, `64`, `80` are all plausible

Conclusion:

- the APPO update chunk size is not obviously wrong
- changing `32` to `64` may matter somewhat, but it is not the main horizon gap

### 2. Episode horizon is short

Our current default:

- `env_max_episode_steps = 5000`

Compared to literature:

- challenge-scale work often uses `50k-100k`

Conclusion:

- `5000` is much closer to a meaningful long-horizon skill-learning regime
- `5000` is still short of full-game NetHack, but no longer toy-scale

### 3. Teacher traces are likely too short

Our common trace lengths:

- `20-50` steps

Conclusion:

- good for debugging
- too short for sustained exploration skill learning
- likely encourages local action preferences rather than longer-horizon behavior

### 4. Total RL budget is still small

Our recent full run:

- about `1.024M` env steps

Literature:

- `700M`
- `1B`
- `2B`

Conclusion:

- even if our alignment stack is improved, we are still under-training compared to serious NetHack systems

## Practical Recommendation

Use two separate regimes.

### Fast debug regime

Keep this for bug fixing and short comparisons:

- `rollout-length 32`
- `env_max_episode_steps 200`
- trace length `20-50`
- short trace-gated sweeps

### Real learning regime

For the next serious skill-learning experiments:

- `rollout-length 32` or `64`
- `env_max_episode_steps 5000`
- teacher trace length `100-200`
- larger total env budget than `1M`

Why this is the right next step:

- it increases behavior horizon without immediately jumping to challenge-scale `50k+` episodes
- it keeps the fast debug loop intact
- it gives the teacher and student a chance to express longer exploration behavior

## Most Important Takeaways

- APPO unroll length is not obviously too short.
- RL episode horizon is improved, but still short of challenge/full-game NetHack.
- teacher traces are probably too short for meaningful long-horizon skill learning.
- total RL training budget remains tiny relative to prior NetHack work.
- the next sensible change is not “make everything huge,” but:
  - keep the fast loop small
  - make the real training loop longer in episode horizon and trace horizon

## Suggested Next Repo Changes

1. Add `fast` and `full` training presets.
2. Raise `env_max_episode_steps` for serious runs from `200` to `5000`.
3. Raise teacher trace generation for serious runs from `20-50` to `100-200`.
4. Keep `rollout-length` around `32-64`.
5. Re-run the teacher-aligned pipeline at the longer horizon before changing algorithms again.

## Sources

- [facebookresearch/nle](https://github.com/facebookresearch/nle)
- [Sample Factory NetHack docs](https://www.samplefactory.dev/09-environment-integrations/nethack/)
- [ngoodger/nle-language-wrapper](https://github.com/ngoodger/nle-language-wrapper)
- [facebookresearch/motif](https://github.com/facebookresearch/motif)
- [HiHack paper PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/764ba7236fb63743014fafbd87dd4f0e-Paper-Conference.pdf)
- [AIcrowd NetHack Challenge rules](https://www.aicrowd.com/challenges/neurips-2021-the-nethack-challenge/challenge_rules)
