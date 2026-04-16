# Behavior-Regularized Improvement

Date: `2026-04-05`

## Purpose

This note captures the next algorithmic step if teacher-regularized APPO plus DAgger still stalls below the BC teacher.

The current repo result is:

- BC teacher: `0.6395`
- best teacher-reg APPO: `0.6318`

That is close enough that a more conservative improvement rule is justified.

## Current Minimum Viable Implementation

The repo now includes an experimental behavior-regularized trainer:

- [rl/train_behavior_reg.py](/home/luc/rl-nethack/rl/train_behavior_reg.py)
- CLI command: `rl-train-behavior-reg`

This first version is intentionally simple:

- supervised imitation on trace rows
- plus a conservative KL term toward the empirical behavior action prior
- evaluated on deterministic trace match

This is not a full AWAC or BRAC implementation yet.
It is a lightweight stepping stone that uses the current trace format and current BC model shape.

## Why This Exists

The current failure mode is:

- plain RL reward rises
- teacher-aligned trace match stalls or regresses

That suggests the learner needs stronger constraints toward the behavior distribution, not more unconstrained optimization.

## Intended Next Expansion

If the current scaffold is promising, the next versions should likely move in one of these directions:

1. KL-regularized actor improvement against a frozen BC reference
2. advantage-weighted behavioral cloning over relabeled DAgger traces
3. mixed offline replay plus short online improvement

## Required Inputs

The current trace format is already enough for an initial conservative method:

- `feature_vector`
- `action`
- `allowed_actions`
- `observation_version`

If we move toward true AWAC-like improvement, future rows should also carry:

- short-horizon return estimate
- teacher preference score
- optional student logprob

## Recommendation

Do not replace teacher-reg APPO with this yet.

Use this only if:

- DAgger is working
- v3 features are in place
- teacher-reg APPO still plateaus below BC

At that point, this becomes the next contained algorithmic branch to explore.
