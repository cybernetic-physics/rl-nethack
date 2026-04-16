# Run Report: X100 Validation on Repaired Non-RNN Warm Start

Date: 2026-04-05

## Purpose

Validate the latest RL harness fixes under a much larger APPO training budget, while using the deterministic trace benchmark as the source of truth instead of training reward.

This run was motivated by two bugs found in the previous APPO path:

1. The BC warm-start checkpoint was being deleted before the learner loaded it.
2. The recurrent APPO bridge was not actually equivalent to the BC teacher because a random GRU sat between the copied encoder weights and the copied action head.

## Code Fixes Validated In This Run

The run exercised these fixes:

- `rl/trainer.py`
  - warm-start runs now clean the experiment directory themselves
  - Sample Factory is launched in `resume` mode for warm-start runs so `checkpoint_000000000_0.pth` is actually loaded
- `rl/train_appo.py`
  - added `--no-rnn`
  - `--no-rnn` forces `recurrence=1`
- `cli.py`
  - exposed `--no-rnn` on `rl-train-appo`

## Baseline Before This Run

Trusted deterministic trace benchmark:

- Trace set: `data/tracefix_v2_explore_traces.jsonl`
- BC teacher-aligned baseline:
  - `match_rate = 0.6395`
- Previous broken recurrent APPO result:
  - `match_rate = 0.5853`
- First broken warm-start checkpoint from the recurrent path:
  - `match_rate = 0.3721`

That `0.3721` result was the smoking gun that the old warm-start bridge was broken.

## Validation Smoke Check

Before the larger run, I validated the repaired bridge with a tiny non-RNN smoke experiment:

Command:

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-appo \
  --experiment appo_tracefix_v2_bc_nornn_smoke2 \
  --num-workers 1 \
  --num-envs-per-worker 1 \
  --rollout-length 8 \
  --recurrence 8 \
  --batch-size 8 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 32 \
  --enabled-skills explore \
  --observation-version v2 \
  --bc-init-path output/tracefix_v2_explore_bc.pt \
  --no-rnn
```

Deterministic trace ranking:

- `checkpoint_000000000_0.pth`: `0.6047`
- `checkpoint_000000006_48.pth`: `0.5775`

This confirmed the repaired warm-start path was now close to the BC baseline instead of collapsing immediately.

## Main X100-Target Run

Command:

```bash
CUDA_VISIBLE_DEVICES=2 uv run python cli.py rl-train-appo \
  --experiment appo_tracefix_v2_bc_nornn_x100 \
  --num-workers 4 \
  --num-envs-per-worker 8 \
  --rollout-length 32 \
  --recurrence 16 \
  --batch-size 1024 \
  --num-batches-per-epoch 1 \
  --ppo-epochs 1 \
  --train-for-env-steps 5000000 \
  --enabled-skills explore \
  --observation-version v2 \
  --bc-init-path output/tracefix_v2_explore_bc.pt \
  --no-rnn
```

Important note:

- The configured target was `5,000,000` env steps.
- I manually stopped the run after the repaired path had clearly validated and enough checkpoints existed to rank them deterministically.
- Final collected frames before interruption: `147,456`
- Average training throughput during the stable part of the run: about `212-214 FPS`

## Artifacts

Experiment directory:

- `train_dir/rl/appo_tracefix_v2_bc_nornn_x100`

Important checkpoints:

- `train_dir/rl/appo_tracefix_v2_bc_nornn_x100/checkpoint_p0/checkpoint_000000120_122880.pth`
- `train_dir/rl/appo_tracefix_v2_bc_nornn_x100/checkpoint_p0/checkpoint_000000144_147456.pth`
- `train_dir/rl/appo_tracefix_v2_bc_nornn_x100/checkpoint_p0/best_000000129_132096_reward_78.613.pth`

Trace benchmark input:

- `data/tracefix_v2_explore_traces.jsonl`

Teacher BC checkpoint:

- `output/tracefix_v2_explore_bc.pt`

## Deterministic Results

Checkpoint ranking command:

```bash
uv run python cli.py rl-rank-checkpoints \
  --experiment appo_tracefix_v2_bc_nornn_x100 \
  --trace-input data/tracefix_v2_explore_traces.jsonl \
  --top-k 10
```

Best deterministic trace results observed:

- `checkpoint_000000120_122880.pth`
  - `match_rate = 0.6240`
- `checkpoint_000000144_147456.pth`
  - `match_rate = 0.6163`
- `best_000000129_132096_reward_78.613.pth`
  - `match_rate = 0.6163`

Comparison to prior baselines:

- BC: `0.6395`
- repaired non-RNN warm-start at step 0: `0.6047`
- best current APPO checkpoint in this run: `0.6240`
- old broken recurrent APPO line: `0.5853`

## What Worked

1. The warm-start checkpoint is now actually loaded.
2. The non-RNN bridge preserves most of the BC behavior.
3. The policy no longer collapses immediately during early APPO updates.
4. The deterministic trace metric improved over both:
   - the repaired step-0 warm start (`0.6047 -> 0.6240`)
   - the old recurrent APPO path (`0.5853 -> 0.6240`)

This is the first APPO result in this repo that materially closes the gap to BC on the trusted trace benchmark without the earlier warm-start collapse.

## What Did Not Work

1. APPO still did not beat the BC teacher.
   - Best APPO checkpoint: `0.6240`
   - BC baseline: `0.6395`
2. Training reward was not sufficient to select the best checkpoint.
   - Best-by-reward did not beat the best-by-trace checkpoint.
3. The policy still over-predicts `west`.
   - Best checkpoint action counts:
     - `north: 60`
     - `east: 62`
     - `south: 47`
     - `west: 88`
     - `search: 1`

## Action-Level Interpretation

The repaired APPO run is now much closer to the teacher’s directional mix, but it still has a structural bias:

- `west` is over-predicted
- `search` is still essentially absent
- `east`/`south` remain better than in the broken recurrent path, but not fully solved

This is consistent with the earlier fast-loop diagnosis that the remaining bottleneck is directional representation plus teacher alignment, not just “more RL scale”.

## Key Conclusion

The main conclusion from this run is not that the repo is solved. It is that the recent harness fixes were real and necessary:

- the old recurrent warm-start path was invalid
- the repaired non-RNN warm-start path produces a stable APPO improvement trajectory
- deterministic trace evaluation correctly detected that improvement

This run changes the diagnosis from:

- “APPO might just be bad here”

to:

- “the old bridge was broken; the repaired bridge works, and now the remaining gap is scientific rather than infrastructural”

## Recommended Next Steps

1. Keep the non-RNN BC-to-APPO path as the current best control baseline.
2. Add teacher-regularized APPO on top of this repaired path.
3. Keep checkpoint selection trace-based, not reward-based.
4. Continue using the deterministic trace benchmark for gating.
5. Revisit recurrence only after the recurrent bridge is made teacher-equivalent.

## Exact Commands Used For Analysis

Trace ranking:

```bash
uv run python cli.py rl-rank-checkpoints \
  --experiment appo_tracefix_v2_bc_nornn_x100 \
  --trace-input data/tracefix_v2_explore_traces.jsonl \
  --top-k 10
```

Explicit checkpoint trace eval:

```bash
uv run python cli.py rl-evaluate-appo \
  --experiment appo_tracefix_v2_bc_nornn_x100 \
  --trace-input data/tracefix_v2_explore_traces.jsonl \
  --checkpoint-path train_dir/rl/appo_tracefix_v2_bc_nornn_x100/checkpoint_p0/checkpoint_000000120_122880.pth
```

BC baseline:

```bash
uv run python cli.py rl-evaluate-bc \
  --model output/tracefix_v2_explore_bc.pt \
  --trace-input data/tracefix_v2_explore_traces.jsonl
```
