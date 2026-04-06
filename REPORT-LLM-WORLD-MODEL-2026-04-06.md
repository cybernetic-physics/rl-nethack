# LLM-Backed World Model Report

## Verdict

The idea is worth promoting in a **constrained** form:

- use a **frozen pretrained text encoder** over the existing trace `prompt`
- fuse that text signal into the current numeric world model
- train only the world-model heads and fusion layers

The idea is **not** worth promoting in the naive form:

- do **not** replace the current world model with a fully fine-tuned LLM
- do **not** drive RL directly by fine-tuning the language model on reward

The literature and the repo now agree on that distinction.

## Why This Was Promoted

The recent text-game / world-model literature points in the same direction:

- [PLM-based World Models for Text-based Games](https://aclanthology.org/2022.emnlp-main.86.pdf)
  argues that pretrained language models are a strong base for text-game world models and improve future valid action prediction and graph-change prediction.
- [On the Effects of Fine-tuning Language Models for Text-Based Reinforcement Learning](https://aclanthology.org/2025.coling-main.445.pdf)
  shows that fine-tuning language models directly to RL reward causes semantic degeneration; fixed pretrained LMs train faster and generalize better.
- [Accelerating exploration and representation learning with offline pre-training](https://arxiv.org/abs/2304.00046)
  shows that offline representation learning and auxiliary reward modeling improve sample efficiency on NetHack.
- [TEXT2WORLD](https://aclanthology.org/2025.findings-acl.1337.pdf)
  shows that even strong LLMs still have limited world-modeling capability in structured domains.
- [WorldLLM](https://arxiv.org/abs/2506.06725)
  is promising, but it is a much heavier active-exploration / hypothesis-refinement framework than this repo currently needs.

So the promoted idea was:

- keep the current structured numeric world model
- add a frozen pretrained text backbone
- evaluate it with the existing direct metrics and downstream BC probe

That is the safest version that is both literature-backed and implementable in this repo.

## Code Changes

Implemented optional text-conditioned world-model support in:

- [rl/world_model.py](/home/luc/rl-nethack/rl/world_model.py)
- [rl/world_model_dataset.py](/home/luc/rl-nethack/rl/world_model_dataset.py)
- [rl/train_world_model.py](/home/luc/rl-nethack/rl/train_world_model.py)
- [rl/world_model_eval.py](/home/luc/rl-nethack/rl/world_model_eval.py)
- [rl/world_model_features.py](/home/luc/rl-nethack/rl/world_model_features.py)
- [cli.py](/home/luc/rl-nethack/cli.py)
- [tests/test_rl_scaffold.py](/home/luc/rl-nethack/tests/test_rl_scaffold.py)

### Implementation shape

- Added `prompt` to world-model examples/arrays.
- Added optional `text_encoder_backend` with:
  - `none`
  - `hash`
  - `transformer`
- Added a frozen Hugging Face transformer path.
- Fused text context with the numeric feature stem before latent prediction.
- Kept old numeric-only checkpoints loadable.
- Passed prompt text through trace transforms and eval.

### Safety choices

- transformer is frozen by default
- only fusion / prediction heads are trained
- old numeric-only path stays the default
- the no-download `hash` backend covers tests

## Validation

Scaffold tests still pass:

- `uv run pytest -q tests/test_rl_scaffold.py`
- result: `55 passed`

One real harness issue was exposed during RL validation:

- APPO with `CUDA_VISIBLE_DEVICES=''` still came up as `device=gpu`
- that path died with `list index out of range`
- rerunning on a real GPU avoided the issue

This is a harness issue, not a world-model-quality issue.

## Experiments

## 1. Train text-conditioned world model

Model:

- backbone: `prajjwal1/bert-tiny`
- backend: `transformer`
- frozen text encoder
- horizon: `4`
- hidden size: `256`
- latent dim: `128`

Output:

- `/tmp/x100_v4_world_model_textbert_h4.pt`

Train summary:

- `text_encoder_backend = transformer`
- `text_model_name = prajjwal1/bert-tiny`

## 2. Held-out world-model eval

Held-out eval on `/tmp/x100_v4_heldout_traces.jsonl`:

- feature cosine mean: `0.6791`
- action accuracy: `0.5625`
- action top-3 accuracy: `0.8750`
- latent dead fraction: `0.5391`

This is not a great pure predictor.

Important comparison:

- the older numeric-only baseline had worse direct action accuracy (`0.25`)
- but direct prediction quality still does **not** fully explain downstream usefulness

That matches the lesson from the earlier world-model work in this repo.

## 3. Downstream BC probe

Using the existing world-model eval’s downstream BC probe:

- `concat` mode reached `0.95`
- `concat_aux` mode also reached `0.95`

This is already stronger than the earlier world-model branches, which topped out at `0.9375`.

## 4. Real teacher checkpoint from transformed traces

I then trained a real BC teacher on transformed `concat` traces:

- train trace: `/tmp/x100_v4_train_textbert_h4_concat.jsonl`
- held-out trace: `/tmp/x100_v4_heldout_textbert_h4_concat.jsonl`
- teacher checkpoint: `/tmp/x100_v4_textbert_h4_concat_bc.pt`
- teacher report: `/tmp/x100_v4_textbert_h4_concat_bc.pt.teacher_report.json`

Held-out result:

- `match_rate = 0.9625`

This is the first LLM-backed world-model teacher in this repo that clearly beat the previous `0.95` line.

## 5. Short RL bridge

Short APPO run:

- experiment: [appo_textbert_wm_concat_short_b](/home/luc/rl-nethack/train_dir/rl/appo_textbert_wm_concat_short_b)

Best trace-ranked checkpoint:

- [best_trace_match.json](/home/luc/rl-nethack/train_dir/rl/appo_textbert_wm_concat_short_b/checkpoint_p0/best_trace_match.json)
- `match_rate = 0.9625`

Interpretation:

- the stronger teacher survived the short online bridge
- unlike many earlier branches, it did **not** immediately collapse below the teacher

This is a meaningful positive result.

## 6. Medium RL validation

Medium APPO run:

- experiment: [appo_textbert_wm_concat_medium_a](/home/luc/rl-nethack/train_dir/rl/appo_textbert_wm_concat_medium_a)

Best trace-ranked checkpoint:

- [best_trace_match.json](/home/luc/rl-nethack/train_dir/rl/appo_textbert_wm_concat_medium_a/checkpoint_p0/best_trace_match.json)
- best checkpoint: [checkpoint_000000052_6656.pth](/home/luc/rl-nethack/train_dir/rl/appo_textbert_wm_concat_medium_a/checkpoint_p0/checkpoint_000000052_6656.pth)
- `match_rate = 0.9375`

Interpretation:

- the stronger offline teacher did **not** translate into a stronger medium online branch
- the same online drift problem is still present

So the LLM-backed world model improved the teacher, but did not fix the improver.

## What We Learned

1. A frozen pretrained text encoder is useful in this repo.
   It improved the world-model branch where it matters most: downstream teacher quality.

2. Direct world-model metrics are still not enough.
   The text-conditioned model was only moderate on direct action prediction, yet it produced the best downstream BC teacher.

3. The world model is a stronger **teacher-builder** than an immediate RL-fixer.
   Offline teacher quality improved from `0.9375` to `0.9625`.

4. Online RL drift is still the bottleneck.
   The medium APPO continuation dropped back to `0.9375`.

5. The literature warning was correct.
   Using pretrained language semantics as a frozen base is helpful.
   Letting RL fine-tune the language model itself would likely be the wrong next move.

## Recommendation

Promote this idea in the repo as:

- **text-conditioned world model**
- **frozen pretrained text encoder**
- **teacher construction / representation path**

Do **not** promote:

- full LLM fine-tuning in the RL loop
- replacing the current numeric world model with pure text generation

## Next Best Moves

1. Keep the text-conditioned world-model path.
   It is now the strongest offline world-model branch.

2. Use it to build better teachers first.
   That is where it clearly paid off.

3. Do not interpret the medium APPO regression as a world-model failure.
   It is more evidence that the improver is still the weak link.

4. If we keep iterating on the world model, optimize for downstream BC / teacher quality, not just direct prediction metrics.

5. If we want the next real breakthrough, pair this stronger text-conditioned teacher with a better online improver than the current APPO family.
