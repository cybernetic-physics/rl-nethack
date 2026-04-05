# TODO

## Immediate Execution Queue

1. Fix train/eval prompt mismatch so evaluation uses the same message structure as training.
2. Add a golden closed-loop replay harness for one tiny deterministic episode.
3. Overfit that golden episode until the model reproduces it without divergence.
4. Upgrade local policy generation from `Qwen/Qwen2.5-0.5B-Instruct` to `Qwen/Qwen2.5-1.5B-Instruct`.
5. Measure action distribution quality before generating a larger corpus.
6. If `1.5B` is still weak, move to `Qwen/Qwen2.5-3B-Instruct`.
7. Once the corpus quality is acceptable, generate `50k-200k` examples on GPUs `0,1`.
8. Then benchmark Qwen 2.5 `3B` vs `7B` forward-model LoRA on all 4 H200s.

## Debugging Principles

- Do not trust aggregate metrics before the single-episode replay test passes.
- Log prompt hashes, prediction text, parsed deltas, and next-state hashes for every replay step.
- Keep random-policy eval, LLM-policy eval, and golden closed-loop replay as separate suites.
