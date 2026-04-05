# TODO

## Immediate Execution Queue

1. Scale the improved replica-backed `Qwen/Qwen2.5-3B-Instruct` generator from `10k` to `50k-200k`.
2. Overfit the golden closed-loop episode until the model reproduces it without divergence.
3. Train a forward-model LoRA on the improved local corpus using all 4 H200s.
4. Benchmark Qwen 2.5 `3B` vs `7B` forward-model LoRA on all 4 H200s.
5. Revisit in-process batching only if we can avoid the current TP=2 startup/throughput penalty.

## Debugging Principles

- Do not trust aggregate metrics before the single-episode replay test passes.
- Log prompt hashes, prediction text, parsed deltas, and next-state hashes for every replay step.
- Keep random-policy eval, LLM-policy eval, and golden closed-loop replay as separate suites.
