# TODO

## Immediate Execution Queue

1. Add a batched/offline generation path so the policy loop is not one HTTP request per env step.
2. Scale the improved replica-backed `Qwen/Qwen2.5-3B-Instruct` generator from `10k` to `50k-200k`.
3. Overfit the golden closed-loop episode until the model reproduces it without divergence.
4. Train a forward-model LoRA on the improved local corpus using all 4 H200s.
5. Benchmark Qwen 2.5 `3B` vs `7B` forward-model LoRA on all 4 H200s.

## Debugging Principles

- Do not trust aggregate metrics before the single-episode replay test passes.
- Log prompt hashes, prediction text, parsed deltas, and next-state hashes for every replay step.
- Keep random-policy eval, LLM-policy eval, and golden closed-loop replay as separate suites.
