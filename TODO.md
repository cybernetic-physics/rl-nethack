# TODO

## Immediate Execution Queue

1. Re-run `Qwen/Qwen2.5-3B-Instruct` at `10k-50k` scale using the frontier-biased fallback and verify the action distribution stays balanced.
2. Change local serving on GPUs `0,1` from one TP=2 `vLLM` instance to two 1-GPU replicas and compare throughput.
3. Add a batched/offline generation path so the policy loop is not one HTTP request per env step.
4. Overfit the golden closed-loop episode until the model reproduces it without divergence.
5. Generate `50k-200k` examples only after the action audit still looks sane at larger scale.
6. Benchmark Qwen 2.5 `3B` vs `7B` forward-model LoRA on all 4 H200s.

## Debugging Principles

- Do not trust aggregate metrics before the single-episode replay test passes.
- Log prompt hashes, prediction text, parsed deltas, and next-state hashes for every replay step.
- Keep random-policy eval, LLM-policy eval, and golden closed-loop replay as separate suites.
