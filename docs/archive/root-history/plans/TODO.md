# TODO

## Immediate Execution Queue

1. Use counterfactual rollouts plus [src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py) to generate task-value labels for `explore` and `survive`.
2. Train a small reward/value model to replace the current fork-per-action one-step planner.
3. Overfit the golden closed-loop episode until the model reproduces it without divergence.
4. Train a forward-model LoRA on the improved local corpus using all 4 H200s.
5. Benchmark Qwen 2.5 `3B` vs `7B` forward-model LoRA on all 4 H200s.
6. Keep scaling the improved replica-backed `Qwen/Qwen2.5-3B-Instruct` generator toward `50k-200k`.

## Debugging Principles

- Do not trust aggregate metrics before the single-episode replay test passes.
- Log prompt hashes, prediction text, parsed deltas, and next-state hashes for every replay step.
- Keep random-policy eval, LLM-policy eval, and golden closed-loop replay as separate suites.
- Keep task-level trajectory eval (`task-evaluate`) as a separate regression layer; field accuracy alone is not enough.
