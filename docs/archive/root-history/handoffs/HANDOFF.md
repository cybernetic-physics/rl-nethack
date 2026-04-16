# rl-nethack Handoff Notes (Apr 5, 2026)

## Current State

NetHack LLM agent exploring memory-augmented forward models for game dynamics prediction.
Train a small model to predict what happens next given accumulated exploration memory.

## Recent Updates (Apr 5, 2026)

- Repo migrated to `uv` with [pyproject.toml](/home/luc/rl-nethack/pyproject.toml) and [uv.lock](/home/luc/rl-nethack/uv.lock)
- Local training path upgraded to real multi-GPU DDP via `torchrun`
- Training validated on this machine with a 2-GPU smoke run using Unsloth LoRA
- Local high-throughput policy generation added via vLLM on GPUs `0,1`
- Generated [data/training_pairs_5k.jsonl](/home/luc/rl-nethack/data/training_pairs_5k.jsonl) and tracked it with Git LFS
- Machine reality check: this host is 4x H200, not 4x H100
- Closed-loop golden replay harness added and validated on a tiny saved episode
- Local policy generator now uses structured prompts, full state history, and frontier-biased action sanitization
- Task-directed closed-loop evaluation harness added for `explore`, `survive`, `combat`, `descend`, and `resource`
- One-step counterfactual control policy added: fork the live env, score candidate actions with shaped task reward, and execute the best action

### What is built
- nle_agent/agent_http.py -- play via local OpenAI-compatible server or OpenRouter
- src/memory_tracker.py -- MemoryTracker, enriched training pair generation
- src/state_encoder.py -- NLE observation to text encoding
- src/reporter.py -- Interactive HTML replay reports
- src/data_generator.py -- Random policy data generation (wall avoidance)
- scripts/generate_training_data.py -- LLM policy data generation (ZAI/OpenRouter/vLLM local)
- scripts/start_vllm_policy_server.sh -- starts local vLLM policy server on GPUs 0,1
- src/closed_loop_debug.py -- golden-episode debug harness for tiny saved trajectories
- references/ -- 6 papers + 6 repos for related work
- Multi-GPU smoke training path validated locally

### Dataset v1 (done)
- 20 games x 50 steps = 1,000 pairs (800 train, 200 eval)
- Policy: ZAI glm-4.5-flash (thinking model, ~500 reasoning tokens/step)
- Cost: $0.22 total (~450K tokens)
- Runtime: ~10 hours sequential, rate-limited
- Files: data/training_pairs.jsonl (1.1MB), data/eval_pairs.jsonl (284KB)
- 15 distinct actions, 20% steps with visible monsters
- Format: ShareGPT conversations with metadata per line

### Dataset v2 throughput run (Apr 5)
- 500 games x 10 steps = 5,000 pairs
- Policy: local vLLM serving `Qwen/Qwen2.5-0.5B-Instruct` on GPUs 0,1
- Runtime: ~30 seconds end-to-end for generation after server warmup
- File: data/training_pairs_5k.jsonl (6.2MB, Git LFS tracked)
- Quality warning: action distribution is poor (`wait` dominates), so this dataset is a throughput baseline and debugging artifact, not final training data

### Dataset v3 policy cleanup experiments (Apr 5)
- Benchmark A: `Qwen/Qwen2.5-1.5B-Instruct` on local vLLM, GPUs 0,1
- Runtime: `1,000` samples in `6.06s`
- Action mix improved substantially over `0.5B`, but still overuses movement and pickup

- Benchmark B: `Qwen/Qwen2.5-3B-Instruct` with naive sanitize fallback
- Runtime: `1,000` samples in `8.05s`
- 10k corpus generated in `78.21s` total
- Files: [data/training_pairs_10k_3b.jsonl](/home/luc/rl-nethack/data/training_pairs_10k_3b.jsonl), [data/eval_pairs_10k_3b.jsonl](/home/luc/rl-nethack/data/eval_pairs_10k_3b.jsonl)
- Quality warning: movement collapse remained severe (`north` dominated), so this corpus should not be scaled further as-is

- Benchmark C: `Qwen/Qwen2.5-3B-Instruct` with frontier-biased fallback + full history
- Runtime: `1,000` samples in `8.75s`
- Action distribution became much healthier:
  - `west 281`, `east 263`, `north 233`, `south 206`, `search 13`
- Average reward improved to `1.451` from the earlier `3B` sample's `1.131`
- Conclusion: frontier-biased sanitization is worth keeping; this is the first local policy run that looks directionally sane

### Dataset v4 serving-topology experiments (Apr 5)
- Added [scripts/start_vllm_policy_replicas.sh](/home/luc/rl-nethack/scripts/start_vllm_policy_replicas.sh) to run two 1-GPU `vLLM` replicas on GPUs `0` and `1`
- Added multi-server fanout support to [scripts/generate_training_data.py](/home/luc/rl-nethack/scripts/generate_training_data.py) via comma-separated `--server-url`

- TP=2 baseline, same improved `3B` generator:
  - `1,000` samples in `9.04s`
  - action mix: `west 280`, `east 263`, `north 225`, `south 204`
  - average reward: `0.675`

- Two 1-GPU replicas, same improved `3B` generator:
  - `1,000` samples in `8.09s`
  - action mix: `west 272`, `east 270`, `north 226`, `south 202`
  - average reward: `0.702`

- Full 10k improved corpus on replicas:
  - files: [data/training_pairs_10k_3b_replicas.jsonl](/home/luc/rl-nethack/data/training_pairs_10k_3b_replicas.jsonl), [data/eval_pairs_10k_3b_replicas.jsonl](/home/luc/rl-nethack/data/eval_pairs_10k_3b_replicas.jsonl)
  - runtime: `78.76s`
  - train actions: `west 2222`, `east 2135`, `north 1795`, `south 1606`, `search 162`
  - eval actions: `west 552`, `east 534`, `north 449`, `south 407`, `search 43`
  - conclusion: the balanced movement mix survived at scale, so this path is good enough to keep scaling

### Dataset v5 in-process batching experiment (Apr 5)
- Added experimental `vllm-batch` backend to [scripts/generate_training_data.py](/home/luc/rl-nethack/scripts/generate_training_data.py)
- This backend removes HTTP serving entirely and does one in-process `LLM.chat(...)` call per rollout step over all active games
- Result on current settings (`tp=2`, `workers=64`):
  - `1,000` samples in `43.30s` end-to-end
  - `10,000` samples in `111.20s` end-to-end to [data/training_pairs_10k_3b_vllmbatch.jsonl](/home/luc/rl-nethack/data/training_pairs_10k_3b_vllmbatch.jsonl) and [data/eval_pairs_10k_3b_vllmbatch.jsonl](/home/luc/rl-nethack/data/eval_pairs_10k_3b_vllmbatch.jsonl)
  - 1k action mix was healthy: `west 612`, `east 590`, `north 419`, `south 346`
- Conclusion: batching works functionally, but the current in-process implementation is slower than the 2-replica HTTP path once engine startup is included. Keep it as an experiment, not the default.

### Key Insight: Memory-Dependent Forward Model

Nobody trains forward model for NetHack. Others do:
- NetPlay: GPT-4 zero-shot with skill system (no training)
- NLE + Sample Factory: RL policy, raw actions
- AutoAscend: hand-coded symbolic AI (wins)

Our approach: predict game dynamics from accumulated exploration history.
Predictions requiring MEMORY are non-trivial and teach game dynamics.

### Problems fixed this session
- memory_tracker.py: last_seen vs last_seen_turn key mismatch (fixed)
- Python urllib hangs on ZAI after rate limiting -- use subprocess curl
- glm-4.5-flash thinking model: needs max_tokens=1024, extract action from reasoning_content fallback
- train.py: LoRA target module selection and DDP cleanup fixed during local smoke validation
- scripts/generate_training_data.py: history context now stores the full prior policy state, not just the last line
- scripts/generate_training_data.py: fallback exploration no longer hard-codes north/east/west/south order; it now prefers low-visit frontier tiles and avoids immediate backtracking

### Closed-Loop Debugging Notes (Apr 5)

Important lesson from Eric Gu's Melee training writeup: overfit a single synthetic example or tiny scripted episode until the model reproduces it perfectly in closed loop, then use that harness to debug preprocessing, prompting, parsing, and evaluation mismatches.

Applied here:
- We do not yet have a real RL loop; the repo is still mostly behavior cloning / forward-model training plus evaluation.
- The right debugging target is therefore a single deterministic NetHack trajectory, not PPO or self-play.
- Train/eval distribution mismatch already exists: training examples include the system prompt, but evaluator requests currently send only the user prompt.
- Before scaling data or model size further, we should be able to overfit one tiny episode and replay it step-by-step without divergence.

### Task Harness Results (Apr 5)

New code:
- [src/task_rewards.py](/home/luc/rl-nethack/src/task_rewards.py) defines repo-owned shaped rewards for `explore`, `survive`, `combat`, `descend`, and `resource`
- [src/task_harness.py](/home/luc/rl-nethack/src/task_harness.py) runs real episodes, tracks trajectory metrics, and implements a one-step counterfactual controller
- [cli.py](/home/luc/rl-nethack/cli.py) now exposes `task-evaluate`

Explore benchmark, seeds `42,43,44`, `10` steps:
- baseline `wall_avoidance`: avg task reward `-1.17`, avg unique tiles `36.33`, repeated action rate `20.0%`
- task-directed `task_greedy`: avg task reward `7.27`, avg unique tiles `93.00`, repeated action rate `6.7%`
- conclusion: shaped exploration reward is good enough to improve closed-loop behavior right now

Survival benchmark, seeds `42,43,44`, `15` steps:
- baseline `wall_avoidance`: avg task reward `-1.63`, avg unique tiles `65.67`, repeated action rate `17.8%`
- task-directed `task_greedy`: avg task reward `-0.37`, avg unique tiles `67.00`, repeated action rate `0.0%`
- conclusion: loop penalties were required; after adding them, the controller is more stable than baseline, but survival shaping is still weak and not yet a strong policy objective

Interpretation:
- This is the repo's first real closed-loop task-control harness
- It is not PPO and not learned RL yet
- It is a useful control/evaluation loop that closes the gap between forward-model work and future RL/planning
- The current controller is one-step greedy, so deeper planning or learned value estimation is the obvious next improvement

## Next Steps

### 1. Turn the task harness into learned control
- Reuse `scripts/generate_counterfactual_data.py` and the new shaped rewards to label one-step actions by task value
- Train a reward/value model that predicts shaped task return for candidate actions
- Replace the expensive fork-per-action planner with a learned scorer once the labels are reliable

### 2. Add task-specific eval suites
- Keep `explore` and `survive` as always-on regression tasks
- Add curated short suites for combat, stairs/descent, and pickup/resource moments
- Report trajectory metrics, not just one-step delta accuracy

### 3. Generate a larger local corpus
- The balanced `3B` path now holds up at 10k scale, so target `50k-200k` examples on GPUs 0,1
- Keep train/eval split by seed
- Combine policy data with counterfactual and AutoAscend-derived data where possible

### 4. Scale training to the full machine
- Use all 4 H200s for LoRA training via `torchrun`
- Start with Qwen 2.5 3B or 7B for the forward model
- Increase sequence length and effective batch once the dataset is no longer tiny

### 5. Evaluate and plan
- Compare predictions vs actual on held-out seeds
- Add metrics for action-conditioned deltas, combat outcomes, and exploration gains
- Once the forward model is competent, use it for look-ahead planning

### 6. Build a real debug harness before more training work
- Create a golden single-episode dataset, around 10-20 steps, with saved prompt, action, target delta, and next-state hash for every step
- Train on only that episode until training loss is near zero
- Add a closed-loop replay script that runs the model on that exact start state and compares prompt hash, predicted delta, and next-state hash step-by-step
- Do not trust larger runs until the golden replay stays aligned for the full episode

### 7. Fix distribution mismatches in the current loop
- Make evaluation use the same message structure as training, including the system prompt
- Keep separate eval suites for random-policy data, LLM-policy data, and golden closed-loop replay
- Log raw observation hashes, formatted prompt hashes, parsed predictions, and next-state hashes so mismatches are obvious

### 8. Improve the future RL / planning loop in the right order
- First make the forward model accurate on short deterministic trajectories
- Then test one-step planning against counterfactual rollouts from the same saved state
- Only after that should we build deeper look-ahead or policy-improvement loops
- If a planner is added later, debug it first on tiny synthetic scenarios with known optimal actions, not full random NetHack

## Data Generator Usage

  # ZAI (default)
  python scripts/generate_training_data.py --api-key KEY --model glm-4.5-flash --num-games 20 --max-steps 50 --cooldown 45

  # OpenRouter
  python scripts/generate_training_data.py --api-key KEY --model openai/gpt-4o-mini --num-games 100 --max-steps 50

  # Local vLLM / OpenAI-compatible server
  python scripts/generate_training_data.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct --server-url http://127.0.0.1:8000/v1 --num-games 10 --max-steps 20 --workers 64 --cooldown 0

  # Dry run
  python scripts/generate_training_data.py --dry-run --api-key KEY --model glm-4.5-flash

## Technical Notes

NLE observation keys:
- chars: 21x79 ASCII map (uint8)
- blstats: [0]=x, [1]=y, [10]=hp, [11]=hp_max, [13]=gold, [12]=depth, [20]=turn
- message: 256-byte game message
- screen_descriptions: 21x79x80 per-tile text

ZAI API:
- Base: https://api.z.ai/api/paas/v4
- Keys: GLM_API_KEY, ZAI_API_KEY, Z_AI_API_KEY
- glm-4.5-flash: thinking model, ~500 reasoning tokens, needs max_tokens=1024
- Rate limit: ~40 sustained then cooldown, use --cooldown 45 between games
- Use subprocess curl (python urllib hangs after rate limiting)
- glm-4-plus needs paid package (429 insufficient balance)

Local vLLM policy serving:
- Use GPUs 0,1 for serving and leave 2,3 available for training or other work
- `Qwen2.5-0.5B-Instruct` is fast enough for ~5k pairs in ~30s, but its policy quality is weak
- Next local policy candidate should be `Qwen2.5-1.5B-Instruct` or `Qwen2.5-3B-Instruct`

Training / evaluation loop notes:
- Current training format is ShareGPT-style with system + user + assistant messages
- Current evaluator now uses the same message structure as training; keep it that way
- The most valuable new test is not another aggregate metric; it is a single-example closed-loop replay test that can fail on the exact divergent step
## Ideas for Better Data (Apr 4 notes)

### Counterfactual Data Generator
From each saved game state, try ALL valid actions (not just the one the LLM picked).
One step = ~15 training pairs instead of 1. Gives negative examples (walking into walls, bumping monsters).
1000 existing pairs could become ~15,000.

### Full-Map Oracle Prompts
At training data generation time, we have the full NLE ground truth (entire dungeon, not just viewport).
Use this to create prompts like "there is a goblin 5 tiles north in an unexplored room" -- teach the model to reason about what it can't see.

### Targeted Scenarios
Use NLE to set up specific situations:
- Agent facing a monster (combat training)
- Agent near stairs (descent prediction)
- Agent at low HP (survival prediction)
- Agent with items in inventory (use/consume prediction)

Dense targeted data instead of waiting for random encounters.

### Expert Traces (AutoAscend)
AutoAscend is a hand-coded symbolic AI that wins NetHack (ascends). If we can record its games and convert to enriched pairs, that's gold-standard training data for real gameplay.

### Maestro Paper (ICLR 2025)
Talk: https://www.youtube.com/watch?v=6b25qXmitaQ
Approach: decompose gameplay into skills (explore, fight, collect, navigate), use LLM for high-level planning, RL for low-level control.
Their "AI feedback" idea: use LLM to judge which skill execution is better -- cheap offline training signal.
Relevant to our forward model: predict outcomes per skill, then plan over skills.

## Counterfactual Data Generator Progress (Apr 4)

### What we proved works
1. os.fork() clones NLE state perfectly -- parent continues, child tries alt action
2. From 10 random games x 100 steps: found 50 interesting moments
3. 9 actions per moment = 450 counterfactual pairs in seconds (no API calls)
4. Real combat data: 42 HP changes, 57 kills, 7 deaths
5. Skills detected: combat (261), threat (162), survival (54)

### Key seeds with adjacent hostile monsters at spawn
Seed 37: jackal, Seed 58: goblin, Seed 127: lichen, Seed 192: sewer rat
Seed 431: grid bug, Seed 481: kobold, Seed 495: lichen
(Found 25+ seeds out of 500 scanned)

### Fork pattern (works):
    pid = os.fork()
    if pid == 0:  # child
        obs_after, reward, term, trunc, _ = env.step(amap[action_name])
        # write result to /tmp/cf_*.json
        env.close()
        os._exit(0)
    # parent continues after waitpid

### NLE seed gotchas
- env.reset(seed=N) does NOT produce deterministic dungeons
- set_initial_seeds() also doesn't work
- Each reset generates a random dungeon regardless
- Fork is the only way to get counterfactuals from same state

### What still needs building
The full generate_counterfactual_data.py script. The approach is validated but
the script file got corrupted during writing. Next session should:
1. Write clean script using the fork pattern above
2. Run 100+ games with aggressive exploration policy
3. Output skill-tagged ShareGPT pairs to data/counterfactual_training_pairs.jsonl

## Script Complete (Apr 4, session 2)

### Script: scripts/generate_counterfactual_data.py
- Uses os.fork() to try all 9 movement actions from interesting states
- Detects skills: combat, threat, survival, descent, pickup
- Outputs ShareGPT-format JSONL with metadata
- Usage: python scripts/generate_counterfactual_data.py --num-games 50 --max-steps 200

### Data generated: data/counterfactual_training_pairs.jsonl
- 4,005 pairs from 50 games x 200 steps
- 1,044 combat pairs, 207 kills, 143 HP changes, 4 deaths
- Skills: threat(219), combat(116), survival(174)
- 2.5 MB, balanced across 9 actions

### Next steps
1. Scale up: 500+ games for full training set (~40K pairs)
2. LoRA fine-tune Qwen 2.5 3B on this data
3. Evaluate: does the fine-tuned model predict combat outcomes better?
4. Build virgin benchmark for evaluation

## Demo Viewer Built (Apr 4, session 2)

### output/game_viewer.html (487 KB, self-contained)
- 5 interesting games with counterfactual "what-if" branches
- Games: Seed 10 (6 kills), Seed 13 (near death!), Seed 23 (3 kills), Seed 25 (5 kills), Seed 29 (3 kills)
- Features: step-by-step replay, HP graph with kill/damage markers, combat counterfactuals, keyboard nav
- Keyboard: arrows/n/p = step, space = play/pause, c = skip to combat, 1-5 = switch games
- Serve: python3 -m http.server 8080 -d output/

### output/demo_games.json (500 KB)
- Raw game data for the 5 games, with full step-by-step state and counterfactuals
- Counterfactuals show what would have happened with 5 different actions at combat moments

### Key insight for demos
- NLE seed non-determinism: env.reset(seed=N) doesn't give same dungeon across runs
- Must record game data in one pass (scan + record simultaneously)
- The viewer embeds data directly, no server needed
