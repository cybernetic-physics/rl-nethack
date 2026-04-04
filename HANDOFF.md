# dstack-lora Handoff Notes (Apr 4, 2026)

## Current State

NetHack LLM agent exploring memory-augmented forward models for game dynamics prediction.
Train a small model to predict what happens next given accumulated exploration memory.

### What is built
- nle_agent/agent_http.py -- play via local llama-server or OpenRouter
- src/memory_tracker.py -- MemoryTracker, enriched training pair generation
- src/state_encoder.py -- NLE observation to text encoding
- src/reporter.py -- Interactive HTML replay reports
- src/data_generator.py -- Random policy data generation (wall avoidance)
- scripts/generate_training_data.py -- LLM policy data generation (ZAI/OpenRouter/local)
- references/ -- 6 papers + 6 repos for related work
- 310 tests passing

### Dataset v1 (done)
- 20 games x 50 steps = 1,000 pairs (800 train, 200 eval)
- Policy: ZAI glm-4.5-flash (thinking model, ~500 reasoning tokens/step)
- Cost: $0.22 total (~450K tokens)
- Runtime: ~10 hours sequential, rate-limited
- Files: data/training_pairs.jsonl (1.1MB), data/eval_pairs.jsonl (284KB)
- 15 distinct actions, 20% steps with visible monsters
- Format: ShareGPT conversations with metadata per line

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
- patch/write_file tools corrupt files -- use python3 heredoc str.replace()

## Next Steps

### 1. LoRA fine-tune Qwen 2.5 3B
- Use Unsloth for fast fine-tuning
- Data is already sharegpt format, minimal processing needed
- Train on 800 enriched pairs, eval on 200 held-out
- Local llama-server on CPU port 8765 for inference
- 800 pairs may be thin -- might need to generate more with stronger policy
- Forward model (predict delta) is simpler than playing, so small data might work for proof of concept

### 2. Evaluate forward model
- Compare predictions vs actual on held-out seeds
- Metrics: exact match on survival, MSE on hp_delta, accuracy on pos_delta
- Qualitative: does model predict goblin attacks from memory?
- Virgin benchmark: TEE-sealed seeds prove no test contamination

### 3. Use forward model for planning
- If model predicts outcomes, use look-ahead search
- Try multiple action sequences, pick best predicted outcome
- Compare agent performance with vs without planning

### 4. Generate more/better data
- Use glm-5.1 or gpt-4o-mini for better gameplay
- Add gameplay hints to system prompt (e.g. fight monsters, go downstairs)
- Mix in expert traces (AutoAscend replays) for combat/examples
- Increase to 100+ games once pipeline validated

## Data Generator Usage

  # ZAI (default)
  python scripts/generate_training_data.py --api-key KEY --model glm-4.5-flash --num-games 20 --max-steps 50 --cooldown 45

  # OpenRouter
  python scripts/generate_training_data.py --api-key KEY --model openai/gpt-4o-mini --num-games 100 --max-steps 50

  # Local llama-server
  python scripts/generate_training_data.py --num-games 10 --max-steps 20

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

File editing: patch/write_file can corrupt files. Use python3 heredoc with str.replace()
Verify: python3 -c "import ast; ast.parse(open(file).read())"
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
