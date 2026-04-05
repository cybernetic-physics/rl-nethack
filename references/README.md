# rl-nethack References

Collected papers and repos for NetHack LLM agent research.

## Papers (papers/)

| File | Paper | Year | Key Relevance |
|------|-------|------|---------------|
| `netplay_cog2024.{pdf,txt}` | "Playing NetHack with LLMs: Potential & Limitations as Zero-Shot Agents" (Jeurissen et al., IEEE CoG 2024) | 2024 | **MOST RELEVANT** -- GPT-4 zero-shot NetHack agent with skill system, structured observations, JSON chain-of-thought. Only works with GPT-4. ArXiv: 2403.00690 |
| `nle_neurips2020.{pdf,txt}` | "The NetHack Learning Environment" (Küttler et al., NeurIPS 2020) | 2020 | Foundation paper. Defines NLE, tasks, baselines. TorchBeast agent gets ~120 avg reward over 1B frames. ArXiv: 2006.13760 |
| `nle_language_wrapper_jors2023.txt` | "A NetHack Language Wrapper for Autonomous Agents" (Goodger et al., JORS 2023) | 2023 | Translates NLE observations to rich text (text_glyphs, text_inventory, text_blstats). Sample Factory RL agent gets ~730 reward. DOI: 10.5334/jors.444 |
| `minihack_neurips2021.{pdf,txt}` | "MiniHack the Planet: A Sandbox for Open-Ended RL Research" (Samvelyan et al., NeurIPS 2021) | 2021 | Configurable MiniHack environments for testing specific behaviors. ArXiv: 2109.13202 |
| `bebold_arxiv2020.{pdf,txt}` | "BeBold: Exploration Beyond the Boundary of Explored Regions" (Zhang et al., 2020) | 2020 | Exploration bonus for NetHack -- reward for visiting new states. ArXiv: 2012.08621 |
| `samplefactory_arxiv2020.{pdf,txt}` | "Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS" (Petrenko et al., 2020) | 2020 | APPO algorithm used by the NLE language wrapper agent. ArXiv: 2006.11751v2 |

## Repos (repos/)

| Directory | Repo | Stars | Description |
|-----------|------|-------|-------------|
| `NetPlay/` | [CommanderCero/NetPlay](https://github.com/CommanderCero/NetPlay) | 21 | LLM-powered NetHack agent (GPT-4 only). Skill system + structured obs + JSON CoT. See `netplay/nethack_agent/` for agent loop, `example_prompt_plus_messages.txt` for full prompt. |
| `autoascend/` | [maciej-sypetkowski/autoascend](https://github.com/maciej-sypetkowski/autoascend) | 63 | **NeurIPS 2021 NetHack Challenge 1st place**. Handcrafted strategy system -- explore, combat, inventory, altar farming, sokoban solver. See `autoascend/global_logic.py` for main strategy, `autoascend/strategy.py` for skill framework. |
| `nle-language-wrapper/` | [ngoodger/nle-language-wrapper](https://github.com/ngoodger/nle-language-wrapper) | -- | Language wrapper for NLE. Rich text observations from NLE glyphs. Includes Sample Factory RL agent (~730 reward). See `nle_language_wrapper/` for core code. |
| `nle-agent-platform/` | [stellapie/nle-agent-platform](https://github.com/stellapie/nle-agent-platform) | 0 | Full platform: LLM agent + memory system (episode + note + reflection) + FastAPI + 3D browser viz. See `backend/agent/` for planner/executor/text_observer architecture. |
| `nle/` | [facebookresearch/nle](https://github.com/facebookresearch/nle) | -- | Official NLE repo. NetHack 3.6.6 RL environment. See `nle/agent/` for TorchBeast baseline, `nle/dataset/` for NetHack Learning Dataset. |
| `minihack/` | [facebookresearch/minihack](https://github.com/facebookresearch/minihack) | -- | Configurable MiniHack environments. Des-file format for creating custom levels. |

## Key Numbers for Baselines

- **Random agent**: ~0 reward, dies quickly
- **TorchBeast (NLE baseline)**: ~120 avg reward after 1B frames
- **Sample Factory on language obs**: ~730 reward after 700M frames
- **AutoAscend (handcrafted)**: Wins the game (ascends), NeurIPS 2021 champion
- **NetPlay (GPT-4 zero-shot)**: Can explore, fight, pickup items but cannot win. Only works with GPT-4.
- **Our current agent (Qwen 2.5 3B)**: ~0 reward, walks into walls, 10.6s/step on CPU

## Most Useful Files to Read

1. `repos/NetPlay/example_prompt_plus_messages.txt` -- Full NetPlay prompt showing skill system + structured obs
2. `repos/NetPlay/netplay/nethack_agent/` -- How NetPlay builds the agent loop
3. `repos/autoascend/autoascend/global_logic.py` -- What "good NetHack play" looks like (handcrafted)
4. `papers/netplay_cog2024.txt` -- NetPlay paper (4 pages, quick read)
5. `repos/nle-agent-platform/backend/agent/` -- Memory + planner + executor architecture
