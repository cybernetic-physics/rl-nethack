#!/usr/bin/env python3
"""
Run AutoAscend agent and record game traces for the game viewer.
Patches agent.step() to capture observations at each step.

Usage: python run_with_trace.py --episodes 5 --output /output/autoascend_traces.json
"""
import argparse
import os
import traceback

import gym
import nle.nethack as nh
import numpy as np

from autoascend import agent as agent_lib
from autoascend.env_wrapper import EnvWrapper
from trace_recorder import TraceRecorder

from autoascend.exceptions import AgentFinished


def run_game(seed, output_path, max_steps=5000):
    """Run a single AutoAscend game with trace recording."""
    print(f'[trace] Starting game seed={seed}')
    
    env = gym.make('NetHackChallenge-v0')
    env.seed(seed)
    
    wrapper = EnvWrapper(
        env,
        step_limit=max_steps,
        agent_args=dict(panic_on_errors=False, verbose=True),
        interactive=False,
    )
    
    # Create recorder
    recorder = TraceRecorder(output_path, seed)
    
    # Monkey-patch agent.step to record each step
    original_step = None
    prev_hp = [None]  # use list for mutability in closure
    step_count = [0]
    
    def patched_step(self_agent, action, *args, **kwargs):
        # Record state BEFORE action
        obs = self_agent._observation if hasattr(self_agent, '_observation') else wrapper.last_observation
        if obs is not None:
            blstats = obs['blstats']
            current_hp = int(blstats[10])
            action_idx = action if isinstance(action, int) else 0
            recorder.record_step(obs, action_idx, prev_hp[0])
            prev_hp[0] = current_hp
            step_count[0] += 1
            if step_count[0] % 500 == 0:
                depth = int(blstats[12])
                print(f'[trace] seed={seed} step={step_count[0]} hp={current_hp} depth={depth}')
        
        return original_step(self_agent, action, *args, **kwargs)
    
    # Apply patch
    original_step = agent_lib.Agent.step
    agent_lib.Agent.step = patched_step
    
    try:
        wrapper.reset()
        wrapper._init_agent()
        wrapper.agent.main()
    except AgentFinished:
        print(f'[trace] seed={seed} agent finished normally at step {step_count[0]}')
    except Exception as e:
        ename = type(e).__name__
        print(f'[trace] seed={seed} {ename} at step {step_count[0]}: {e}')
        traceback.print_exc()
    finally:
        # Restore original
        agent_lib.Agent.step = original_step
    
    # Determine end reason
    died = True
    end_reason = 'died'
    if hasattr(wrapper, 'last_observation') and wrapper.last_observation is not None:
        try:
            tty = bytes(wrapper.last_observation['tty_chars'].reshape(-1)).decode('ascii', errors='replace')
            if 'ascended' in tty.lower() or 'defeated' in tty.lower():
                end_reason = tty[:200]
                died = 'killed' not in tty.lower()
        except:
            pass
    
    game_data = recorder.save(died=died, end_reason=end_reason)
    env.close()
    
    print(f'[trace] seed={seed} done: {step_count[0]} steps, '
          f'kills={recorder.kills}, depth={recorder.max_depth}, died={died}')
    return game_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='/output/autoascend_traces.json')
    parser.add_argument('--max-steps', type=int, default=5000)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    games = []
    for i in range(args.episodes):
        seed = args.seed + i
        try:
            game = run_game(seed, args.output, max_steps=args.max_steps)
            games.append(game)
        except Exception as e:
            print(f'[trace] FAILED seed={seed}: {e}')
            traceback.print_exc()
    
    print(f'\n[trace] All done. {len(games)}/{args.episodes} games recorded.')
    print(f'[trace] Output: {args.output}')


if __name__ == '__main__':
    main()
