#!/usr/bin/env python3
"""Patch NLE NetHackChallenge to allow seeding."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else '/nle/nle/env/tasks.py'
content = open(path).read()

# Remove seed override that blocks seeding
lines = content.split('\n')
new_lines = []
skip = 0
for i, line in enumerate(lines):
    if skip > 0:
        skip -= 1
        continue
    if 'def seed(self, core=None, disp=None, reseed=True):' in line:
        # Skip this def and the next line (raise RuntimeError)
        skip = 1
        continue
    if "self.env.set_initial_seeds = f" in line:
        continue
    if "self.env.set_current_seeds = f" in line:
        continue
    if "self.env.get_current_seeds = f" in line:
        continue
    new_lines.append(line)

open(path, 'w').write('\n'.join(new_lines))
print(f'Patched {path}: removed {len(lines) - len(new_lines)} lines')
