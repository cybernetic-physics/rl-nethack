#!/usr/bin/env python3
"""Quick smoke test of the reporter."""
from src.state_encoder import StateEncoder
from src.reporter import run_and_report, format_replay, format_summary

enc = StateEncoder()
report = run_and_report(seed=42, max_steps=15, encoder=enc)

print("=== SUMMARY ===")
print(format_summary(report["step_data"], seed=42))
print()
print("=== REPLAY ===")
print(format_replay(report["step_data"], seed=42))
print()
print(f"Outcome: {report['outcome']}")
print(f"Steps: {len(report['step_data'])}")
print(f"Gold: {report['total_gold']}")
print(f"Tiles explored: {report['tiles_explored']}")
print(f"Died: {report['outcome'] == 'died'}")
