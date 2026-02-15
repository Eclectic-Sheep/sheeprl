#!/usr/bin/env python3
"""Analyze a MuZero training run from TensorBoard logs.

Usage:
    python scripts/diagnostics/analyze_training.py <log_dir>

Example:
    python scripts/diagnostics/analyze_training.py \
        logs/runs/muzero/2026-02-14_13-06-45/LunarLander-v2_default_42/version_0
"""
import math
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def analyze(log_dir: str):
    ea = EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"No scalar data found in {log_dir}")
        sys.exit(1)

    print(f"=== MuZero Training Diagnostics ===\n")

    # --- Metric summary ---
    print("Metric summary (first → last):")
    for tag in sorted(tags):
        events = ea.Scalars(tag)
        if events:
            first, last = events[0], events[-1]
            arrow = "↑" if last.value > first.value else "↓" if last.value < first.value else "→"
            print(f"  {tag:<35s} {first.value:>10.3f} {arrow} {last.value:>10.3f}  ({len(events)} pts, step {last.step})")

    # --- Detect number of actions from entropy ---
    entropy_events = ea.Scalars("Info/policy_entropy") if "Info/policy_entropy" in tags else []
    if entropy_events:
        max_ent = entropy_events[0].value
        # Entropy is summed over chunk_sequence_len steps
        # Try to detect num_actions: if summed entropy ≈ K * ln(N), find N
        for chunk_len in [10, 5, 20, 1]:
            per_step = max_ent / chunk_len
            num_actions = round(math.exp(per_step))
            if abs(per_step - math.log(num_actions)) < 0.05:
                break
        print(f"\n  Detected: {num_actions} actions, chunk_sequence_len={chunk_len}")
        print(f"  Max entropy (uniform policy): {chunk_len * math.log(num_actions):.3f}")

    # --- Health checks ---
    print("\n=== Health Checks ===\n")
    issues = []

    # 1. Gradient clipping saturation
    grad_events = ea.Scalars("Gradient/gradient_norm") if "Gradient/gradient_norm" in tags else []
    if grad_events:
        clipped_ratio = sum(1 for e in grad_events if abs(e.value - 1.0) < 0.01) / len(grad_events)
        status = "PROBLEM" if clipped_ratio > 0.9 else "OK" if clipped_ratio < 0.5 else "WARNING"
        print(f"  [{status}] Gradient clipping: {clipped_ratio:.0%} of steps clipped at max_norm")
        if clipped_ratio > 0.9:
            issues.append("Gradient always clipped → increase max_grad_norm (try 5.0 or 10.0)")

    # 2. Policy entropy near maximum (not learning)
    if entropy_events:
        max_theoretical = chunk_len * math.log(num_actions)
        last_ent = entropy_events[-1].value
        ratio = last_ent / max_theoretical
        status = "PROBLEM" if ratio > 0.95 else "OK" if ratio < 0.7 else "WARNING"
        print(f"  [{status}] Policy entropy: {last_ent:.2f} / {max_theoretical:.2f} ({ratio:.0%} of max)")
        if ratio > 0.95:
            issues.append("Policy near-uniform → MCTS targets may be uniform or policy can't learn")

    # 3. KL divergence trend (should decrease, not increase)
    kl_events = ea.Scalars("Info/kl_div") if "Info/kl_div" in tags else []
    if len(kl_events) > 10:
        first_kl = sum(e.value for e in kl_events[:5]) / 5
        last_kl = sum(e.value for e in kl_events[-5:]) / 5
        status = "PROBLEM" if last_kl > first_kl * 2 else "OK" if last_kl < first_kl else "WARNING"
        print(f"  [{status}] KL divergence trend: {first_kl:.4f} → {last_kl:.4f}")
        if last_kl > first_kl * 2:
            issues.append("KL increasing → policy diverging from MCTS targets (gradient starvation?)")

    # 4. Value loss decreasing
    vl_events = ea.Scalars("Loss/value_loss") if "Loss/value_loss" in tags else []
    if len(vl_events) > 10:
        first_vl = sum(e.value for e in vl_events[:5]) / 5
        last_vl = sum(e.value for e in vl_events[-5:]) / 5
        ratio = last_vl / first_vl if first_vl > 0 else 1.0
        status = "OK" if ratio < 0.7 else "WARNING" if ratio < 0.95 else "PROBLEM"
        print(f"  [{status}] Value loss: {first_vl:.2f} → {last_vl:.2f} ({ratio:.0%} of initial)")

    # 5. Reward loss decreasing
    rl_events = ea.Scalars("Loss/reward_loss") if "Loss/reward_loss" in tags else []
    if len(rl_events) > 10:
        first_rl = sum(e.value for e in rl_events[:5]) / 5
        last_rl = sum(e.value for e in rl_events[-5:]) / 5
        ratio = last_rl / first_rl if first_rl > 0 else 1.0
        status = "OK" if ratio < 0.7 else "WARNING" if ratio < 0.95 else "PROBLEM"
        print(f"  [{status}] Reward loss: {first_rl:.2f} → {last_rl:.2f} ({ratio:.0%} of initial)")

    # 6. Reward trend
    rew_events = ea.Scalars("Rewards/rew_avg") if "Rewards/rew_avg" in tags else []
    if len(rew_events) > 10:
        first_rew = sum(e.value for e in rew_events[:5]) / 5
        last_rew = sum(e.value for e in rew_events[-5:]) / 5
        status = "OK" if last_rew > first_rew * 1.1 else "WARNING" if last_rew > first_rew * 0.9 else "PROBLEM"
        print(f"  [{status}] Reward trend: {first_rew:.1f} → {last_rew:.1f}")
        if last_rew < first_rew * 0.9:
            issues.append("Rewards getting worse → agent may be learning bad policy from MCTS")

    # --- Summary ---
    print(f"\n=== Summary ===\n")
    if not issues:
        print("  No issues detected. Training looks healthy.")
    else:
        print(f"  Found {len(issues)} issue(s):\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    analyze(sys.argv[1])
