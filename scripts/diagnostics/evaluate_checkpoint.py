#!/usr/bin/env python3
"""Evaluate a MuZero checkpoint: test MCTS behavior and compare to random baseline.

Usage:
    python scripts/diagnostics/evaluate_checkpoint.py <checkpoint_path> [--episodes 20] [--env LunarLander-v2]

Example:
    python scripts/diagnostics/evaluate_checkpoint.py \
        logs/runs/muzero/.../checkpoint/ckpt_1300_0.ckpt
"""
import argparse

import gymnasium as gym
import numpy as np
import torch

from sheeprl.algos.muzero.agent import MlpDynamics, MuzeroAgent, Predictor
from sheeprl.algos.muzero.ctree import cytree
from sheeprl.algos.muzero.utils import MCTS, support_to_scalar
from sheeprl.models.models import MLP


def load_agent(ckpt_path, obs_shape, num_actions, embedding_size=64, support_size=300):
    full_support_size = 2 * support_size + 1
    agent = MuzeroAgent(
        representation=MLP(
            input_dims=obs_shape, hidden_sizes=(128, 64),
            output_dim=embedding_size, activation=torch.nn.ELU,
        ),
        dynamics=MlpDynamics(
            num_actions=num_actions, embedding_size=embedding_size,
            full_support_size=full_support_size,
        ),
        prediction=Predictor(
            embedding_size=embedding_size, num_actions=num_actions,
            full_support_size=full_support_size,
        ),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {k.replace("_forward_module.", ""): v for k, v in ckpt["agent"].items()}
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def run_episodes(env_id, agent, num_episodes, num_actions, support_size=300, use_mcts=True):
    mcts = MCTS(
        num_simulations=50, value_delta_max=0.01, device="cpu",
        pb_c_base=19652, pb_c_init=1.25, discount=0.99, support_range=support_size,
    )
    env = gym.make(env_id)
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).reshape(1, -1)
            hidden, logits, _ = agent.initial_inference(obs_t)

            if use_mcts:
                roots = cytree.Roots(1, num_actions, 50)
                roots.prepare_no_noise([0.0], logits.tolist())
                mcts.search(roots, agent, hidden.squeeze(0).tolist())
                dists = roots.get_distributions()[0]
                action = int(np.argmax(dists))
            else:
                action = int(torch.argmax(torch.softmax(logits, dim=-1)))

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    env.close()
    return rewards


def run_random(env_id, num_episodes):
    env = gym.make(env_id)
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    env.close()
    return rewards


def inspect_mcts(env_id, agent, num_actions, support_size=300, num_obs=5):
    """Show MCTS visit distributions and model predictions for a few observations."""
    mcts = MCTS(
        num_simulations=50, value_delta_max=0.01, device="cpu",
        pb_c_base=19652, pb_c_init=1.25, discount=0.99, support_range=support_size,
    )
    env = gym.make(env_id)

    print("\n=== MCTS Visit Distributions ===\n")
    for i in range(num_obs):
        obs, _ = env.reset(seed=i)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).reshape(1, -1)
        hidden, logits, values = agent.initial_inference(obs_t)

        roots = cytree.Roots(1, num_actions, 50)
        noises = [np.random.dirichlet([0.25] * num_actions).astype(np.float32).tolist()]
        roots.prepare(0.25, noises, [0.0], logits.tolist())
        mcts.search(roots, agent, hidden.squeeze(0).tolist())

        dists = roots.get_distributions()[0]
        total = sum(dists)
        probs = [round(v / total, 3) for v in dists]
        val = support_to_scalar(torch.softmax(values, -1), support_size).item()
        root_val = roots.get_values()[0]

        print(f"  obs {i}: visits={dists}  probs={probs}")
        print(f"          net_value={val:.2f}  mcts_value={root_val:.2f}")

        # Show per-action reward/value predictions for first obs
        if i == 0:
            print(f"          Per-action predictions:")
            for a in range(num_actions):
                act = torch.tensor([[a]]).float()
                h_in = torch.tensor([hidden.squeeze(0).tolist()[0]]).float()
                _, rew, _, v = agent.recurrent_inference(act, h_in)
                r = support_to_scalar(torch.softmax(rew, -1), support_size).item()
                v_s = support_to_scalar(torch.softmax(v, -1), support_size).item()
                print(f"            action {a}: pred_reward={r:>8.3f}  pred_value={v_s:>8.3f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--env", default="LunarLander-v2", help="Environment ID")
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--support-size", type=int, default=300)
    args = parser.parse_args()

    env = gym.make(args.env)
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    env.close()

    print(f"=== MuZero Checkpoint Evaluation ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Environment: {args.env} (obs={obs_shape}, actions={num_actions})")
    print(f"  Episodes: {args.episodes}")

    agent = load_agent(args.checkpoint, obs_shape, num_actions, args.embedding_size, args.support_size)

    # 1. MCTS inspection
    inspect_mcts(args.env, agent, num_actions, args.support_size)

    # 2. Random baseline
    print(f"\n=== Random Baseline ({args.episodes} episodes) ===\n")
    random_rewards = run_random(args.env, args.episodes)
    print(f"  Mean: {np.mean(random_rewards):.1f}  Std: {np.std(random_rewards):.1f}")
    print(f"  Min: {np.min(random_rewards):.1f}  Max: {np.max(random_rewards):.1f}")

    # 3. Agent with MCTS
    print(f"\n=== Agent + MCTS ({args.episodes} episodes) ===\n")
    mcts_rewards = run_episodes(args.env, agent, args.episodes, num_actions, args.support_size, use_mcts=True)
    print(f"  Mean: {np.mean(mcts_rewards):.1f}  Std: {np.std(mcts_rewards):.1f}")
    print(f"  Min: {np.min(mcts_rewards):.1f}  Max: {np.max(mcts_rewards):.1f}")

    # 4. Agent without MCTS (raw policy)
    print(f"\n=== Agent Raw Policy ({args.episodes} episodes) ===\n")
    raw_rewards = run_episodes(args.env, agent, args.episodes, num_actions, args.support_size, use_mcts=False)
    print(f"  Mean: {np.mean(raw_rewards):.1f}  Std: {np.std(raw_rewards):.1f}")
    print(f"  Min: {np.min(raw_rewards):.1f}  Max: {np.max(raw_rewards):.1f}")

    # 5. Verdict
    print(f"\n=== Verdict ===\n")
    mcts_vs_random = np.mean(mcts_rewards) - np.mean(random_rewards)
    raw_vs_random = np.mean(raw_rewards) - np.mean(random_rewards)
    mcts_vs_raw = np.mean(mcts_rewards) - np.mean(raw_rewards)

    print(f"  MCTS vs Random:     {mcts_vs_random:>+.1f}")
    print(f"  Raw Policy vs Random: {raw_vs_random:>+.1f}")
    print(f"  MCTS vs Raw Policy: {mcts_vs_raw:>+.1f}")

    if mcts_vs_random > 20:
        print(f"\n  Agent is LEARNING (MCTS improves over random)")
    elif mcts_vs_random > -20:
        print(f"\n  Agent is NOT YET learning (similar to random)")
    else:
        print(f"\n  Agent is WORSE than random (something is broken)")

    if mcts_vs_raw > 10:
        print(f"  MCTS search is HELPING (adds {mcts_vs_raw:.0f} reward over raw policy)")
    else:
        print(f"  MCTS search is NOT HELPING (may need better value/reward model)")


if __name__ == "__main__":
    main()
