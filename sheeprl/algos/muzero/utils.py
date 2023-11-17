USE_C = True

import os

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric

if USE_C:
    from sheeprl._node import MinMaxStats, Node, backpropagate, rollout
else:
    from sheeprl.algos.muzero.mcts_utils import Node, MinMaxStats, backpropagate, rollout

from sheeprl.algos.muzero.agent import MuzeroAgent
from sheeprl.utils.utils import inverse_symsqrt, symsqrt, two_hot_decoder, two_hot_encoder


def support_to_scalar(array: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Converts a support representation to a scalar."""
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    return inverse_symsqrt(two_hot_decoder(array, support))


def scalar_to_support(scalar: np.ndarray, support_range: int) -> torch.Tensor:
    """Converts a scalar to a support representation."""
    return two_hot_encoder(symsqrt(scalar), support_range)


def make_env(
    env_id,
    seed,
    idx,
    capture_video,
    run_name,
    prefix: str = "",
    vector_env_idx: int = 0,
    frame_stack: int = 1,
    screen_size: int = 64,
    action_repeat: int = 1,
):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")  # , frameskip=1)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and vector_env_idx == 0 and idx == 0:
            env = gym.experimental.wrappers.RecordVideoV0(
                env, os.path.join(run_name, prefix + "_videos" if prefix else "videos"), disable_logger=True
            )
        # env = AtariPreprocessing(
        #     env,
        #     screen_size=screen_size,
        #     frame_skip=action_repeat,
        #     grayscale_obs=False,
        #     grayscale_newaxis=True,
        #     scale_obs=True,
        # )
        # env = gym.wrappers.TransformObservation(env, lambda obs: obs.transpose(2, 0, 1))
        # env.observation_space = gym.spaces.Box(
        #     0, 1, (env.observation_space.shape[2], screen_size, screen_size), np.float32
        # )
        # if frame_stack > 0:
        #     env = gym.wrappers.FrameStack(env, frame_stack)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def visit_softmax_temperature(training_steps: int) -> float:
    """Scale the temperature of the softmax action selection by the number of moves played in the game.

    Args:
        training_steps (int): The number of time the network's weight have been updated (an optimizer.step()).
    """
    if training_steps < 500e3:
        return 1.0
    elif training_steps < 750e3:
        return 0.5
    else:
        return 0.25


class MCTS:
    def __init__(
        self,
        agent: MuzeroAgent,
        num_simulations: int,
        gamma: float = 0.997,
        dirichlet_alpha: float = 0.25,
        exploration_fraction: float = 0.25,
        support_size: int = 300,
        pbc_base: float = 19652.0,
        pbc_init: float = 1.25,
    ):
        self.agent = agent
        self.num_simulations = num_simulations
        self.gamma = gamma
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.support_size = support_size
        self.pbc_base = pbc_base
        self.pbc_init = pbc_init
        self.support = np.linspace(-support_size, support_size, support_size * 2 + 1)

    def search(self, observation: np.ndarray) -> Node:
        """Runs MCTS for num_simulations and modifies the root node in place with the result."""
        # Initialize the root, the hidden state and compute the actor's policy logits
        root = Node(0.0)
        hidden_state, policy_logits, _ = self.agent.initial_inference(torch.as_tensor(observation))
        root.hidden_state = hidden_state.reshape(1, 1, -1)

        # Use the actor's policy to initialize the children of the node.
        # The policy results are used as prior probabilities.
        normalized_policy = torch.nn.functional.softmax(policy_logits, dim=-1)
        root.expand(normalized_policy.squeeze().tolist())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.children)).tolist()
        root.add_exploration_noise(noise, self.exploration_fraction)

        # Expand until an unvisited node (i.e. not yet expanded) is reached
        min_max_stats = MinMaxStats()

        for _ in range(self.num_simulations):
            search_path = rollout(root, self.pbc_base, self.pbc_init, self.gamma, min_max_stats)
            # When a path from the starting node to an unvisited node is found, expand the unvisited node
            node = search_path[-1]
            parent = search_path[-2]
            imagined_action = parent.imagined_action
            hidden_state, reward, policy_logits, value = self.agent.recurrent_inference(
                torch.tensor([imagined_action]).view(1, 1).to(device=parent.hidden_state.device, dtype=torch.float32),
                parent.hidden_state,
            )
            value = support_to_scalar(torch.nn.functional.softmax(value, dim=-1), self.support).item()
            node.hidden_state = hidden_state
            node.reward = support_to_scalar(torch.nn.functional.softmax(reward, dim=-1), self.support).item()
            normalized_policy = torch.nn.functional.softmax(policy_logits, dim=-1)
            priors = normalized_policy.squeeze().tolist()

            # Backpropagate the search path to update the nodes' statistics
            backpropagate(search_path, priors, value, self.gamma, min_max_stats)
        return root


@torch.no_grad()
def test(agent: MuzeroAgent, env: gym.Env, fabric: Fabric):  # , args: MuzeroArgs):
    # agent.eval()
    # done = False
    # cumulative_rew = 0
    # next_obs = torch.tensor(
    #     np.array(env.reset(seed=args.seed)[0]), device=fabric.device, dtype=torch.float32
    # ).unsqueeze(0)
    # while not done:
    #     # Act greedly through the environment
    #     node = Node(prior=0, image=next_obs)
    #
    #     # start MCTS
    #     node.mcts(agent, args.num_simulations, args.gamma, args.dirichlet_alpha, args.exploration_fraction)
    #
    #     # Select action based on the visit count distribution and the temperature
    #     visits_count = torch.tensor([child.visit_count for child in node.children.values()])
    #     visits_count = visits_count
    #     action = torch.distributions.Categorical(logits=visits_count).sample()
    #     print(f"Mcts completed, action: {action}")
    #     # Single environment step
    #     next_obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
    #     cumulative_rew += reward
    #
    #     if args.dry_run:
    #         done = True
    # fabric.print("Test - Reward:", cumulative_rew)
    # fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    # env.close()
    pass
