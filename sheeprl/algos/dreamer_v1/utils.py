import os
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from lightning import Fabric
from torch import Tensor, nn
from torch.distributions import Distribution, Independent, Normal

if TYPE_CHECKING:
    from sheeprl.algos.dreamer_v1.agent import Player

from sheeprl.algos.dreamer_v1.args import DreamerV1Args
from sheeprl.algos.dreamer_v2.utils import make_env
from sheeprl.envs.wrappers import ActionRepeat


def compute_stochastic_state(
    state_information: Tensor,
    event_shape: Optional[int] = 1,
    min_std: Optional[float] = 0.1,
) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
    """
    Compute the stochastic state from the information of the distribution of the stochastic state.

    Args:
        state_information (Tensor): information about the distribution of the stochastic state,
            it is the output of either the representation model or the transition model.
        event_shape (int, optional): how many batch dimensions have to be reinterpreted as event dims.
            Default to 1.
        min_std (float, optional): the minimum value for the standard deviation.
            Default to 0.1.

    Returns:
        The mean and the standard deviation of the distribution of the stochastic state.
        The sampled stochastic state.
    """
    mean, std = torch.chunk(state_information, 2, -1)
    std = F.softplus(std) + min_std
    state_distribution: Distribution = Normal(mean, std)
    if event_shape:
        # it is necessary an Independent distribution because
        # it is necessary to create (batch_size * sequence_length) independent distributions,
        # each producing a sample of size equal to the stochastic size
        state_distribution = Independent(state_distribution, event_shape)
    stochastic_state = state_distribution.rsample()
    return (mean, std), stochastic_state


def cnn_forward(
    model: nn.Module,
    input: Tensor,
    input_dim: Union[torch.Size, Tuple[int, ...]],
    output_dim: Union[torch.Size, Tuple[int, ...]],
) -> Tensor:
    """
    Compute the forward of either the encoder or the observation model of the World model.
    It flattens all the dimensions before the model input_size, i.e., (C_in, H, W) for the encoder
    and (recurrent_state_size + stochastic_size) for the observation model.

    Args:
        model (nn.Module): the model.
        input (Tensor): the input tensor of dimension (*, C_in, H, W) or (*, recurrent_state_size + stochastic_size),
            where * means any number of dimensions including None.
        input_dim (Union[torch.Size, Tuple[int, ...]]): the input dimensions,
            i.e., either (C_in, H, W) or (recurrent_state_size + stochastic_size).
        output_dim: the desired dimensions in output.

    Returns:
        The output of dimensions (*, *output_dim).

    Examples:
        >>> encoder
        CNN(
            (network): Sequential(
                (0): Conv2d(3, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU()
                (2): Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (3): ReLU()
                (4): Flatten(start_dim=1, end_dim=-1)
                (5): Linear(in_features=128, out_features=25, bias=True)
            )
        )
        >>> input = torch.rand(10, 20, 3, 4, 4)
        >>> cnn_forward(encoder, input, (3, 4, 4), -1).shape
        torch.Size([10, 20, 25])

        >>> observation_model
        Sequential(
            (0): Linear(in_features=230, out_features=1024, bias=True)
            (1): Unflatten(dim=-1, unflattened_size=(1024, 1, 1))
            (2): ConvTranspose2d(1024, 128, kernel_size=(5, 5), stride=(2, 2))
            (3): ReLU()
            (4): ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2))
            (5): ReLU()
            (6): ConvTranspose2d(64, 32, kernel_size=(6, 6), stride=(2, 2))
            (7): ReLU()
            (8): ConvTranspose2d(32, 3, kernel_size=(6, 6), stride=(2, 2))
        )
        >>> input = torch.rand(10, 20, 230)
        >>> cnn_forward(model, input, (230,), (3, 64, 64)).shape
        torch.Size([10, 20, 3, 64, 64])
    """
    batch_shapes = input.shape[: -len(input_dim)]
    flatten_input = input.reshape(-1, *input_dim)
    model_out = model(flatten_input)
    return model_out.reshape(*batch_shapes, *output_dim)


@torch.no_grad()
def test(player: "Player", fabric: Fabric, args: DreamerV1Args, test_name: str = ""):
    """Test the model on the environment with the frozen model.

    Args:
        player (Player): the agent which contains all the models needed to play.
        fabric (Fabric): the fabric instance.
    """
    env: gym.Env = make_env(
        args.env_id, args.seed, 0, args, fabric.logger.log_dir, "test" + (f"_{test_name}" if test_name != "" else "")
    )()
    done = False
    cumulative_rew = 0
    next_obs = torch.tensor(env.reset(seed=args.seed)[0]["rgb"], device=fabric.device).view(
        1, 1, *env.observation_space["rgb"].shape
    )
    player.num_envs = 1
    player.init_states()
    while not done:
        # Act greedly through the environment
        action = player.get_greedy_action(next_obs / 255 - 0.5, False)
        if not player.actor.is_continuous:
            action = np.array([act.cpu().argmax(dim=-1) for act in action])
        else:
            action = action[0].cpu().numpy()

        # Single environment step
        next_obs, reward, done, truncated, _ = env.step(action.reshape(env.action_space.shape))
        done = done or truncated or args.dry_run
        cumulative_rew += reward
        next_obs = torch.tensor(next_obs["rgb"], device=fabric.device).view(1, 1, *env.observation_space["rgb"].shape)
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
