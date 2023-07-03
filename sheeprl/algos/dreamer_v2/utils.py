import os
from typing import TYPE_CHECKING, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from torch import Tensor, nn
from torch.distributions import OneHotCategoricalStraightThrough

if TYPE_CHECKING:
    from sheeprl.algos.dreamer_v2.agent import Player

from sheeprl.algos.dreamer_v2.args import DreamerV2Args
from sheeprl.envs.wrappers import ActionRepeat


def make_env(
    env_id: str,
    seed: int,
    rank: int,
    args: DreamerV2Args,
    run_name: Optional[str] = None,
    prefix: str = "",
) -> gym.Env:
    """
    Create the callable function to createenvironment and
    force the environment to return only pixels observations.

    Args:
        env_id (str): the id of the environment to initialize.
        seed (int): the seed to use.
        rank (int): the rank of the process.
        args (DreamerV2Args): the configs of the experiment.
        run_name (str, optional): the name of the run.
            Default to None.
        prefix (str): the prefix to add to the video folder.
            Default to "".

    Returns:
        The callable function that initializes the environment.
    """
    if "dmc" in env_id.lower():
        from sheeprl.envs.dmc import DMCWrapper

        _, domain, task = env_id.lower().split("_")
        env = DMCWrapper(
            domain,
            task,
            from_pixels=True,
            height=64,
            width=64,
            frame_skip=args.action_repeat,
            seed=seed,
        )
    else:
        env_spec = gym.spec(env_id).entry_point
        if "mujoco" in env_spec:
            try:
                env = gym.make(env_id, render_mode="rgb_array", terminate_when_unhealthy=False)
            except:
                env = gym.make(env_id, render_mode="rgb_array")
            env.frame_skip = 0
        else:
            env = gym.make(env_id, render_mode="rgb_array")
        if "atari" in env_spec:
            if args.atari_noop_max < 0:
                raise ValueError(
                    f"Negative value of atart_noop_max parameter ({args.atari_noop_max}), the minimum value allowed is 0"
                )
            env = gym.wrappers.AtariPreprocessing(
                env,
                noop_max=args.atari_noop_max,
                frame_skip=args.action_repeat,
                screen_size=64,
                grayscale_obs=args.grayscale_obs,
                scale_obs=False,
                terminal_on_life_loss=False,
                grayscale_newaxis=True,
            )
        else:
            env = ActionRepeat(env, args.action_repeat)
            if isinstance(env.observation_space, gym.spaces.Box) or len(env.observation_space.shape) < 3:
                env = gym.wrappers.PixelObservationWrapper(env)
                env = gym.wrappers.TransformObservation(env, lambda obs: obs["pixels"])
                env.observation_space = env.observation_space["pixels"]
            env = gym.wrappers.ResizeObservation(env, (64, 64))
            if args.grayscale_obs:
                env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        env = gym.wrappers.TransformObservation(env, lambda obs: obs.transpose(2, 0, 1))
        env.observation_space = gym.spaces.Box(
            0, 255, (env.observation_space.shape[-1], *env.observation_space.shape[:2]), np.uint8
        )
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    if args.max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps // args.action_repeat)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if args.capture_video and rank == 0 and run_name is not None:
        env = gym.experimental.wrappers.RecordVideoV0(
            env, os.path.join(run_name, prefix + "_videos" if prefix else "videos"), disable_logger=True
        )
        env.metadata["render_fps"] = env.frames_per_sec
    return env


def compute_stochastic_state(
    logits: Tensor,
    discrete: int = 32,
) -> Tensor:
    """
    Compute the stochastic state from the logits computed by the transition or representaiton model.

    Args:
        logits (Tensor): logits from either the representation model or the transition model.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.

    Returns:
        The mean and the standard deviation of the distribution of the stochastic state.
        The sampled stochastic state.
    """
    logits = logits.view(*logits.shape[:-1], -1, discrete)
    dist = OneHotCategoricalStraightThrough(logits=logits)
    return dist.rsample()


def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    done_mask: Tensor,
    last_values: Tensor,
    horizon: int = 15,
    lmbda: float = 0.95,
) -> Tensor:
    """
    Compute the lambda values by keeping the gradients of the variables.

    Args:
        rewards (Tensor): the estimated rewards in the latent space.
        values (Tensor): the estimated values in the latent space.
        done_mask (Tensor): 1s for the entries that are relative to a terminal step, 0s otherwise.
        last_values (Tensor): the next values for the last state in the horzon.
        horizon: (int, optional): the horizon of imagination.
            Default to 15.
        lmbda (float, optional): the discout lmbda factor for the lambda values computation.
            Default to 0.95.

    Returns:
        The tensor of the computed lambda values.
    """
    last_values = torch.clone(last_values)

    last_lambda_values = 0
    lambda_targets = []
    for step in reversed(range(horizon - 1)):
        if step == horizon - 2:
            next_values = last_values
        else:
            next_values = values[step + 1] * (1 - lmbda)
        delta = rewards[step] + next_values * done_mask[step]
        last_lambda_values = delta + lmbda * done_mask[step] * last_lambda_values
        lambda_targets.append(last_lambda_values)

    return torch.stack(list(reversed(lambda_targets)), dim=0)


def init_weights(m: nn.Module):
    """
    Initialize the parameters of the m module acording to the method described in
    [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852) using a uniform distribution.

    Args:
        m (nn.Module): the module to be initialized.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


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
def test(player: "Player", fabric: Fabric, args: DreamerV2Args):
    """Test the model on the environment with the frozen model.

    Args:
        player (Player): the agent which contains all the models needed to play.
        fabric (Fabric): the fabric instance.
    """
    env: gym.Env = make_env(args.env_id, args.seed, 0, args, fabric.logger.log_dir, "test")
    done = False
    cumulative_rew = 0
    next_obs = torch.tensor(env.reset(seed=args.seed)[0], device=fabric.device).view(1, 1, *env.observation_space.shape)
    player.init_states()
    while not done:
        # Act greedly through the environment
        action = player.get_greedy_action(next_obs / 255 - 0.5, False).cpu().numpy()

        # Single environment step
        if player.actor.is_continuous:
            next_obs, reward, done, truncated, _ = env.step(action[0, 0])
        else:
            next_obs, reward, done, truncated, _ = env.step(action.argmax())
        done = done or truncated or args.dry_run
        cumulative_rew += reward
        next_obs = torch.tensor(next_obs, device=fabric.device).view(1, 1, *env.observation_space.shape)
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
