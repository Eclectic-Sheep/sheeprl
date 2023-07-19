import os
from typing import Any, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from sheeprl.algos.args import StandardArgs
from sheeprl.envs.wrappers import ActionRepeat, FrameStack, MaskVelocityWrapper


@torch.no_grad()
def gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    next_value: Tensor,
    next_done: Tensor,
    num_steps: int,
    gamma: float,
    gae_lambda: float,
) -> Tuple[Tensor, Tensor]:
    """Compute returns and advantages following https://arxiv.org/abs/1506.02438

    Args:
        rewards (Tensor): all rewards collected from the last rollout
        values (Tensor): all values collected from the last rollout
        dones (Tensor): all dones collected from the last rollout
        next_value (Tensor): next observation
        next_done (Tensor): next done
        num_steps (int): the number of steps played
        gamma (float): discout factor
        gae_lambda (float): lambda for GAE estimation

    Returns:
        estimated returns
        estimated advantages
    """
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    not_done = torch.logical_not(dones)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = torch.logical_not(next_done)
            nextvalues = next_value
        else:
            nextnonterminal = not_done[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return returns, advantages


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


@torch.no_grad()
def normalize_tensor(tensor: Tensor, eps: float = 1e-8, mask: Optional[Tensor] = None):
    if mask is None:
        mask = torch.ones_like(tensor, dtype=torch.bool)
    return (tensor - tensor[mask].mean()) / (tensor[mask].std() + eps)


def polynomial_decay(
    current_step: int,
    *,
    initial: float = 1.0,
    final: float = 0.0,
    max_decay_steps: int = 100,
    power: float = 1.0,
) -> float:
    if current_step > max_decay_steps or initial == final:
        return final
    else:
        return (initial - final) * ((1 - current_step / max_decay_steps) ** power) + final


def make_env(
    env_id: str,
    seed: Optional[int],
    idx: int,
    capture_video: bool,
    run_name: Optional[str] = None,
    prefix: str = "",
    mask_velocities: bool = False,
    vector_env_idx: int = 0,
    action_repeat: int = 1,
):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        if mask_velocities:
            env = MaskVelocityWrapper(env)
        env = ActionRepeat(env, action_repeat)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if vector_env_idx == 0 and idx == 0 and run_name is not None:
                env = gym.experimental.wrappers.RecordVideoV0(
                    env,
                    os.path.join(run_name, prefix + "_videos" if prefix else "videos"),
                    disable_logger=True,
                )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_dict_env(
    env_id: str,
    seed: int,
    rank: int,
    args: "StandardArgs",
    run_name: Optional[str] = None,
    prefix: str = "",
    mask_velocities: bool = False,
    vector_env_idx: int = 0,
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

    def thunk():
        env_spec = ""
        _env_id = env_id.lower()
        if "dummy" in _env_id:
            env = get_dummy_env(_env_id)
        elif "dmc" in _env_id:
            from sheeprl.envs.dmc import DMCWrapper

            _, domain, task = _env_id.split("_")
            env = DMCWrapper(
                domain,
                task,
                from_pixels=True,
                height=args.frame_size,
                width=args.frame_size,
                frame_skip=args.action_repeat,
                seed=seed,
            )
        elif "minedojo" in _env_id:
            from sheeprl.envs.minedojo import MineDojoWrapper

            task_id = "_".join(env_id.split("_")[1:])
            start_position = (
                {
                    "x": float(args.mine_start_position[0]),
                    "y": float(args.mine_start_position[1]),
                    "z": float(args.mine_start_position[2]),
                    "pitch": float(args.mine_start_position[3]),
                    "yaw": float(args.mine_start_position[4]),
                }
                if args.mine_start_position is not None
                else None
            )
            env = MineDojoWrapper(
                task_id,
                height=args.frame_size,
                width=args.frame_size,
                pitch_limits=(args.mine_min_pitch, args.mine_max_pitch),
                seed=args.seed,
                start_position=start_position,
            )
            args.action_repeat = 1
        elif "minerl" in _env_id:
            from sheeprl.envs.minerl import MineRLWrapper

            task_id = "_".join(env_id.split("_")[1:])
            env = MineRLWrapper(
                task_id,
                height=args.frame_size,
                width=args.frame_size,
                pitch_limits=(args.mine_min_pitch, args.mine_max_pitch),
                seed=args.seed,
                break_speed_multiplier=args.mine_break_speed,
                sticky_attack=args.mine_sticky_attack,
                sticky_jump=args.mine_sticky_jump,
                dense=args.minerl_dense,
                extreme=args.minerl_extreme,
            )
            args.action_repeat = 1
        else:
            env_spec = gym.spec(env_id).entry_point
            env = gym.make(env_id, render_mode="rgb_array")
            if "mujoco" in env_spec:
                env.frame_skip = 0
            elif "atari" in env_spec:
                if args.atari_noop_max < 0:
                    raise ValueError(
                        f"Negative value of atart_noop_max parameter ({args.atari_noop_max}), the minimum value allowed is 0"
                    )
                env = gym.wrappers.AtariPreprocessing(
                    env,
                    noop_max=args.atari_noop_max,
                    frame_skip=args.action_repeat,
                    screen_size=args.frame_size,
                    grayscale_obs=args.grayscale_obs,
                    scale_obs=False,
                    terminal_on_life_loss=False,
                    grayscale_newaxis=True,
                )
        if mask_velocities:
            env = MaskVelocityWrapper(env)

        # action repeat
        if "atari" not in env_spec and "dmc" not in env_id:
            env = ActionRepeat(env, args.action_repeat)

        # create dict
        if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) < 3:
            env = gym.wrappers.PixelObservationWrapper(
                env, pixels_only=len(env.observation_space.shape) == 2, pixel_keys=("rgb",)
            )
        elif isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) == 3:
            env = gym.wrappers.TransformObservation(env, lambda obs: {"rgb": obs})
            env.observation_space = gym.spaces.Dict({"rgb": env.observation_space})

        shape = env.observation_space["rgb"].shape
        is_3d = len(shape) == 3
        is_grayscale = not is_3d or shape[0] == 1 or shape[-1] == 1
        channel_first = not is_3d or shape[0] in (1, 3)

        def transform_obs(obs: Dict[str, Any]):
            # to 3D image
            if not is_3d:
                obs.update({"rgb": np.expand_dims(obs["rgb"], axis=0)})

            # channel last (opencv needs it)
            if channel_first:
                obs.update({"rgb": obs["rgb"].transpose(1, 2, 0)})

            # resize
            if obs["rgb"].shape[:-1] != (args.frame_size, args.frame_size):
                obs.update(
                    {"rgb": cv2.resize(obs["rgb"], (args.frame_size, args.frame_size), interpolation=cv2.INTER_AREA)}
                )

            # to grayscale
            if args.grayscale_obs and not is_grayscale:
                obs.update({"rgb": cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2GRAY)})

            # back to 3D
            if len(obs["rgb"].shape) == 2:
                obs.update({"rgb": np.expand_dims(obs["rgb"], axis=-1)})
                if not args.grayscale_obs:
                    obs.update({"rgb": np.repeat(obs["rgb"], 3, axis=-1)})

            # channel first (PyTorch default)
            obs.update({"rgb": obs["rgb"].transpose(2, 0, 1)})

            return obs

        env = gym.wrappers.TransformObservation(env, transform_obs)
        env.observation_space["rgb"] = gym.spaces.Box(0, 255, (1 if args.grayscale_obs else 3, 64, 64), np.uint8)

        if args.frame_stack > 0:
            env = FrameStack(env, args.frame_stack, args.frame_stack_keys)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        if args.max_episode_steps > 0:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps // args.action_repeat)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if args.capture_video and rank == 0 and vector_env_idx == 0 and run_name is not None:
            env = gym.experimental.wrappers.RecordVideoV0(
                env, os.path.join(run_name, prefix + "_videos" if prefix else "videos"), disable_logger=True
            )
            env.metadata["render_fps"] = env.frames_per_sec
        return env

    return thunk


def get_dummy_env(env_id: str):
    if "continuous" in env_id:
        from sheeprl.envs.dummy import ContinuousDummyEnv

        env = ContinuousDummyEnv()
    elif "multidiscrete" in env_id:
        from sheeprl.envs.dummy import MultiDiscreteDummyEnv

        env = MultiDiscreteDummyEnv()
    elif "discrete" in env_id:
        from sheeprl.envs.dummy import DiscreteDummyEnv

        env = DiscreteDummyEnv()
    else:
        raise ValueError(f"Unrecognized dummy environment: {env_id}")
    return env
