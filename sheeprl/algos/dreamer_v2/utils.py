import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from lightning import Fabric
from torch import Tensor, nn
from torch.distributions import Independent, OneHotCategoricalStraightThrough

from sheeprl.utils.utils import get_dummy_env

if TYPE_CHECKING:
    from sheeprl.algos.dreamer_v3.agent import Player

from sheeprl.algos.dreamer_v2.args import DreamerV2Args
from sheeprl.envs.wrappers import ActionRepeat


def make_env(
    env_id: str,
    seed: int,
    rank: int,
    args: DreamerV2Args,
    run_name: Optional[str] = None,
    prefix: str = "",
) -> Callable[..., gym.Env]:
    """
    Create the callable function to createenvironment and
    force the environment to return only pixels observations.

    Args:
        env_id (str): the id of the environment to initialize.
        seed (int): the seed to use.
        rank (int): the rank of the process.
        args (DreamerV3Args): the configs of the experiment.
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
                height=64,
                width=64,
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
                height=64,
                width=64,
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
                height=64,
                width=64,
                pitch_limits=(args.mine_min_pitch, args.mine_max_pitch),
                seed=args.seed,
                break_speed_multiplier=args.mine_break_speed,
                sticky_attack=args.mine_sticky_attack,
                sticky_jump=args.mine_sticky_jump,
                dense=args.minerl_dense,
                extreme=args.minerl_extreme,
            )
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
                    screen_size=64,
                    grayscale_obs=args.grayscale_obs,
                    scale_obs=False,
                    terminal_on_life_loss=False,
                    grayscale_newaxis=True,
                )

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
            if obs["rgb"].shape[:-1] != (64, 64):
                obs.update({"rgb": cv2.resize(obs["rgb"], (64, 64), interpolation=cv2.INTER_AREA)})

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

    return thunk


def compute_stochastic_state(logits: Tensor, discrete: int = 32, sample=True) -> Tensor:
    """
    Compute the stochastic state from the logits computed by the transition or representaiton model.

    Args:
        logits (Tensor): logits from either the representation model or the transition model.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
        sample (bool): whether or not to sample the stochastic state.
            Default to True.

    Returns:
        The sampled stochastic state.
    """
    logits = logits.view(*logits.shape[:-1], -1, discrete)
    dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
    stochastic_state = dist.rsample() if sample else dist.mode
    return stochastic_state


def init_weights(m: nn.Module):
    """
    Initialize the parameters of the m module acording to the Xavier
    normal method.

    Args:
        m (nn.Module): the module to be initialized.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def zero_init_weights(m: nn.Module):
    """
    Initialize the parameters of the m module acording to the Xavier
    normal method.

    Args:
        m (nn.Module): the module to be initialized.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.zeros_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    bootstrap: Optional[Tensor] = None,
    horizon: int = 15,
    lmbda: float = 0.95,
):
    if bootstrap is None:
        bootstrap = torch.zeros_like(values[-2:-1])
    agg = bootstrap
    next_val = torch.cat((values[1:], bootstrap), dim=0)
    inputs = rewards + continues * next_val * (1 - lmbda)
    lv = []
    for i in reversed(range(horizon)):
        agg = inputs[i] + continues[i] * lmbda * agg
        lv.append(agg)
    return torch.cat(list(reversed(lv)), dim=0)


@torch.no_grad()
def test(
    player: "Player", fabric: Fabric, args: DreamerV2Args, cnn_keys: List[str], mlp_keys: List[str], test_name: str = ""
):
    """Test the model on the environment with the frozen model.

    Args:
        player (Player): the agent which contains all the models needed to play.
        fabric (Fabric): the fabric instance.
    """
    env = make_env(
        args.env_id, args.seed, 0, args, fabric.logger.log_dir, "test" + (f"_{test_name}" if test_name != "" else "")
    )()
    done = False
    cumulative_rew = 0
    device = fabric.device
    next_obs = env.reset(seed=args.seed)[0]
    for k in next_obs.keys():
        next_obs[k] = torch.from_numpy(next_obs[k]).view(1, *next_obs[k].shape).float()
    player.num_envs = 1
    player.init_states()
    while not done:
        # Act greedly through the environment
        preprocessed_obs = {}
        for k, v in next_obs.items():
            if k in cnn_keys:
                preprocessed_obs[k] = v[None, ...].to(device) / 255 - 0.5
            else:
                preprocessed_obs[k] = v[None, ...].to(device)
        real_actions = player.get_greedy_action(
            preprocessed_obs, False, {k: v for k, v in preprocessed_obs.items() if k.startswith("mask")}
        )
        if player.actor.is_continuous:
            real_actions = torch.cat(real_actions, -1).cpu().numpy()
        else:
            real_actions = np.array([real_act.cpu().argmax(dim=-1).numpy() for real_act in real_actions])

        # Single environment step
        next_obs, reward, done, truncated, _ = env.step(real_actions.reshape(env.action_space.shape))
        for k in next_obs.keys():
            next_obs[k] = torch.from_numpy(next_obs[k]).view(1, *next_obs[k].shape).float()
        done = done or truncated or args.dry_run
        cumulative_rew += reward
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()


def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    bootstrap: Optional[Tensor] = None,
    horizon: int = 15,
    lmbda: float = 0.95,
):
    if bootstrap is None:
        bootstrap = torch.zeros_like(values[-1:])
    agg = bootstrap
    next_val = torch.cat((values[1:], bootstrap), dim=0)
    inputs = rewards + continues * next_val * (1 - lmbda)
    lv = []
    for i in reversed(range(horizon)):
        agg = inputs[i] + continues[i] * lmbda * agg
        lv.append(agg)
    return torch.cat(list(reversed(lv)), dim=0)
