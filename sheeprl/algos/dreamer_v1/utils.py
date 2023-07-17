import os
from typing import TYPE_CHECKING, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from lightning import Fabric
from torch import Tensor
from torch.distributions import Distribution, Independent, Normal

from sheeprl.utils.utils import get_dummy_env

if TYPE_CHECKING:
    from sheeprl.algos.dreamer_v1.agent import Player

from sheeprl.algos.dreamer_v1.args import DreamerV1Args
from sheeprl.envs.wrappers import ActionRepeat


def make_env(
    env_id: str,
    seed: int,
    rank: int,
    args: DreamerV1Args,
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
        args (DreamerV1Args): the configs of the experiment.
        run_name (str, optional): the name of the run.
            Default to None.
        prefix (str): the prefix to add to the video folder.
            Default to "".

    Returns:
        The callable function that initializes the environment.
    """
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
                "x": args.mine_start_position[0],
                "y": args.mine_start_position[1],
                "z": args.mine_start_position[2],
                "pitch": args.mine_start_position[3],
                "yaw": args.mine_start_position[4],
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
        env = ActionRepeat(env, args.action_repeat)
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
                terminal_on_life_loss=True,
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


@torch.no_grad()
def test(player: "Player", fabric: Fabric, args: DreamerV1Args, test_name: str = ""):
    """Test the model on the environment with the frozen model.

    Args:
        player (Player): the agent which contains all the models needed to play.
        fabric (Fabric): the fabric instance.
    """
    env: gym.Env = make_env(
        args.env_id, args.seed, 0, args, fabric.logger.log_dir, "test" + (f"_{test_name}" if test_name != "" else "")
    )
    done = False
    cumulative_rew = 0
    next_obs = torch.tensor(env.reset(seed=args.seed)[0], device=fabric.device).view(1, 1, *env.observation_space.shape)
    player.init_states()
    while not done:
        # Act greedly through the environment
        action = player.get_greedy_action(next_obs / 255 - 0.5, False)
        if not player.actor.is_continuous:
            action = np.array([act.cpu().argmax() for act in action])
        else:
            action = action[0].cpu().numpy()

        # Single environment step
        next_obs, reward, done, truncated, _ = env.step(action.reshape(env.action_space.shape))
        done = done or truncated or args.dry_run
        cumulative_rew += reward
        next_obs = torch.tensor(next_obs, device=fabric.device).view(1, 1, *env.observation_space.shape)
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
