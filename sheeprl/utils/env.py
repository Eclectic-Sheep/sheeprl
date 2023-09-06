import os
import warnings
from typing import Any, Dict, Optional

import cv2
import gymnasium as gym
import numpy as np

from sheeprl.algos.args import StandardArgs
from sheeprl.envs.wrappers import ActionRepeat, FrameStack, MaskVelocityWrapper


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
                height=args.screen_size,
                width=args.screen_size,
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
                height=args.screen_size,
                width=args.screen_size,
                pitch_limits=(args.mine_min_pitch, args.mine_max_pitch),
                seed=seed,
                start_position=start_position,
            )
            args.action_repeat = 1
        elif "minerl" in _env_id:
            from sheeprl.envs.minerl import MineRLWrapper

            task_id = "_".join(env_id.split("_")[1:])
            env = MineRLWrapper(
                task_id,
                height=args.screen_size,
                width=args.screen_size,
                pitch_limits=(args.mine_min_pitch, args.mine_max_pitch),
                seed=seed,
                break_speed_multiplier=args.mine_break_speed,
                sticky_attack=args.mine_sticky_attack,
                sticky_jump=args.mine_sticky_jump,
                dense=args.minerl_dense,
                extreme=args.minerl_extreme,
            )
            args.action_repeat = 1
        elif "diambra" in _env_id:
            from sheeprl.envs.diambra_wrapper import DiambraWrapper

            if not args.sync_env:
                raise ValueError("You must use the SyncVectorEnv with DIAMBRA envs, set args.sync_env to True")
            if args.diambra_noop_max < 0:
                raise ValueError(
                    f"Negative value of diambra_noop_max parameter ({args.atari_noop_max}), "
                    "the minimum value allowed is 0"
                )
            task_id = "_".join(env_id.split("_")[1:])
            env = DiambraWrapper(
                env_id=task_id,
                action_space=args.diambra_action_space,
                screen_size=args.screen_size,
                grayscale=args.grayscale_obs,
                attack_but_combination=args.diambra_attack_but_combination,
                actions_stack=args.diambra_actions_stack,
                noop_max=args.diambra_noop_max,
                sticky_actions=args.action_repeat,
                seed=seed,
                rank=rank + vector_env_idx,
                diambra_settings={},
                diambra_wrappers={},
            )
        else:
            env_spec = gym.spec(env_id).entry_point
            env = gym.make(env_id, render_mode="rgb_array")
            if "mujoco" in env_spec:
                env.frame_skip = 0
            elif "atari" in env_spec:
                if args.atari_noop_max < 0:
                    raise ValueError(
                        f"Negative value of atari_noop_max parameter ({args.atari_noop_max}), "
                        "the minimum value allowed is 0"
                    )
                env = gym.wrappers.AtariPreprocessing(
                    env,
                    noop_max=args.atari_noop_max,
                    frame_skip=args.action_repeat,
                    screen_size=args.screen_size,
                    grayscale_obs=args.grayscale_obs,
                    scale_obs=False,
                    terminal_on_life_loss=False,
                    grayscale_newaxis=True,
                )
        if mask_velocities:
            env = MaskVelocityWrapper(env)

        # action repeat
        if "atari" not in env_spec and "dmc" not in _env_id and "diambra" not in _env_id:
            env = ActionRepeat(env, args.action_repeat)

        # Create observation dict
        if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) < 2:
            if args.cnn_keys is not None and len(args.cnn_keys) > 0:
                if len(args.cnn_keys) > 1:
                    warnings.warn(
                        f"Multiple cnn keys have been specified and only one pixel observation is allowed in {env_id}, "
                        f"only the first one is kept: {args.cnn_keys[0]}"
                    )
                env = gym.wrappers.PixelObservationWrapper(
                    env, pixels_only=len(args.mlp_keys) == 0, pixel_keys=(args.cnn_keys[0],)
                )
            else:
                if args.mlp_keys is not None and len(args.mlp_keys) > 0:
                    if len(args.mlp_keys) > 1:
                        warnings.warn(
                            "Multiple mlp keys have been specified and only one pixel observation "
                            f"is allowed in {env_id}, "
                            f"only the first one is kept: {args.mlp_keys[0]}"
                        )
                    mlp_key = args.mlp_keys[0]
                else:
                    mlp_key = "state"
                    args.mlp_keys = [mlp_key]
                env = gym.wrappers.TransformObservation(env, lambda obs: {mlp_key: obs})
                env.observation_space = gym.spaces.Dict({mlp_key: env.observation_space})
        elif isinstance(env.observation_space, gym.spaces.Box) and 2 <= len(env.observation_space.shape) <= 3:
            if args.cnn_keys is not None and len(args.cnn_keys) > 1:
                warnings.warn(
                    f"Multiple cnn keys have been specified and only one pixel observation is allowed in {env_id}, "
                    f"only the first one is kept: {args.cnn_keys[0]}"
                )
                cnn_key = args.cnn_keys[0]
            else:
                cnn_key = "rgb"
                args.cnn_keys = [cnn_key]
            env = gym.wrappers.TransformObservation(env, lambda obs: {cnn_key: obs})
            env.observation_space = gym.spaces.Dict({cnn_key: env.observation_space})

        env_cnn_keys = set(
            [k for k in env.observation_space.spaces.keys() if len(env.observation_space[k].shape) in {2, 3}]
        )
        if args.cnn_keys is None:
            user_cnn_keys = set()
        else:
            user_cnn_keys = set(args.cnn_keys)
        cnn_keys = env_cnn_keys.intersection(user_cnn_keys)

        def transform_obs(obs: Dict[str, Any]):
            for k in cnn_keys:
                shape = obs[k].shape
                is_3d = len(shape) == 3
                is_grayscale = not is_3d or shape[0] == 1 or shape[-1] == 1
                channel_first = not is_3d or shape[0] in (1, 3)

                # to 3D image
                if not is_3d:
                    obs.update({k: np.expand_dims(obs[k], axis=0)})

                # channel last (opencv needs it)
                if channel_first:
                    obs.update({k: obs[k].transpose(1, 2, 0)})

                # resize
                if obs[k].shape[:-1] != (args.screen_size, args.screen_size):
                    obs.update(
                        {k: cv2.resize(obs[k], (args.screen_size, args.screen_size), interpolation=cv2.INTER_AREA)}
                    )

                # to grayscale
                if args.grayscale_obs and not is_grayscale:
                    obs.update({k: cv2.cvtColor(obs[k], cv2.COLOR_RGB2GRAY)})

                # back to 3D
                if len(obs[k].shape) == 2:
                    obs.update({k: np.expand_dims(obs[k], axis=-1)})
                    if not args.grayscale_obs:
                        obs.update({k: np.repeat(obs[k], 3, axis=-1)})

                # channel first (PyTorch default)
                obs.update({k: obs[k].transpose(2, 0, 1)})

            return obs

        env = gym.wrappers.TransformObservation(env, transform_obs)
        for k in cnn_keys:
            env.observation_space[k] = gym.spaces.Box(
                0, 255, (1 if args.grayscale_obs else 3, args.screen_size, args.screen_size), np.uint8
            )

        if cnn_keys is not None and len(cnn_keys) > 0 and args.frame_stack > 0:
            if args.frame_stack_dilation <= 0:
                raise ValueError(
                    f"The frame stack dilation argument must be greater than zero, got: {args.frame_stack_dilation}"
                )
            env = FrameStack(env, args.frame_stack, cnn_keys, args.frame_stack_dilation)

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
