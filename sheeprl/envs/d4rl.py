from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import gym
import gym.spaces
import gym.wrappers
import gymnasium
import numpy as np

from sheeprl.data.buffers import ReplayBuffer

ENV_SPACES_MAPPING: dict[type[gym.spaces.Space], Callable[..., gymnasium.spaces.Box]] = {
    gym.spaces.Box: gymnasium.spaces.Box,
    gym.spaces.Discrete: partial(gymnasium.spaces.Box, low=0.0, dtype=np.float32, shape=(1,)),
}

import d4rl


class D4RLWrapper(gymnasium.Wrapper):
    """Wrapper for D4RL environments to be SheepRL and Gymnasium compliant.

    Args:
        env: The D4RL environment.
    """

    def __init__(self, id: str):
        env = gym.make(id)
        super().__init__(env)  # type: ignore [arg-type]
        self.env = env  # type: ignore [assignment]
        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = gymnasium.spaces.Dict(
                {
                    k: ENV_SPACES_MAPPING[type(v)](
                        high=v.high if isinstance(v, gym.spaces.Box) else v.n,
                        **(
                            {}
                            if isinstance(v, gym.spaces.Discrete)
                            else {"low": v.low, "shape": v.shape, "dtype": v.dtype}
                        ),
                    )
                    for k, v in env.observation_space.spaces.items()
                }
            )
        else:
            obs_space_kwargs = {}
            if isinstance(env.observation_space, gym.spaces.Box):
                obs_space_kwargs = {
                    "low": env.observation_space.low,
                    "high": env.observation_space.high,
                    "shape": env.observation_space.shape,
                    "dtype": env.observation_space.dtype,
                }
            elif isinstance(env.observation_space, gym.spaces.Discrete):
                obs_space_kwargs = {"high": env.observation_space.n}
            else:
                raise RuntimeError(
                    f"Invalid Observation space, only Box and Discrete are allowed. Got {type(env.observation_space)}"
                )
            self.observation_space = gymnasium.spaces.Dict(
                {"observations": ENV_SPACES_MAPPING[type(env.observation_space)](**obs_space_kwargs)}
            )

        self.action_space: gymnasium.spaces.Box | gymnasium.spaces.Discrete | gymnasium.spaces.MultiDiscrete
        if isinstance(env.action_space, gym.spaces.Box):
            self.action_space = gymnasium.spaces.Box(
                low=env.action_space.low,
                high=env.action_space.high,
                shape=env.action_space.shape,
                dtype=env.action_space.dtype,  # type: ignore [arg-type]
            )
        elif isinstance(env.action_space, gym.spaces.Discrete):
            self.action_space = gymnasium.spaces.Discrete(env.action_space.n)
        elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
            self.action_space = gymnasium.spaces.MultiDiscrete(env.action_space.nvec)
        else:
            raise RuntimeError(
                f"Invalid Action Space ({type(env.action_space)}). "
                "Only discrete, multi-discrete and continuous actions spaces are supported."
            )

        self._render_mode = "rgb_array"

    @property
    def render_mode(self) -> str | None:  # type: ignore [override]
        """Render mode getter."""
        return self._render_mode

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """The reset method."""
        obs = self.env.reset(seed=seed, options=options)
        if not isinstance(obs, np.ndarray):
            raise RuntimeError(f"Observation type not valid, got {type(obs)}")
        return {"observations": obs}, {}

    def step(self, action: np.ndarray | int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Perform one step in the environment.

        Args:
            action: The action to compute.
        """
        if isinstance(self.action_space, gymnasium.spaces.Discrete) and isinstance(action, np.ndarray):
            action = action.squeeze().item()
        obs, reward, done, info = self.env.step(action)  # type: ignore [misc]
        if not isinstance(obs, np.ndarray):
            raise RuntimeError(f"Observation type not valid, got {type(obs)}")
        is_timelimit = False
        if isinstance(self.env, gym.wrappers.TimeLimit):
            is_timelimit = self.env._past_limit()
        return {"observations": obs}, reward, done and not is_timelimit, done and is_timelimit, info

    def render(self):
        """Render function."""
        return self.env.render()

    def get_dataset(
        self, validation_split: float = 0.2, seed: int | None = None
    ) -> tuple[ReplayBuffer, ReplayBuffer | None]:
        """Return a ReplayBuffer as offline dataset.

        Args:
            validation_split: The percentage of the validation set (must be in [0, 1]).
                Default to 0.2.
            seed: The seed to use for the split.
                Default to None.

        Returns the training replay buffer the validation replay buffer.
            (None if the `validation_split` argument is equal to zero).
        """
        dataset: dict[str, np.ndarray] = d4rl.qlearning_dataset(self.env)

        dataset["terminated"] = dataset.pop("terminals").reshape(-1, 1)
        dataset["truncated"] = (
            dataset.pop("timeouts").reshape(-1, 1) if "timeouts" in dataset else np.zeros_like(dataset["terminated"])
        )
        dataset["rewards"] = dataset.pop("rewards").reshape(-1, 1)
        dataset.pop("infos/action_log_probs", None)
        dataset.pop("infos/qpos", None)
        dataset.pop("infos/qvel", None)

        for k in dataset:
            dataset[k] = dataset[k][:, np.newaxis, ...]

        # Split dataset in training and validation sets
        dataset_shape = dataset["terminated"].shape[0]
        num_validation_samples = int(dataset_shape * validation_split)
        num_training_samples = dataset_shape - num_validation_samples
        training_offline_buffer = ReplayBuffer(num_training_samples)
        validation_offline_buffer: ReplayBuffer | None = None
        if validation_split > 0:
            validation_offline_buffer = ReplayBuffer(num_validation_samples)
            permutation = np.random.default_rng(seed).permutation(dataset_shape)
            train_set = {k: v[permutation[:num_training_samples]] for k, v in dataset.items()}
            val_set = {k: v[permutation[num_training_samples:]] for k, v in dataset.items()}
            training_offline_buffer.add(train_set)
            validation_offline_buffer.add(val_set)
        else:
            training_offline_buffer.add(dataset)

        return training_offline_buffer, validation_offline_buffer
