# Environment Requirements
This repository requires that the environments have certain characteristics, in particular, that they have a [gymnasium-compliant interface](https://gymnasium.farama.org/api/env/).

The main properties/methods that the environment has to provide are the following:
* A `step` function which takes in input the actions and which outputs the next observations, the reward for taking that actions, whether the environment has terminated, whether the environment was truncated, and infomration from the environment about the step.
* A `reset` function which resets the environment and returns the initial observations and some info about the episode.
* A `render` function that renders the environment to help visualizing what the agent sees, some possible render mode are: `human` or `rgb_array`.
* A `close` function that closes the environment.
* An `action_space` property indicating the valid actions, i.e., all the valid actions should be contained in that space. For more info, check [here](https://gymnasium.farama.org/api/spaces/fundamental/).
* An `observation_space` property indicating all the valid observation that an agent can receive from the environment. This observation space must be of type [`gymnasium.spaces.Dict`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Dict), and, its elements cannot be of type `gymnasium.spaces.Dict`, so it must be a flatten dictionary.

> **Note**
>
> All the observations returned by the `step` and `reset` functions must be python dictionary of numpy arrays.

# Add a new Environment
There are two ways to add a new environment:
1. Create from scratch a custom environment by inheriting from the [`gymnasium.Env`](https://gymnasium.farama.org/api/env/#gymnasium-env) class.
2. Take an existing environment and add a wrapper to be compliant with the above directives.

In both cases, the environment or wrapper must be inserted in a dedicated file the `./sheeprl/envs` folder, for instance you should add the `custom_env.py` file in `./sheeprl/envs` folder.
After that, you have to create a new config file and place it in the `./sheeprl/configs/env` folder.

> **Note**
>
> It could be necessary to define the `metadata` property that contains some metadata information about the environment. It is used by the `gym.experimental.wrappers.RecordVideoV0` wrapper, which is responsible to capture the video of the episode.

## Crate from Scratch
If one needs to create a custom environment, then he/she can define a class by by inheriting from the `gymnasium.Env` class. So, you need to define the `__init__` function for initializing the required properties, and then define the `step`, `reset`, `close`, and `render` functions.

The following shows an example of how you can define an environment with continuous actions from scratch:
```python
from typing import List, Tuple

import gymnasium as gym
import numpy as np


class ContinuousDummyEnv(gym.Env):
    def __init__(self, action_dim: int = 2, size: Tuple[int, int, int] = (3, 64, 64), n_steps: int = 128):
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(action_dim,))
        self.observation_space = gym.spaces.Box(0, 256, shape=size, dtype=np.uint8)
        self.reward_range = (-np.inf, np.inf)
        self._current_step = 0
        self._n_steps = n_steps

    def step(self, action):
        done = self._current_step == self._n_steps
        self._current_step += 1
        return (
            np.zeros(self.observation_space.shape, dtype=np.uint8),
            np.zeros(1, dtype=np.float32).item(),
            done,
            False,
            {},
        )

    def reset(self, seed=None, options=None):
        self._current_step = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass
```

## Define a Wrapper for existing Environments
The second option is to create a wrapper for existing environments, so define a class that inherits from the `gymnasium.Wrapper` class.
Then you can redefine, if necessary, the `action_space`, `observation_space`, `render_mode` and `reward_range` properties in the `__init__` function.
Finally, you can define the other functions to make the environment compatible with the library.

The following is the example, we implemented the wrapper for the [Crafter](https://github.com/danijar/crafter) environment. As one can notice, the observations are converted by the `_convert_obs` function. Moreover, in the `step` function, the `truncated` is always set to `False`, since the original environment does not provide this information. Finally, in the `__init__` function the `reward_range`, `observation_space`, `action_space`, `render_mode`, and `metadata` properties are redefined.
```python
from sheeprl.utils.imports import _IS_CRAFTER_AVAILABLE

if not _IS_CRAFTER_AVAILABLE:
    raise ModuleNotFoundError(_IS_CRAFTER_AVAILABLE)

from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame


class CrafterWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, seed: int | None = None) -> None:
        super().__init__(env)
        # We know by the documentation that the observation of the environment is a single image.
        # So we convert it in a dictionary containing only that observation. 
        self.observation_space = spaces.Dict(
            {
                # convert from gym to gymnasium
                "rgb": spaces.Box(
                    self.env.observation_space.low,
                    self.env.observation_space.high,
                    self.env.observation_space.shape,
                    self.env.observation_space.dtype,
                )
            }
        )
        # convert from gym to gymnasium
        self.action_space = spaces.Discrete(self.env.action_space.n)
        self.reward_range = self.env.reward_range or (-np.inf, np.inf)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        # render
        self._render_mode: str = "rgb_array"
        # metadata
        self._metadata = {"render_fps": 30}

    @property
    def render_mode(self) -> str | None:
        return self._render_mode

    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {"rgb": obs}

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        return self._convert_obs(obs), reward, done, False, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs = self.env.reset()
        return self._convert_obs(obs), {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self.env.render()

    def close(self) -> None:
        return super().close()
```

## Add Config File
The last step to perform is to add the config file, more precisely, it must contain the following fields:
* `id` of the environment you want to instantiate.
* `wrapper`: the settings to instantiate the environment.

For example, the Crafter config file is the following:
```yaml
defaults:
  - default
  - _self_

# Override from `default` config
id: crafter_reward
action_repeat: 1
capture_video: False
reward_as_observation: True

# Wrapper to be instantiated
wrapper:
  # the custom wrapper is instantiated and the original environment
  # is passed as argument.
  _target_: sheeprl.envs.crafter.CrafterWrapper
  env:
    # creation of the original environment.
    _target_: crafter.Env
    reward: True
    size: 
      - ${env.screen_size}
      - ${env.screen_size}
    seed: ${seed}
  seed: ${seed}
```
In this case, the main environment is instantiated, then it is passed to the wrapper.