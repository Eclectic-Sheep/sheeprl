"""Print the observation space of an environment as an agent sees it, for the keys its inputs declare: as the
environment returns it, after the `frame` stage of the transforms (what the vectorized environment reports) and after
the `rows` stage (what the store holds). E.g.

    python examples/observation_space.py agent=dreamer_v3 env=atari algo.inputs.image.keys=[state]
"""

import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf

from sheeprl.envs.wrappers import TransformsWrapper
from sheeprl.transforms.observations import RowsTransforms
from sheeprl.utils.env import make_env
from sheeprl.utils.registry import algorithm_registry
from sheeprl.utils.utils import dotdict


@hydra.main(version_base="1.3", config_path="../sheeprl/configs", config_name="env_config")
def main(cfg: DictConfig) -> None:
    cfg.env.capture_video = False
    # The names of the registered algorithms, listed by `python sheeprl/available_agents.py`
    available_agents = {algo["name"] for algos in algorithm_registry.values() for algo in algos}
    if cfg.agent not in available_agents:
        raise ValueError(
            "Invalid selected agent: check the available agents with the command `python sheeprl/available_agents.py`"
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    env: gym.Env = make_env(cfg, cfg.seed, 0)()
    # The space before the `frame` stage: the one the wrappers of `make_env` give to `TransformsWrapper`
    wrapper = env
    while not isinstance(wrapper, TransformsWrapper):
        wrapper = wrapper.env
    rows = RowsTransforms.from_config(cfg, env.observation_space)

    print()
    print(f"Observation space of `{cfg.env.id}` environment for `{cfg.agent}` agent:")
    for k in env.observation_space.spaces:
        print(f"- {k}")
        print(f"    from the environment: {wrapper.env.observation_space[k]}")
        print(f"    after the frame stage: {env.observation_space[k]}")
        print(f"    after the rows stage (stored): {rows.space[k]}")
    env.close()


if __name__ == "__main__":
    main()
