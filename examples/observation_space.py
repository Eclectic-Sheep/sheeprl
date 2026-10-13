import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf

from sheeprl.utils.env import make_env
from sheeprl.utils.registry import algorithm_registry
from sheeprl.utils.utils import dotdict


@hydra.main(version_base="1.3", config_path="../sheeprl/configs", config_name="env_config")
def main(cfg: DictConfig) -> None:
    cfg.env.capture_video = False
    # The names of the registered algorithms, listed by `python sheeprl/available_agents.py`
    available_agents = {algo["name"] for algos in algorithm_registry.values() for algo in algos}
    if cfg.agent in available_agents:
        cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
        env: gym.Env = make_env(cfg, cfg.seed, 0)()
    else:
        raise ValueError(
            "Invalid selected agent: check the available agents with the command `python sheeprl/available_agents.py`"
        )

    print()
    print(f"Observation space of `{cfg.env.id}` environment for `{cfg.agent}` agent:")
    print(env.observation_space)
    env.close()
    return


if __name__ == "__main__":
    main()
