import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf

from sheeprl.utils.env import make_env
from sheeprl.utils.utils import dotdict


@hydra.main(version_base="1.3", config_path="../sheeprl/configs", config_name="env_config")
def main(cfg: DictConfig) -> None:
    cfg.env.capture_video = False
    if cfg.agent in {
        "dreamer_v1",
        "dreamer_v2",
        "dreamer_v3",
        "p2e_dv1",
        "p2e_dv2",
        "p2e_dv3",
        "sac_ae",
        "ppo",
        "ppo_decoupled",
        "sac",
        "sac_decoupled",
        "droq",
        "ppo_recurrent",
    }:
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
