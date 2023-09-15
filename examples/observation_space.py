import gymnasium as gym
import hydra
from omegaconf import DictConfig

from sheeprl.utils.env import make_env


@hydra.main(version_base=None, config_path="../sheeprl/configs", config_name="env_config")
def main(cfg: DictConfig) -> None:
    cfg.env.capture_video = False
    if cfg.agent in {
        "dreamer_v1",
        "dreamer_v2",
        "dreamer_v3",
        "p2e_dv1",
        "p2e_dv2",
        "sac_ae",
        "ppo",
        "ppo_decoupled",
    }:
        env: gym.Env = make_env(cfg, cfg.seed, 0)()
    elif cfg.agent in {"sac", "sac_decoupled", "droq", "ppo_recurrent"}:
        env: gym.Env = make_env(
            cfg.env.id,
            cfg.seed,
            0,
            False,
            mask_velocities="mask_velocities" in cfg.env and cfg.mask_velocities,
            action_repeat=cfg.env.action_repeat,
        )()
    else:
        raise ValueError(
            "Invalid selected agent: check the available agents with the command `python sheeprl.py --sheeprl_help`"
        )

    print()
    print(f"Observation space of `{cfg.env.id}` environment for `{cfg.agent}` agent:")
    print(env.observation_space)
    env.close()
    return


if __name__ == "__main__":
    main()
