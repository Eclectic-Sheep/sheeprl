"""The experiment configurations (`exp=...`) set only keys that exist in the configuration."""

import os

import pytest
from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf

from sheeprl import ROOT_DIR

CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
EXPERIMENTS = sorted(
    name[: -len(".yaml")]
    for name in os.listdir(os.path.join(CONFIGS_DIR, "exp"))
    if name.endswith(".yaml") and name != "default.yaml"
)


def root_keys():
    """The keys of the root of the configuration: the ones of `config.yaml`, its config groups and `run_benchmarks`
    (`exp=sac_benchmarks`)."""
    config = OmegaConf.load(os.path.join(CONFIGS_DIR, "config.yaml"))
    groups = {group for default in config.defaults if not isinstance(default, str) for group in default}
    return (set(config) - {"defaults"}) | groups | {"run_benchmarks"}


def experiment(exp: str) -> DictConfig:
    """The configuration of `exp=<exp>`."""
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        return compose(config_name="config", overrides=[f"exp={exp}"])


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_the_experiments_set_no_unknown_root_key(exp):
    # Some set `total_steps` (or the keys of the observations) at the root, where nothing reads them: the runs lasted
    # the steps of the algorithm, 5M instead of 1M for Crafter
    assert set(experiment(exp)) - root_keys() == set()


@pytest.mark.parametrize("exp", [exp for exp in EXPERIMENTS if "crafter" in exp])
def test_the_crafter_experiments_choose_a_crafter_task(exp):
    # `exp=dreamer_v2_crafter` set `env.id=reward`: the Crafter environment (`sheeprl.envs.crafter.CrafterWrapper`)
    # takes only these two ids
    assert experiment(exp).env.id in {"crafter_reward", "crafter_nonreward"}


def test_dreamer_v2_on_atari_scales_the_discount_loss_as_the_paper():
    # It was 0.5: DreamerV2 scales the discount loss by 5 on Atari (`loss_scales.discount: 5.0` in the Atari config of
    # `danijar/dreamerv2`), as the command in the README of DreamerV2 does
    assert experiment("dreamer_v2_ms_pacman").algo.world_model.discount_scale_factor == 5.0


def test_dreamer_v1_pretrains_as_the_reference_implementation():
    # It did no pretraining: `danijar/dreamer` does 100 gradient steps after the `prefill` (`pretrain=100`)
    assert experiment("dreamer_v1").algo.per_rank_pretrain_steps == 100


@pytest.mark.parametrize(
    "exp", ["dreamer_v3", "dreamer_v3_100k_ms_pacman", "p2e_dv3_exploration", "p2e_dv3_finetuning"]
)
def test_dreamer_v3_learns_the_actor_and_the_critic_at_the_rate_of_the_official_code(exp):
    # They learned at 8e-5, the rate of DreamerV1: DreamerV3 learns both at 3e-5 (`actor_opt` and `critic_opt` of the
    # official `configs.yaml` of 2023, https://github.com/danijar/dreamerv3/blob/8fa35f8/dreamerv3/configs.yaml)
    algo = experiment(exp).algo
    assert algo.actor.optimizer.lr == 3e-5
    assert algo.critic.optimizer.lr == 3e-5
