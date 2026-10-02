"""The experiment configurations (`exp=...`) set only keys that exist in the configuration."""

import os

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

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


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_the_experiments_set_no_unknown_root_key(exp):
    # Some set `total_steps` (or the keys of the observations) at the root, where nothing reads them: the runs lasted
    # the steps of the algorithm, 5M instead of 1M for Crafter
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"exp={exp}"])
    assert set(cfg) - root_keys() == set()
