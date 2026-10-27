"""The `dotdict` of the configuration: a missing key raises `AttributeError` instead of reading as `None`."""

import copy
import pickle

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from sheeprl.cli import check_configs
from sheeprl.utils.utils import dotdict


def config(exp: str, *overrides: str) -> dotdict:
    """The configuration of `exp=<exp>` as the algorithms receive it (`sheeprl.cli.run`)."""
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"exp={exp}", *overrides])
    return dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))


def test_nested_dicts_become_dotdicts():
    d = dotdict({"algo": {"world_model": {"units": 8}}, "seed": None})
    assert type(d.algo) is dotdict and type(d.algo.world_model) is dotdict
    assert d.algo.world_model.units == 8
    # A key that exists with a `None` value reads as `None`
    assert d.seed is None


def test_a_missing_key_raises():
    # It read as `None`: a typo in a config key (`cfg.algo.lerning_starts`) went unnoticed
    d = dotdict({"algo": {"learning_starts": 0}})
    with pytest.raises(AttributeError, match="'dotdict' object has no key 'lerning_starts'"):
        d.algo.lerning_starts
    # As for any other attribute, `getattr` returns the default and `hasattr` is false (they were `None` and true)
    assert getattr(d, "nope", 1) == 1
    assert hasattr(d, "nope") is False
    # A key that may be absent is read with `get`
    assert d.get("nope") is None


def test_set_and_delete_by_attribute():
    d = dotdict()
    d.k = 1
    assert d["k"] == 1
    del d.k
    assert "k" not in d
    with pytest.raises(AttributeError, match="'dotdict' object has no key 'nope'"):
        del d.nope


@pytest.mark.parametrize(
    "copy_fn", [lambda d: pickle.loads(pickle.dumps(d)), copy.deepcopy], ids=["pickle", "deepcopy"]
)
def test_pickle_and_deepcopy_keep_the_nested_dotdicts(copy_fn):
    # `pickle` and `deepcopy` look up methods such as `__deepcopy__` with `getattr(d, name, None)`: they must get the
    # default, not a missing key
    d = dotdict({"algo": {"world_model": {"units": 8}}, "seed": None})
    c = copy_fn(d)
    assert c == d
    assert type(c) is dotdict and type(c.algo) is dotdict and type(c.algo.world_model) is dotdict


def test_check_configs_accepts_an_on_policy_config():
    # The on-policy algorithms have no `algo.learning_starts`, which `check_configs` read as `cfg.algo.learning_starts`
    cfg = config("ppo")
    assert "learning_starts" not in cfg.algo
    check_configs(cfg)


def test_check_configs_rejects_a_negative_learning_starts():
    with pytest.raises(ValueError, match="learning_starts"):
        check_configs(config("sac", "algo.learning_starts=-1"))
