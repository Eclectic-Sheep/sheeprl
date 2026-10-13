import importlib
import os

import pytest
from omegaconf import OmegaConf

import sheeprl
from sheeprl.utils.registry import algorithm_registry

ALGORITHMS = [(module, algo["name"]) for module, algos in algorithm_registry.items() for algo in algos]


@pytest.mark.parametrize("module,name", ALGORITHMS)
def test_every_model_of_the_model_manager_config_can_be_registered(module, name):
    # The CLI drops the models of `model_manager.models` missing from `MODELS_TO_REGISTER`: the registration at the end
    # of the training then fails, because the algorithm logs more models than there are registration configs
    utils = importlib.import_module(f"{module}.utils")
    cfg = OmegaConf.load(os.path.join(os.path.dirname(sheeprl.__file__), "configs", "model_manager", f"{name}.yaml"))
    models = set(cfg.models)
    assert len(models) > 0
    assert models <= getattr(utils, "MODELS_TO_REGISTER", set()), models - getattr(utils, "MODELS_TO_REGISTER", set())
