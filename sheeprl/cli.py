import datetime
import importlib
import os
import time
import warnings

import hydra
from lightning import Fabric
from lightning.fabric.accelerators.tpu import TPUAccelerator
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from lightning.fabric.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import tasks
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import dotdict, print_config


def run_algorithm(cfg: DictConfig):
    """Run the algorithm specified in the configuration.

    Args:
        cfg (DictConfig): the loaded configuration.
    """
    if cfg.fabric.strategy == "fsdp":
        raise ValueError(
            "FSDPStrategy is currently not supported. Please launch the script with another strategy: "
            "`python sheeprl.py fabric.strategy=...`"
        )

    print_config(cfg)
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    # Given the algorithm's name, retrieve the module where
    # 'cfg.algo.name'.py is contained; from there retrieve the
    # `register_algorithm`-decorated entrypoint;
    # the entrypoint will be launched by Fabric with `fabric.launch(entrypoint)`
    module = None
    decoupled = False
    entrypoint = None
    algo_name = cfg.algo.name
    for _module, _algos in tasks.items():
        for _algo in _algos:
            if algo_name == _algo["name"]:
                module = _module
                entrypoint = _algo["entrypoint"]
                decoupled = _algo["decoupled"]
                break
    if module is None:
        raise RuntimeError(f"Given the algorithm named `{algo_name}`, no module has been found to be imported.")
    if entrypoint is None:
        raise RuntimeError(
            f"Given the module and algorithm named `{module}` and `{algo_name}` respectively, "
            "no entrypoint has been found to be imported."
        )
    task = importlib.import_module(f"{module}.{algo_name}")
    utils = importlib.import_module(f"{module}.utils")
    command = task.__dict__[entrypoint]
    if decoupled:
        root_dir = (
            os.path.join("logs", "runs", cfg.root_dir)
            if cfg.root_dir is not None
            else os.path.join("logs", "runs", algo_name, datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
        )
        run_name = (
            cfg.run_name if cfg.run_name is not None else f"{cfg.env.id}_{cfg.exp_name}_{cfg.seed}_{int(time.time())}"
        )
        logger = None
        if cfg.metric.log_level > 0:
            logger = TensorBoardLogger(root_dir=root_dir, name=run_name)
            logger.log_hyperparams(cfg)
        fabric = Fabric(**cfg.fabric, loggers=logger, callbacks=[CheckpointCallback()])
    else:
        if "sac_ae" in module:
            strategy = cfg.fabric.strategy
            is_tpu_available = TPUAccelerator.is_available()
            if strategy is not None:
                warnings.warn(
                    "You are running the SAC-AE algorithm you have specified a strategy different than `ddp`: "
                    f"`python sheeprl.py fabric.strategy={strategy}`. This algorithm is run with the "
                    "`lightning.fabric.strategies.DDPStrategy` strategy, unless a TPU is available."
                )
            if is_tpu_available:
                strategy = "auto"
            else:
                strategy = DDPStrategy(find_unused_parameters=True)
            cfg.fabric.pop("strategy", None)
            fabric = Fabric(**cfg.fabric, strategy=strategy, callbacks=[CheckpointCallback()])
        else:
            fabric = Fabric(**cfg.fabric, callbacks=[CheckpointCallback()])

    if "rlhf" in algo_name:
        cfg.metric = dotdict({"log_level": 1})
    else:
        timer.disabled = cfg.metric.log_level == 0 or cfg.metric.disable_timer
        keys_to_remove = set(cfg.metric.aggregator.metrics.keys()) - utils.AGGREGATOR_KEYS
        for k in keys_to_remove:
            cfg.metric.aggregator.metrics.pop(k, None)
        MetricAggregator.disabled = cfg.metric.log_level == 0 or len(cfg.metric.aggregator.metrics) == 0
    fabric.launch(command, cfg)


def check_configs(cfg: DictConfig):
    """Check the validity of the configuration.

    Args:
        cfg (DictConfig): the loaded configuration to check.
    """


@hydra.main(version_base="1.13", config_path="configs", config_name="config")
def run(cfg: DictConfig):
    """SheepRL zero-code command line utility."""
    check_configs(cfg)
    run_algorithm(cfg)
