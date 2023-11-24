import importlib
import os
import pathlib
import warnings
from pathlib import Path
from typing import Any, Dict

import hydra
from lightning import Fabric
from lightning.fabric.strategies import STRATEGY_REGISTRY, DDPStrategy, SingleDeviceStrategy, Strategy
from omegaconf import DictConfig, OmegaConf, open_dict

from sheeprl.utils.logger import get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import algorithm_registry, evaluation_registry
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import dotdict, print_config, register_model_from_checkpoint


def resume_from_checkpoint(cfg: DictConfig) -> Dict[str, Any]:
    root_dir = cfg.root_dir
    run_name = cfg.run_name
    ckpt_path = pathlib.Path(cfg.checkpoint.resume_from)
    old_cfg = OmegaConf.load(ckpt_path.parent.parent.parent / ".hydra" / "config.yaml")
    old_cfg = dotdict(OmegaConf.to_container(old_cfg, resolve=True, throw_on_missing=True))
    if old_cfg.env.id != cfg.env.id:
        raise ValueError(
            "This experiment is run with a different environment from the one of the experiment you want to restart. "
            f"Got '{cfg.env.id}', but the environment of the experiment of the checkpoint was {old_cfg.env.id}. "
            "Set properly the environment for restarting the experiment."
        )
    if old_cfg.algo.name != cfg.algo.name:
        raise ValueError(
            "This experiment is run with a different algorithm from the one of the experiment you want to restart. "
            f"Got '{cfg.algo.name}', but the algorithm of the experiment of the checkpoint was {old_cfg.algo.name}. "
            "Set properly the algorithm name for restarting the experiment."
        )
    old_cfg.pop("root_dir", None)
    old_cfg.pop("run_name", None)
    cfg = dotdict(old_cfg)
    cfg.checkpoint.resume_from = str(ckpt_path)
    cfg.root_dir = root_dir
    cfg.run_name = run_name
    return cfg


def run_algorithm(cfg: Dict[str, Any]):
    """Run the algorithm specified in the configuration.

    Args:
        cfg (Dict[str, Any]): the loaded configuration.
    """
    # Given the algorithm's name, retrieve the module where
    # 'cfg.algo.name'.py is contained; from there retrieve the
    # 'register_algorithm'-decorated entrypoint;
    # the entrypoint will be launched by Fabric with 'fabric.launch(entrypoint)'
    module = None
    decoupled = False
    entrypoint = None
    algo_name = cfg.algo.name
    for _module, _algos in algorithm_registry.items():
        for _algo in _algos:
            if algo_name == _algo["name"]:
                module = _module
                entrypoint = _algo["entrypoint"]
                decoupled = _algo["decoupled"]
                break
    if module is None:
        raise RuntimeError(f"Given the algorithm named '{algo_name}', no module has been found to be imported.")
    if entrypoint is None:
        raise RuntimeError(
            f"Given the module and algorithm named '{module}' and '{algo_name}' respectively, "
            "no entrypoint has been found to be imported."
        )
    task = importlib.import_module(f"{module}.{algo_name}")
    utils = importlib.import_module(f"{module}.utils")
    command = task.__dict__[entrypoint]
    kwargs = {}
    if decoupled:
        fabric: Fabric = hydra.utils.instantiate(cfg.fabric, _convert_="all")
        logger = get_logger(fabric, cfg)
        if logger and fabric.is_global_zero:
            fabric._loggers = [logger]
            fabric.logger.log_hyperparams(cfg)
    else:
        strategy = cfg.fabric.pop("strategy", "auto")
        if "sac_ae" in module:
            if strategy is not None:
                warnings.warn(
                    "You are running the SAC-AE algorithm you have specified a strategy different than 'ddp': "
                    f"'python sheeprl.py fabric.strategy={strategy}'. This algorithm is run with the "
                    "'lightning.fabric.strategies.DDPStrategy' strategy."
                )
            strategy = DDPStrategy(find_unused_parameters=True)
        elif "finetuning" in algo_name and "p2e" in module:
            # Load exploration configurations
            ckpt_path = pathlib.Path(cfg.checkpoint.exploration_ckpt_path)
            exploration_cfg = OmegaConf.load(ckpt_path.parent.parent.parent / ".hydra" / "config.yaml")
            exploration_cfg = dotdict(OmegaConf.to_container(exploration_cfg, resolve=True, throw_on_missing=True))
            if exploration_cfg.env.id != cfg.env.id:
                raise ValueError(
                    "This experiment is run with a different environment from "
                    "the one of the exploration you want to finetune. "
                    f"Got '{cfg.env.id}', but the environment used during exploration was {exploration_cfg.env.id}. "
                    "Set properly the environment for finetuning the experiment."
                )
            # Take environment configs from exploration
            cfg.env.frame_stack = exploration_cfg.env.frame_stack
            cfg.env.screen_size = exploration_cfg.env.screen_size
            cfg.env.action_repeat = exploration_cfg.env.action_repeat
            cfg.env.grayscale = exploration_cfg.env.grayscale
            cfg.env.clip_rewards = exploration_cfg.env.clip_rewards
            cfg.env.frame_stack_dilation = exploration_cfg.env.frame_stack_dilation
            cfg.env.max_episode_steps = exploration_cfg.env.max_episode_steps
            cfg.env.reward_as_observation = exploration_cfg.env.reward_as_observation
            _env_target = cfg.env.wrapper._target_.lower()
            if "minerl" in _env_target or "minedojo" in _env_target:
                cfg.env.max_pitch = exploration_cfg.env.max_pitch
                cfg.env.min_pitch = exploration_cfg.env.min_pitch
                cfg.env.sticky_jump = exploration_cfg.env.sticky_jump
                cfg.env.sticky_attack = exploration_cfg.env.sticky_attack
                cfg.env.break_speed_multiplier = exploration_cfg.env.break_speed_multiplier
            kwargs["exploration_cfg"] = exploration_cfg
            cfg.fabric = exploration_cfg.fabric
            strategy = cfg.fabric.pop("strategy", "auto")
        fabric: Fabric = hydra.utils.instantiate(cfg.fabric, strategy=strategy, _convert_="all")

    if hasattr(cfg, "metric") and cfg.metric is not None:
        predefined_metric_keys = set()
        if not hasattr(utils, "AGGREGATOR_KEYS"):
            warnings.warn(
                f"No 'AGGREGATOR_KEYS' set found for the {algo_name} algorithm under the {module} module. "
                "No metric will be logged.",
                UserWarning,
            )
        else:
            predefined_metric_keys = utils.AGGREGATOR_KEYS
        timer.disabled = cfg.metric.log_level == 0 or cfg.metric.disable_timer
        keys_to_remove = set(cfg.metric.aggregator.metrics.keys()) - predefined_metric_keys
        for k in keys_to_remove:
            cfg.metric.aggregator.metrics.pop(k, None)
        MetricAggregator.disabled = cfg.metric.log_level == 0 or len(cfg.metric.aggregator.metrics) == 0

    # Model Manager
    if hasattr(cfg, "model_manager") and not cfg.model_manager.disabled and cfg.model_manager.models is not None:
        predefined_models_keys = set()
        if not hasattr(utils, "MODELS_TO_REGISTER"):
            warnings.warn(
                f"No 'MODELS_TO_REGISTER' set found for the {algo_name} algorithm under the {module} module. "
                "No model will be registered.",
                UserWarning,
            )
        else:
            predefined_models_keys = utils.MODELS_TO_REGISTER
        keys_to_remove = set(cfg.model_manager.models.keys()) - predefined_models_keys
        for k in keys_to_remove:
            cfg.model_manager.models.pop(k, None)
        cfg.model_manager.disabled == cfg.model_manager.disabled or len(cfg.model_manager.models) == 0
    fabric.launch(command, cfg, **kwargs)


def eval_algorithm(cfg: DictConfig):
    """Run the algorithm specified in the configuration.

    Args:
        cfg (DictConfig): the loaded configuration.
    """
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    # TODO: change the number of devices when FSDP will be supported
    accelerator = cfg.fabric.get("accelerator", "auto")
    fabric: Fabric = hydra.utils.instantiate(
        cfg.fabric, accelerator=accelerator, devices=1, num_nodes=1, _convert_="all"
    )

    # Load the checkpoint
    state = fabric.load(cfg.checkpoint_path)

    # Given the algorithm's name, retrieve the module where
    # 'cfg.algo.name'.py is contained; from there retrieve the
    # `register_algorithm`-decorated entrypoint;
    # the entrypoint will be launched by Fabric with `fabric.launch(entrypoint)`
    module = None
    entrypoint = None
    algo_name = cfg.algo.name
    for _module, _algos in evaluation_registry.items():
        for _algo in _algos:
            if algo_name == _algo["name"]:
                module = _module
                entrypoint = _algo["entrypoint"]
                break
    if module is None:
        raise RuntimeError(f"Given the algorithm named `{algo_name}`, no module has been found to be imported.")
    if entrypoint is None:
        raise RuntimeError(
            f"Given the module and algorithm named `{module}` and `{algo_name}` respectively, "
            "no entrypoint has been found to be imported."
        )
    task = importlib.import_module(f"{module}.evaluate")
    command = task.__dict__[entrypoint]
    fabric.launch(command, cfg, state)


def check_configs(cfg: Dict[str, Any]):
    """Check the validity of the configuration.

    Args:
        cfg (Dict[str, Any]): the loaded configuration to check.
    """
    decoupled = False
    algo_name = cfg.algo.name
    for _, _algos in algorithm_registry.items():
        for _algo in _algos:
            if algo_name == _algo["name"]:
                decoupled = _algo["decoupled"]
                break
    strategy = cfg.fabric.strategy
    available_strategies = STRATEGY_REGISTRY.available_strategies()
    if decoupled:
        if isinstance(strategy, str):
            strategy = strategy.lower()
            if not (strategy in available_strategies and "ddp" in strategy):
                raise ValueError(
                    f"{strategy} is currently not supported for decoupled algorithm. "
                    "Please launch the script with a DDP strategy: "
                    "'python sheeprl.py fabric.strategy=ddp'"
                )
        elif (
            "_target_" in strategy
            and issubclass((strategy := hydra.utils.get_class(strategy._target_)), Strategy)
            and not issubclass(strategy, DDPStrategy)
        ):
            raise ValueError(
                f"{strategy.__qualname__} is currently not supported for decoupled algorithms. "
                "Please launch the script with a 'DDP' strategy with 'python sheeprl.py fabric.strategy=ddp'"
            )
    else:
        if isinstance(strategy, str):
            strategy = strategy.lower()
            if strategy != "auto" and not (strategy in available_strategies and "ddp" in strategy):
                warnings.warn(
                    f"Running an algorithm with a strategy ({strategy}) "
                    "different than 'auto' or 'dpp' can cause unexpected problems. "
                    "Please launch the script with a 'DDP' strategy with 'python sheeprl.py fabric.strategy=ddp' "
                    "or the 'auto' one with 'python sheeprl.py fabric.strategy=auto' if you run into any problems.",
                    UserWarning,
                )
        elif (
            "_target_" in strategy
            and issubclass((strategy := hydra.utils.get_class(strategy._target_)), Strategy)
            and not issubclass(strategy, (DDPStrategy, SingleDeviceStrategy))
        ):
            warnings.warn(
                f"Running an algorithm with a strategy ({strategy.__qualname__}) "
                "different than 'SingleDeviceStrategy' or 'DDPStrategy' can cause unexpected problems. "
                "Please launch the script with a 'DDP' strategy with 'python sheeprl.py fabric.strategy=ddp' "
                "or with a single device with 'python sheeprl.py fabric.strategy=auto fabric.devices=1' "
                "if you run into any problems.",
                UserWarning,
            )


def check_configs_evaluation(cfg: DictConfig):
    if cfg.checkpoint_path is None:
        raise ValueError("You must specify the evaluation checkpoint path")


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def run(cfg: DictConfig):
    """SheepRL zero-code command line utility."""
    print_config(cfg)
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    if cfg.checkpoint.resume_from:
        cfg = resume_from_checkpoint(cfg)
    check_configs(cfg)
    run_algorithm(cfg)


@hydra.main(version_base="1.3", config_path="configs", config_name="eval_config")
def evaluation(cfg: DictConfig):
    # Load the checkpoint configuration
    checkpoint_path = Path(cfg.checkpoint_path)
    ckpt_cfg = OmegaConf.load(checkpoint_path.parent.parent.parent / ".hydra" / "config.yaml")

    # Merge the two configs
    with open_dict(cfg):
        capture_video = getattr(cfg.env, "capture_video", True)
        cfg.env = {"capture_video": capture_video, "num_envs": 1}
        cfg.exp = {}
        cfg.algo = {}
        cfg.fabric = {
            "devices": 1,
            "num_nodes": 1,
            "strategy": "auto",
            "accelerator": getattr(cfg.fabric, "accelerator", "auto"),
        }

        # Merge configs
        ckpt_cfg.merge_with(cfg)

        # Update values after merge
        run_name = Path(
            os.path.join(
                os.path.basename(checkpoint_path.parent.parent.parent),
                os.path.basename(checkpoint_path.parent.parent),
                "evaluation",
            )
        )
        ckpt_cfg.run_name = str(run_name)

    # Check the validity of the configuration and run the evaluation
    check_configs_evaluation(ckpt_cfg)
    eval_algorithm(ckpt_cfg)


@hydra.main(version_base="1.3", config_path="configs", config_name="model_manager_config")
def registration(cfg: DictConfig):
    checkpoint_path = Path(cfg.checkpoint_path)
    ckpt_cfg = OmegaConf.load(checkpoint_path.parent.parent.parent / ".hydra" / "config.yaml")

    # Merge the two configs
    with open_dict(cfg):
        cfg.env = ckpt_cfg.env
        cfg.exp_name = ckpt_cfg.exp_name
        cfg.algo = ckpt_cfg.algo
        cfg.distribution = ckpt_cfg.distribution
        cfg.seed = ckpt_cfg.seed

    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    cfg.to_log = dotdict(OmegaConf.to_container(ckpt_cfg, resolve=True, throw_on_missing=True))

    precision = getattr(ckpt_cfg.fabric, "precision", None)
    fabric = Fabric(devices=1, accelerator="cpu", num_nodes=1, precision=precision)

    # Load the checkpoint
    state = fabric.load(cfg.checkpoint_path)
    # Retrieve the algorithm name, used to import the custom
    # log_models_from_checkpoint function.
    algo_name = cfg.algo.name
    if "decoupled" in cfg.algo.name:
        algo_name = algo_name.replace("_decoupled", "")
    if algo_name.startswith("p2e_dv"):
        algo_name = "_".join(algo_name.split("_")[:2])
    try:
        log_models_from_checkpoint = importlib.import_module(
            f"sheeprl.algos.{algo_name}.utils"
        ).log_models_from_checkpoint
    except Exception as e:
        print(e)
        raise RuntimeError(
            f"Make sure that the algorithm is defined in the `./sheeprl/algos/{algo_name}` folder "
            "and that the `log_models_from_checkpoint` function is defined "
            f"in the `./sheeprl/algos/{algo_name}/utils.py` file."
        )

    fabric.launch(register_model_from_checkpoint, cfg, state, log_models_from_checkpoint)
