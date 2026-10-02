from __future__ import annotations

from typing import Any, Dict

from lightning import Fabric

from sheeprl.algos.dreamer_v2.utils import test
from sheeprl.algos.p2e_dv1.utils import trained_algorithm
from sheeprl.core import load_trained_state
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms=["p2e_dv1_exploration", "p2e_dv1_finetuning"])
def evaluate(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    # The spaces of the environment, to build the agent
    env = make_env(cfg, cfg.seed, 0, log_dir, "test", vector_env_idx=0)()
    algo = trained_algorithm(fabric, cfg)
    trained = load_trained_state(fabric, cfg, algo, state, env.observation_space, env.action_space)
    env.close()
    # The task actor plays
    test(algo.policy(trained, trained.actor_task, "task"), fabric, cfg, log_dir, greedy=True)
