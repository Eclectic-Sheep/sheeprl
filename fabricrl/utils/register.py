from typing import Dict, List

# Mapping of tasks with their relative algorithms.
# A new task can be added as: tasks[module] = [..., algorithm]
# where `module` and `algorithm` are respectively taken from fabricrl/algos/{module}/{algorithm}.py
tasks: Dict[str, List[str]] = {
    "droq": ["droq"],
    "sac": ["sac", "sac_decoupled"],
    "ppo": ["ppo", "ppo_decoupled"],
    "ppo_continuous": ["ppo_continuous"],
    "ppo_recurrent": ["ppo_recurrent"],
}

# A list representing the `decoupled` algorithms
decoupled_tasks: List[str] = ["sac_decoupled", "ppo_decoupled"]
