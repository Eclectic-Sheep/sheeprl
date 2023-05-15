import sys
from typing import Any, Callable, Dict, List, Optional

# Mapping of tasks with their relative algorithms.
# A new task can be added as: tasks[module] = [..., algorithm]
# where `module` and `algorithm` are respectively taken from fabricrl/algos/{module}/{algorithm}.py
tasks: Dict[str, List[str]] = {}

# A list representing the `decoupled` algorithms
decoupled_tasks: List[str] = []


def _register(fn: Callable[..., Any], decoupled: bool = False) -> Callable[..., Any]:
    # lookup containing module
    if fn.__module__ == "__main__":
        return fn
    module_name_split = fn.__module__.split(".")
    module_name, algorithm = module_name_split[-2:]
    algos = tasks.get(module_name, None)
    if algos is None:
        tasks[module_name] = [algorithm]
    else:
        if algorithm in algos:
            raise ValueError(f"The algorithm `{algorithm}` has already been registered!")
        tasks[module_name].append(algorithm)
    if decoupled:
        decoupled_tasks.append(algorithm)

    # add the decorated function to __all__ in algorithm
    entrypoint = fn.__name__
    mod = sys.modules[fn.__module__]
    if hasattr(mod, "__all__"):
        mod.__all__.append(entrypoint)
    else:
        mod.__all__ = [entrypoint]
    return fn


def register_algorithm(fn: Optional[Callable[..., Any]] = None, decoupled: bool = False):
    if fn:
        return _register(fn, decoupled)
    return _register
