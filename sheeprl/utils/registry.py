import sys
from typing import Any, Callable, Dict, List

# Mapping of tasks with their relative algorithms.
# A new task can be added as: tasks[module] = [..., algorithm]
# where `module` and `algorithm` are respectively taken from sheeprl/algos/{module}/{algorithm}.py
tasks: Dict[str, List[str]] = {}

# A list representing the `decoupled` algorithms
decoupled_tasks: List[str] = []


def _register(fn: Callable[..., Any], decoupled: bool = False) -> Callable[..., Any]:
    # lookup containing module
    if fn.__module__ == "__main__":
        return fn
    module_split = fn.__module__.split(".")
    module, algorithm = module_split[-2:]
    algos = tasks.get(module, None)
    if algos is None:
        tasks[module] = [algorithm]
    else:
        if algorithm in algos:
            raise ValueError(f"The algorithm `{algorithm}` has already been registered!")
        tasks[module].append(algorithm)
    if decoupled or "decoupled" in algorithm:
        decoupled_tasks.append(algorithm)

    # add the decorated function to __all__ in algorithm
    entrypoint = fn.__name__
    mod = sys.modules[fn.__module__]
    if hasattr(mod, "__all__"):
        mod.__all__.append(entrypoint)
    else:
        mod.__all__ = [entrypoint]
    return fn


def register_algorithm(decoupled: bool = False):
    def inner_decorator(fn):
        return _register(fn, decoupled=decoupled)

    return inner_decorator
