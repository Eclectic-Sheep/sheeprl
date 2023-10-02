import sys
from typing import Any, Callable, Dict, List

# Mapping of tasks with their relative algorithms.
# A new task can be added as:
# tasks[module] = [..., {"name": algorithm, "entrypoint": entrypoint, "decoupled": decoupled}]
# where `module` and `algorithm` are respectively taken from sheeprl/algos/{module}/{algorithm}.py,
# while `entrypoint` is the decorated function
tasks: Dict[str, List[Dict[str, Any]]] = {}


def _register(fn: Callable[..., Any], decoupled: bool = False) -> Callable[..., Any]:
    # lookup containing module
    if fn.__module__ == "__main__":
        return fn
    entrypoint = fn.__name__
    module_split = fn.__module__.split(".")
    algorithm = module_split[-1]
    module = ".".join(module_split[:-1])
    algos = tasks.get(module, None)
    if algos is None:
        tasks[module] = [{"name": algorithm, "entrypoint": entrypoint, "decoupled": decoupled}]
    else:
        if algorithm in algos:
            raise ValueError(f"The algorithm `{algorithm}` has already been registered!")
        tasks[module].append({"name": algorithm, "entrypoint": entrypoint, "decoupled": decoupled})

    # add the decorated function to __all__ in algorithm
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
