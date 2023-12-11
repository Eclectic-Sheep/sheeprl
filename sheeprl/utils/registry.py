from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List

# Mapping of tasks with their relative algorithms.
# A new task can be added as:
# tasks[module] = [..., {"name": algorithm, "entrypoint": entrypoint, "decoupled": decoupled}]
# where `module` and `algorithm` are respectively taken from sheeprl/algos/{module}/{algorithm}.py,
# while `entrypoint` is the decorated function
algorithm_registry: Dict[str, List[Dict[str, Any]]] = {}
evaluation_registry: Dict[str, List[Dict[str, Any]]] = {}


def _register_algorithm(fn: Callable[..., Any], decoupled: bool = False) -> Callable[..., Any]:
    # lookup containing module
    if fn.__module__ == "__main__":
        return fn
    entrypoint = fn.__name__
    module_split = fn.__module__.split(".")
    algorithm = module_split[-1]
    module = ".".join(module_split[:-1])
    registered_algos = algorithm_registry.get(module, None)
    if registered_algos is None:
        algorithm_registry[module] = [{"name": algorithm, "entrypoint": entrypoint, "decoupled": decoupled}]
    else:
        algorithm_registry[module].append({"name": algorithm, "entrypoint": entrypoint, "decoupled": decoupled})

    # add the decorated function to __all__ in algorithm
    mod = sys.modules[fn.__module__]
    if hasattr(mod, "__all__"):
        mod.__all__.append(entrypoint)
    else:
        mod.__all__ = [entrypoint]
    return fn


def _register_evaluation(fn: Callable[..., Any], algorithms: str | List[str]) -> Callable[..., Any]:
    # lookup containing module
    if fn.__module__ == "__main__":
        return fn
    entrypoint = fn.__name__
    module_split = fn.__module__.split(".")
    module = ".".join(module_split[:-1])
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    # Check that the algorithms which we want to register an evaluation function for
    # have been registered as algorithms
    registered_algos = algorithm_registry.get(module, None)
    if registered_algos is None:
        raise ValueError(
            f"The evaluation function `{module+'.'+entrypoint}` for the algorithms named `{', '.join(algorithms)}` "
            "is going to be registered, but no algorithm has been registered!"
        )
    registered_algo_names = {algo["name"] for algo in registered_algos}
    if len(set(algorithms) - registered_algo_names) > 0:
        raise ValueError(
            f"You are trying to register the evaluation function `{module+'.'+entrypoint}` "
            f"for algorithms which have not been registered for the module `{module}`!\n"
            f"Registered algorithms: {', '.join(registered_algo_names)}\n"
            f"Specified algorithms: {', '.join(algorithms)}"
        )
    registered_evals = evaluation_registry.get(module, None)
    if registered_evals is None:
        evaluation_registry[module] = []
        for algorithm in algorithms:
            evaluation_registry[module].append({"name": algorithm, "entrypoint": entrypoint})
    else:
        for registered_eval in registered_evals:
            if registered_eval["name"] in algorithms:
                raise ValueError(
                    f"Cannot register the evaluate function `{module+'.'+entrypoint}` "
                    f"for the algorithm `{registered_eval['name']}`: "
                    f"the evaluation function `{module+'.'+registered_eval['entrypoint']}` has already "
                    f"been registered for the algorithm named `{registered_eval['name']}` in the module `{module}`!"
                )
        evaluation_registry[module].extend([{"name": algorithm, "entrypoint": entrypoint} for algorithm in algorithms])

    # add the decorated function to __all__ in algorithm
    mod = sys.modules[fn.__module__]
    if hasattr(mod, "__all__"):
        mod.__all__.append(entrypoint)
    else:
        mod.__all__ = [entrypoint]
    return fn


def register_algorithm(decoupled: bool = False):
    def inner_decorator(fn):
        return _register_algorithm(fn, decoupled=decoupled)

    return inner_decorator


def register_evaluation(algorithms: str | List[str]):
    def inner_decorator(fn):
        return _register_evaluation(fn, algorithms=algorithms)

    return inner_decorator
