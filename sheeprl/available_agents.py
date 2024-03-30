from rich.console import Console
from rich.table import Table

from sheeprl.utils.registry import algorithm_registry, evaluation_registry


def available_agents():
    table = Table(title="SheepRL Agents")
    table.add_column("Module")
    table.add_column("Algorithm")
    table.add_column("Entrypoint")
    table.add_column("Decoupled")
    table.add_column("Evaluated by")

    for module, implementations in algorithm_registry.items():
        for algo in implementations:
            evaluated_by = "Undefined"
            if module in evaluation_registry:
                for evaluation in evaluation_registry[module]:
                    if algo["name"] == evaluation["name"]:
                        evaluation_file = evaluation["evaluation_file"]
                        evaluation_entrypoint = evaluation["entrypoint"]
                        evaluated_by = module + "." + evaluation_file + "." + evaluation_entrypoint
                        break
            table.add_row(
                module,
                algo["name"],
                algo["entrypoint"],
                str(algo["decoupled"]),
                evaluated_by,
            )

    console = Console()
    console.print(table)


if __name__ == "__main__":
    available_agents()
