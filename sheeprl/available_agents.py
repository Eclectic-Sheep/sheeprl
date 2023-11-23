if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    from sheeprl.utils.registry import algorithm_registry, evaluation_registry

    table = Table(title="SheepRL Agents")
    table.add_column("Module")
    table.add_column("Algorithm")
    table.add_column("Entrypoint")
    table.add_column("Decoupled")
    table.add_column("Evaluated by")

    for module, implementations in algorithm_registry.items():
        for algo in implementations:
            evaluation_entrypoint = "Undefined"
            for evaluation in evaluation_registry[module]:
                if algo["name"] == evaluation["name"]:
                    evaluation_entrypoint = evaluation["entrypoint"]
                    break
            table.add_row(
                module,
                algo["name"],
                algo["entrypoint"],
                str(algo["decoupled"]),
                module + ".evaluate." + evaluation_entrypoint,
            )

    console = Console()
    console.print(table)
