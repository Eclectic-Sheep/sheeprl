if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    from sheeprl.utils.registry import tasks

    table = Table(title="SheepRL Agents")
    table.add_column("Module")
    table.add_column("Algorithm")
    table.add_column("Entrypoint")
    table.add_column("Decoupled")

    for module, implementations in tasks.items():
        for algo in implementations:
            table.add_row(module, algo["name"], algo["entrypoint"], str(algo["decoupled"]))

    console = Console()
    console.print(table)
