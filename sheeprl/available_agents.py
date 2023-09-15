if __name__ == "__main__":
    from rich.pretty import pprint

    from sheeprl.utils.registry import tasks

    pprint(tasks, expand_all=True)
