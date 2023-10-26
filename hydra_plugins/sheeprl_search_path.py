"""Thank you @rcmalli: https://github.com/orobix/quadra/blob/main/hydra_plugins/quarda_searchpath_plugin.py"""

import os

import dotenv
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class SheepRLSearchPathPlugin(SearchPathPlugin):
    """Generic Search Path Plugin class."""

    def __init__(self):
        try:
            os.getcwd()
        except FileNotFoundError:
            # This may happen when running tests
            return

        if os.path.exists(os.path.join(os.getcwd(), ".env")):
            dotenv.load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=True)

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Plugin used to add custom config to searchpath to be discovered by sheeprl."""
        # This can be global or taken from the .env
        sheeprl_search_path = os.environ.get("SHEEPRL_SEARCH_PATH", None)

        # Path should be specified as a list of hydra path separated by ";"
        # E.g pkg://package1.configs;file:///path/to/configs
        if sheeprl_search_path is not None:
            for i, path in enumerate(sheeprl_search_path.split(";")):
                search_path.append(provider=f"sheeprl-searchpath-plugin-{i}", path=path)
