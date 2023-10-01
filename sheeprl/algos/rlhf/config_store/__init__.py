from hydra.core.config_store import ConfigStore

from sheeprl.algos.rlhf.config_store.algo import register_algo_configs
from sheeprl.algos.rlhf.config_store.data import register_data_configs
from sheeprl.algos.rlhf.config_store.model import register_model_configs


def register_configs() -> None:
    cs = ConfigStore.instance()
    register_algo_configs(cs)
    register_model_configs(cs)
    register_data_configs(cs)
