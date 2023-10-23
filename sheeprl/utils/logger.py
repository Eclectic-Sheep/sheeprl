import os
import warnings
from typing import Any, Dict, Optional, Tuple

from lightning import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem


def create_tensorboard_logger(fabric: Fabric, cfg: Dict[str, Any]) -> Tuple[Optional[TensorBoardLogger]]:
    # Set logger only on rank-0 but share the logger directory: since we don't know
    # what is happening during the `fabric.save()` method, at least we assure that all
    # ranks save under the same named folder.
    # As a plus, rank-0 sets the time uniquely for everyone
    logger = None
    if fabric.is_global_zero:
        root_dir = os.path.join("logs", "runs", cfg.root_dir)
        if cfg.metric.log_level > 0:
            logger = TensorBoardLogger(root_dir=root_dir, name=cfg.run_name)
    return logger


def get_log_dir(fabric: Fabric, root_dir: str, run_name: str, share: bool = True) -> str:
    world_collective = TorchCollective()
    if fabric.world_size > 1 and share:
        world_collective.setup()
        world_collective.create_group()
    if fabric.is_global_zero:
        save_dir = os.path.join("logs", "runs", root_dir, run_name)
        fs = get_filesystem(root_dir)
        try:
            listdir_info = fs.listdir(save_dir)
            existing_versions = []
            for listing in listdir_info:
                d = listing["name"]
                bn = os.path.basename(d)
                if _is_dir(fs, d) and bn.startswith("version_"):
                    dir_ver = bn.split("_")[1].replace("/", "")
                    existing_versions.append(int(dir_ver))
            if len(existing_versions) == 0:
                version = 0
            else:
                version = max(existing_versions) + 1
            log_dir = os.path.join(save_dir, f"version_{version}")
        except OSError:
            warnings.warn("Missing logger folder: %s", save_dir)
            log_dir = os.path.join(save_dir, f"version_{0}")

        os.makedirs(log_dir, exist_ok=True)
        if fabric.world_size > 1 and share:
            world_collective.broadcast_object_list([log_dir], src=0)
    else:
        data = [None]
        world_collective.broadcast_object_list(data, src=0)
        log_dir = data[0]
    return log_dir
