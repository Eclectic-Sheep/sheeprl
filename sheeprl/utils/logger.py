import os
import pathlib
import time
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Tuple

from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective

from sheeprl.algos.args import StandardArgs


def create_tensorboard_logger(
    fabric: Fabric, args: StandardArgs, algo_name: str
) -> Tuple[Optional[TensorBoardLogger], str]:
    # Set logger only on rank-0 but share the logger directory: since we don't know
    # what is happening during the `fabric.save()` method, at least we assure that all
    # ranks save under the same named folder.
    # As a plus, rank-0 sets the time uniquely for everyone
    world_collective = TorchCollective()
    if fabric.world_size > 1:
        world_collective.setup()
        world_collective.create_group()
    if fabric.is_global_zero:
        root_dir = (
            args.root_dir
            if args.root_dir is not None
            else os.path.join("logs", algo_name, datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
        )
        run_name = (
            args.run_name
            if args.run_name is not None
            else f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
        )
        if args.checkpoint_path:
            ckpt_path = pathlib.Path(args.checkpoint_path)
            root_dir = ckpt_path.parent.parent
            run_name = "resume_from_checkpoint"
        logger = TensorBoardLogger(root_dir=root_dir, name=run_name)
        log_dir = logger.log_dir
        fabric.logger.log_hyperparams(asdict(args))
        if fabric.world_size > 1:
            world_collective.broadcast_object_list([log_dir], src=0)

        # Save args as dict automatically
        args.log_dir = log_dir
    else:
        data = [None]
        world_collective.broadcast_object_list(data, src=0)
        logger = None
        log_dir = data[0]
        os.makedirs(log_dir, exist_ok=True)
    return logger, log_dir
