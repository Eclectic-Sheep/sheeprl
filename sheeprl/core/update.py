"""Device, precision and optimizer steps: the parts of training that are the same for every algorithm.

Modules are not wrapped in `DistributedDataParallel` (`setup_module`): with several processes, `update` averages the
gradients itself, with one all-reduce per optimizer step and only for the weights that step updates. Both live in
`sheeprl.utils.fabric`, with the algorithms that don't use the shared loop.
"""

from __future__ import annotations

from contextlib import AbstractContextManager

from lightning import Fabric

from sheeprl.utils.fabric import all_reduce_gradients, autocast_cache_scope, setup_module, update

__all__ = ["all_reduce_gradients", "autocast", "setup_module", "update"]


def autocast(fabric: Fabric) -> AbstractContextManager:
    """The precision scope of one update phase: the forward passes and the loss of one optimizer step.

    Close it before calling `update`. The modules prepared by `setup_module` run every call in the precision of
    `fabric`; this scope keeps autocast's low-precision copies of the weights for the whole phase, instead of
    casting the weights again at every call (see `sheeprl.utils.fabric.autocast_cache_scope`). The copies are not
    updated by the optimizer: that's why the scope must be closed before the update.
    """
    return autocast_cache_scope(fabric)
