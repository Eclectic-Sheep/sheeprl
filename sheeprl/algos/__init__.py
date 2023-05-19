from datetime import timedelta

from sheeprl.utils.imports import _IS_WINDOWS

if _IS_WINDOWS:
    default_pg_timeout = timedelta(days=1)
else:
    from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
