import os

from sheeprl.utils.imports import _IS_LIGHTNING_GREATER_EQUAL_2_1, _IS_TORCH_GREATER_EQUAL_2_1

if not _IS_LIGHTNING_GREATER_EQUAL_2_1:
    raise ModuleNotFoundError(_IS_LIGHTNING_GREATER_EQUAL_2_1)

# TODO: ugly, but necessary until we move to PyTorch >=2.1
if (
    os.environ.get("SHEEPRL_TEST", None) is None
    and os.environ["SHEEPRL_TEST"] != "1"
    and not _IS_TORCH_GREATER_EQUAL_2_1
):
    raise ModuleNotFoundError(_IS_TORCH_GREATER_EQUAL_2_1)
