from sheeprl.utils.imports import _IS_LIGHTNING_GREATER_EQUAL_2_1, _IS_TORCH_GREATER_EQUAL_2_1

if not _IS_LIGHTNING_GREATER_EQUAL_2_1:
    raise ModuleNotFoundError(_IS_LIGHTNING_GREATER_EQUAL_2_1)
if not _IS_TORCH_GREATER_EQUAL_2_1:
    raise ModuleNotFoundError(_IS_TORCH_GREATER_EQUAL_2_1)
