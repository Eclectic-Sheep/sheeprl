import platform

from lightning_utilities.core.imports import RequirementCache

_IS_ATARI_AVAILABLE = RequirementCache("gymnasium[atari]")
_IS_ATARI_ROMS_AVAILABLE = RequirementCache("gymnasium[accept-rom-license]")
_IS_DMC_AVAILABLE = RequirementCache("dm_control==1.0.11")
_IS_TORCH_GREATER_EQUAL_2_0 = RequirementCache("torch>=2.0")
_IS_WINDOWS = platform.system() == "Windows"
_IS_TRANSFORMERS_AVAILABLE = RequirementCache("transformers")
_IS_DATASETS_AVAILABLE = RequirementCache("datasets")
_IS_GRADIO_AVAILABLE = RequirementCache("gradio")

