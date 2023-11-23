import platform

from lightning_utilities.core.imports import RequirementCache

_IS_ATARI_AVAILABLE = RequirementCache("gymnasium[atari]")
_IS_ATARI_ROMS_AVAILABLE = RequirementCache("gymnasium[accept-rom-license]")
_IS_BOX2D_AVAILABLE = RequirementCache("gymnasium[box2d]")
_IS_CRAFTER_AVAILABLE = RequirementCache("crafter")
_IS_DIAMBRA_AVAILABLE = RequirementCache("diambra")
_IS_DIAMBRA_ARENA_AVAILABLE = RequirementCache("diambra-arena")
_IS_DMC_AVAILABLE = RequirementCache("dm_control")
_IS_MINEDOJO_AVAILABLE = RequirementCache("minedojo")
_IS_MINERL_0_4_4_AVAILABLE = RequirementCache("minerl==0.4.4")
_IS_TORCH_GREATER_EQUAL_2_0 = RequirementCache("torch>=2.0")
_IS_WINDOWS = platform.system() == "Windows"
