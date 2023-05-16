import platform

from lightning_utilities.core.imports import RequirementCache

_IS_WINDOWS = platform.system() == "Windows"
_IS_ATARI_AVAILABLE = RequirementCache("gymnasium[atari]")
_IS_ATARI_ROMS_AVAILABLE = RequirementCache("gymnasium[accept-rom-license]")
