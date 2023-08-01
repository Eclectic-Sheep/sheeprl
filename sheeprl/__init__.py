import os

ROOT_DIR = os.path.dirname(__file__)

from dotenv import load_dotenv

load_dotenv()

from sheeprl.utils.imports import _IS_TORCH_GREATER_EQUAL_2_0

if not _IS_TORCH_GREATER_EQUAL_2_0:
    raise ModuleNotFoundError(_IS_TORCH_GREATER_EQUAL_2_0)

from sheeprl.algos.dreamer_v1 import dreamer_v1
from sheeprl.algos.dreamer_v2 import dreamer_v2
from sheeprl.algos.droq import droq
from sheeprl.algos.p2e_dv1 import p2e_dv1
from sheeprl.algos.p2e_dv2 import p2e_dv2
from sheeprl.algos.ppo import ppo, ppo_decoupled
from sheeprl.algos.ppo_continuous import ppo_continuous
from sheeprl.algos.ppo_pixel import ppo_pixel_continuous
from sheeprl.algos.ppo_recurrent import ppo_recurrent
from sheeprl.algos.sac import sac, sac_decoupled
from sheeprl.algos.sac_pixel import sac_pixel_continuous
from sheeprl.algos.test import test

try:
    from sheeprl.algos.ppo_pixel import ppo_atari
except ModuleNotFoundError:
    pass

# Needed because MineRL 0.4.4 is not compatible with the latest version of numpy
import numpy as np

np.float = np.float32
np.int = np.int64
np.bool = bool

__version__ = "0.1.0"
