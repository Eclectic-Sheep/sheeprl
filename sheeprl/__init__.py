from dotenv import load_dotenv

load_dotenv()

from sheeprl.utils.imports import _IS_TORCH_GREATER_EQUAL_2_0

if not _IS_TORCH_GREATER_EQUAL_2_0:
    raise ModuleNotFoundError(_IS_TORCH_GREATER_EQUAL_2_0)

from sheeprl.algos.droq import droq
from sheeprl.algos.muzero import muzero, muzero_decoupled
from sheeprl.algos.ppo import ppo, ppo_decoupled
from sheeprl.algos.ppo_continuous import ppo_continuous
from sheeprl.algos.ppo_pixel import ppo_pixel_continuous
from sheeprl.algos.ppo_recurrent import ppo_recurrent
from sheeprl.algos.sac import sac, sac_decoupled
from sheeprl.algos.sac_pixel import sac_pixel_continuous

try:
    from sheeprl.algos.ppo_pixel import ppo_atari
except ModuleNotFoundError:
    pass


__version__ = "0.1.0"
