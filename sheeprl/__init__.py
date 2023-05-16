from dotenv import load_dotenv

from sheeprl.algos.droq import droq
from sheeprl.algos.ppo import ppo, ppo_decoupled
from sheeprl.algos.ppo_continuous import ppo_continuous
from sheeprl.algos.ppo_recurrent import ppo_recurrent
from sheeprl.algos.sac import sac, sac_decoupled

try:
    from sheeprl.algos.ppo import ppo_atari
except ModuleNotFoundError:
    pass

load_dotenv()
