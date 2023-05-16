from dotenv import load_dotenv

from fabricrl.algos.droq import droq
from fabricrl.algos.ppo import ppo, ppo_decoupled
from fabricrl.algos.ppo_continuous import ppo_continuous
from fabricrl.algos.ppo_recurrent import ppo_recurrent
from fabricrl.algos.sac import sac, sac_decoupled

try:
    from fabricrl.algos.ppo import ppo_atari
except ModuleNotFoundError:
    pass

load_dotenv()
