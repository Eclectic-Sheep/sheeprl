import os

from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.path.dirname(__file__)


from sheeprl.utils.imports import _IS_TORCH_GREATER_EQUAL_2_0

if not _IS_TORCH_GREATER_EQUAL_2_0:
    raise ModuleNotFoundError(_IS_TORCH_GREATER_EQUAL_2_0)

import numpy as np

from sheeprl.algos.dreamer_v1 import dreamer_v1 as dreamer_v1
from sheeprl.algos.dreamer_v1 import dreamer_v1_np as dreamer_v1_np
from sheeprl.algos.dreamer_v2 import dreamer_v2 as dreamer_v2
from sheeprl.algos.dreamer_v2 import dreamer_v2_np as dreamer_v2_np
from sheeprl.algos.dreamer_v3 import dreamer_v3 as dreamer_v3
from sheeprl.algos.dreamer_v3 import dreamer_v3_np as dreamer_v3_np
from sheeprl.algos.droq import droq as droq
from sheeprl.algos.muzero import muzero as muzero
from sheeprl.algos.p2e_dv1 import p2e_dv1 as p2e_dv1
from sheeprl.algos.p2e_dv2 import p2e_dv2 as p2e_dv2
from sheeprl.algos.ppo import ppo as ppo
from sheeprl.algos.ppo import ppo_decoupled as ppo_decoupled
from sheeprl.algos.ppo import ppo_np as ppo_np
from sheeprl.algos.ppo_recurrent import ppo_recurrent as ppo_recurrent
from sheeprl.algos.sac import sac as sac
from sheeprl.algos.sac import sac_decoupled as sac_decoupled
from sheeprl.algos.sac import sac_np as sac_np
from sheeprl.algos.sac_ae import sac_ae as sac_ae

np.float = np.float32
np.int = np.int64
np.bool = bool

__version__ = "0.4.4"
