import os

from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.path.dirname(__file__)
# The code compiled by `torch.compile` (`algo.compile`) is cached in the home of the user, where it survives the
# reboots (by default PyTorch caches it in `/tmp`, often cleared at boot): the runs that compile the same code reuse it.
# Set before the algorithms are imported, which already create the directory of the cache
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "sheeprl", "torchinductor")
)


from sheeprl.utils.imports import _IS_TORCH_GREATER_EQUAL_2_0

if not _IS_TORCH_GREATER_EQUAL_2_0:
    raise ModuleNotFoundError(_IS_TORCH_GREATER_EQUAL_2_0)

# fmt: off
from sheeprl.algos.a2c import a2c  # noqa: F401
from sheeprl.algos.dreamer_v1 import dreamer_v1  # noqa: F401
from sheeprl.algos.dreamer_v2 import dreamer_v2  # noqa: F401
from sheeprl.algos.dreamer_v3 import dreamer_v3  # noqa: F401
from sheeprl.algos.dreamer_v3_5 import dreamer_v3_5  # noqa: F401
from sheeprl.algos.droq import droq  # noqa: F401
from sheeprl.algos.p2e_dv1 import p2e_dv1_exploration  # noqa: F401
from sheeprl.algos.p2e_dv1 import p2e_dv1_finetuning  # noqa: F401
from sheeprl.algos.p2e_dv2 import p2e_dv2_exploration  # noqa: F401
from sheeprl.algos.p2e_dv2 import p2e_dv2_finetuning  # noqa: F401
from sheeprl.algos.p2e_dv3 import p2e_dv3_exploration  # noqa: F401
from sheeprl.algos.p2e_dv3 import p2e_dv3_finetuning  # noqa: F401
from sheeprl.algos.ppo import ppo  # noqa: F401
from sheeprl.algos.ppo_recurrent import ppo_recurrent  # noqa: F401
from sheeprl.algos.sac import sac  # noqa: F401
from sheeprl.algos.sac_ae import sac_ae  # noqa: F401

from sheeprl.algos.a2c import evaluate as a2c_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.dreamer_v1 import evaluate as dreamer_v1_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.dreamer_v2 import evaluate as dreamer_v2_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.dreamer_v3 import evaluate as dreamer_v3_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.dreamer_v3_5 import evaluate as dreamer_v3_5_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.droq import evaluate as droq_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.p2e_dv1 import evaluate as p2e_dv1_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.p2e_dv2 import evaluate as p2e_dv2_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.p2e_dv3 import evaluate as p2e_dv3_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.ppo import evaluate as ppo_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.ppo_recurrent import evaluate as ppo_recurrent_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.sac import evaluate as sac_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.sac_ae import evaluate as sac_ae_evaluate  # noqa: F401, isort:skip
# fmt: on

__version__ = "1.0.0rc1"
