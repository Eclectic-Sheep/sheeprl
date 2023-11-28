import os

ROOT_DIR = os.path.dirname(__file__)

import decorator
from dotenv import load_dotenv

load_dotenv()

from sheeprl.utils.imports import _IS_TORCH_GREATER_EQUAL_2_0

if not _IS_TORCH_GREATER_EQUAL_2_0:
    raise ModuleNotFoundError(_IS_TORCH_GREATER_EQUAL_2_0)

import numpy as np

# fmt: off
from sheeprl.algos.dreamer_v1 import dreamer_v1  # noqa: F401
from sheeprl.algos.dreamer_v2 import dreamer_v2  # noqa: F401
from sheeprl.algos.dreamer_v3 import dreamer_v3  # noqa: F401
from sheeprl.algos.droq import droq  # noqa: F401
from sheeprl.algos.p2e_dv1 import p2e_dv1_exploration  # noqa: F401
from sheeprl.algos.p2e_dv1 import p2e_dv1_finetuning  # noqa: F401
from sheeprl.algos.p2e_dv2 import p2e_dv2_exploration  # noqa: F401
from sheeprl.algos.p2e_dv2 import p2e_dv2_finetuning  # noqa: F401
from sheeprl.algos.p2e_dv3 import p2e_dv3_exploration  # noqa: F401
from sheeprl.algos.p2e_dv3 import p2e_dv3_finetuning  # noqa: F401
from sheeprl.algos.ppo import ppo  # noqa: F401
from sheeprl.algos.ppo import ppo_decoupled  # noqa: F401
from sheeprl.algos.ppo_recurrent import ppo_recurrent  # noqa: F401
from sheeprl.algos.sac import sac  # noqa: F401
from sheeprl.algos.sac import sac_decoupled  # noqa: F401
from sheeprl.algos.sac_ae import sac_ae  # noqa: F401

from sheeprl.algos.dreamer_v1 import evaluate as dreamer_v1_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.dreamer_v2 import evaluate as dreamer_v2_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.dreamer_v3 import evaluate as dreamer_v3_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.droq import evaluate as droq_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.p2e_dv1 import evaluate as p2e_dv1_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.p2e_dv2 import evaluate as p2e_dv2_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.p2e_dv3 import evaluate as p2e_dv3_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.ppo import evaluate as ppo_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.ppo_recurrent import evaluate as ppo_recurrent_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.sac import evaluate as sac_evaluate  # noqa: F401, isort:skip
from sheeprl.algos.sac_ae import evaluate as sac_ae_evaluate  # noqa: F401, isort:skip
# fmt: on

# Needed because MineRL 0.4.4 is not compatible with the latest version of numpy
np.float = np.float32
np.int = np.int64
np.bool = bool

__version__ = "0.4.8"


# Replace `moviepy.decorators.use_clip_fps_by_default` method to work with python 3.8, 3.9, and 3.10
import moviepy.decorators


# Taken from https://github.com/Zulko/moviepy/blob/master/moviepy/decorators.py#L118
@decorator.decorator
def custom_use_clip_fps_by_default(func, clip, *args, **kwargs):
    """Will use ``clip.fps`` if no ``fps=...`` is provided in **kwargs**."""
    import inspect

    def find_fps(fps):
        if fps is not None:
            return fps
        elif getattr(clip, "fps", None):
            return clip.fps
        raise AttributeError(
            "No 'fps' (frames per second) attribute specified"
            " for function %s and the clip has no 'fps' attribute. Either"
            " provide e.g. fps=24 in the arguments of the function, or define"
            " the clip's fps with `clip.fps=24`" % func.__name__
        )

    names = inspect.getfullargspec(func).args[1:]

    new_args = [find_fps(arg) if (name == "fps") else arg for (arg, name) in zip(args, names)]
    new_kwargs = {kwarg: find_fps(value) if kwarg == "fps" else value for (kwarg, value) in kwargs.items()}

    return func(clip, *new_args, **new_kwargs)


moviepy.decorators.use_clip_fps_by_default = custom_use_clip_fps_by_default
