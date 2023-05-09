import os

import pytest


@pytest.fixture(autouse=True)
def hide_gpus():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
