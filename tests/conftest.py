import os

import pytest


@pytest.fixture(autouse=True)
def hide_gpus():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["SHEEPRL_SEARCH_PATH"] = "file://tests/configs;pkg://sheeprl.configs"
