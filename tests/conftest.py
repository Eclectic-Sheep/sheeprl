"""Adapted from: https://github.com/Lightning-AI/lightning/blob/master/tests/tests_fabric/conftest.py"""

import os

import pytest
import torch.distributed


@pytest.fixture(autouse=True)
def preserve_global_rank_variable():
    """Ensures that the rank_zero_only.rank global variable gets reset in each test."""
    from lightning.fabric.utilities.rank_zero import rank_zero_only

    rank = getattr(rank_zero_only, "rank", None)
    yield
    if rank is not None:
        setattr(rank_zero_only, "rank", rank)


@pytest.fixture(autouse=True)
def restore_env_variables():
    """Ensures that environment variables set during the test do not leak out."""
    env_backup = os.environ.copy()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["SHEEPRL_SEARCH_PATH"] = "file://tests/configs;pkg://sheeprl.configs"
    yield
    leaked_vars = os.environ.keys() - env_backup.keys()
    # restore environment as it was before running the test
    os.environ.clear()
    os.environ.update(env_backup)
    # these are currently known leakers - ideally these would not be allowed
    # TODO(fabric): this list can be trimmed, maybe PL's too after moving tests
    allowlist = {
        "CUDA_DEVICE_ORDER",
        "LOCAL_RANK",
        "NODE_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "PL_GLOBAL_SEED",
        "PL_SEED_WORKERS",
        "RANK",  # set by DeepSpeed
        "POPLAR_ENGINE_OPTIONS",  # set by IPUStrategy
        "CUDA_MODULE_LOADING",  # leaked since PyTorch 1.13
        "CRC32C_SW_MODE",  # set by tensorboardX
        "OMP_NUM_THREADS",  # set by our launchers
        # set by XLA FSDP on XRT
        "XRT_TORCH_DIST_ROOT",
        "XRT_MESH_SERVICE_ADDRESS",
        # set by torchdynamo
        "TRITON_CACHE_DIR",
        # set by Pygame
        "SDL_VIDEO_X11_WMCLASS",
        # set by us
        "CUDA_VISIBLE_DEVICES",
        "SHEEPRL_SEARCH_PATH",
        "LT_ACCELERATOR",
        "LT_DEVICES",
    }
    leaked_vars.difference_update(allowlist)
    assert not leaked_vars, f"test is leaking environment variable(s): {set(leaked_vars)}"


@pytest.fixture(autouse=True)
def teardown_process_group():
    """Ensures that the distributed process group gets closed before the next test runs."""
    yield
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def pytest_collection_modifyitems(items):
    """Adds a timeout marker to all tests in test_algos.py."""
    timeout = 60 if os.environ.get("MLFLOW_TRACKING_URI", None) is not None else 180
    for item in items:
        if "test_algos.py" in item.module.__name__:
            item.add_marker(pytest.mark.timeout(timeout))
