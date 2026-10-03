from types import SimpleNamespace

import pytest

from sheeprl.algos.dreamer_v2.utils import build_buffer
from sheeprl.data.buffers import EnvIndependentReplayBuffer, EpisodeBuffer
from sheeprl.utils.utils import dotdict


@pytest.mark.parametrize("buffer_type", ["sequential", "episode"])
def test_the_buffer_holds_buffer_size_steps_of_the_process(buffer_type, tmp_path):
    # The episode buffer, shared by the environments of the process, held `buffer.size` divided by their number
    cfg = dotdict(
        {
            "dry_run": False,
            "seed": 0,
            "buffer": {"size": 1000, "type": buffer_type, "memmap": False, "prioritize_ends": False},
            "env": {"num_envs": 4},
            "algo": {"per_rank_sequence_length": 5, "cnn_keys": {"encoder": []}, "mlp_keys": {"encoder": ["state"]}},
        }
    )
    buffer = build_buffer(SimpleNamespace(world_size=2, global_rank=0), cfg, str(tmp_path), dry_run_size=2)
    if buffer_type == "episode":
        assert isinstance(buffer, EpisodeBuffer) and buffer.buffer_size == 500
    else:
        assert isinstance(buffer, EnvIndependentReplayBuffer) and buffer.buffer_size == 125
