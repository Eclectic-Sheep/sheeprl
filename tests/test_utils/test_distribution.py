"""The truncated normal of the DreamerV2 actor: its bounds are validated only with the validation of the arguments."""

import pytest
import torch

from sheeprl.utils.distribution import TruncatedNormal


def test_the_truncated_normal_reads_no_tensor_on_the_host_without_validation(monkeypatch):
    # It checked its bounds on the host at every construction: a synchronization with the device at every step of the
    # imagination of DreamerV2, and a break of its compiled graph
    def host_read(self, *args, **kwargs):
        raise AssertionError("A tensor read on the host")

    loc, scale = torch.zeros(3, 2), torch.ones(3, 2)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", host_read)
        patch.setattr(torch.Tensor, "item", host_read)
        dist = TruncatedNormal(loc, scale, -1, 1, validate_args=False)
    assert dist.mean.shape == (3, 2)


def test_the_truncated_normal_validates_its_bounds_with_the_validation_of_the_arguments():
    with pytest.raises(ValueError, match="Incorrect truncation range"):
        TruncatedNormal(torch.zeros(2), torch.ones(2), 1, -1, validate_args=True)
