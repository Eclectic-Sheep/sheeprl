import numpy as np
import torch

from sheeprl.algos.muzero.utils import scalar_to_support, support_to_scalar
from sheeprl.utils.utils import inverse_symsqrt, symsqrt


def test_symsqrt_on_0():
    a = np.array(0.0)
    b = symsqrt(a)
    assert b == 0.0


def test_symsqrt_on_3():
    a = np.array(3.0)
    b = symsqrt(a)
    assert b >= 0.9
    assert b <= 1.1


def test_inverse_symsqrt_on_0():
    a = np.array([0.0])
    b = inverse_symsqrt(a)
    assert b == 0.0


def test_inverting_symsqrt():
    a = torch.tensor(3.0)
    b = inverse_symsqrt(symsqrt(a))
    assert np.allclose(a, b, atol=1e-3)


def test_scalar_to_support():
    a = np.array([0.0])
    b = scalar_to_support(a, support_range=20)
    assert np.argmax(b) == 20
    assert b[..., 20] == 1.0


def test_support_scalar_inversion():
    a = np.array([0.0])
    b = support_to_scalar(scalar_to_support(a, support_range=20), support_range=20)
    assert np.allclose(a, b)


def test_support_scalar_inversion_batch_size():
    a = np.random.rand(32, 1)
    b = support_to_scalar(scalar_to_support(a, support_range=20), support_range=20)
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=1e-3)
