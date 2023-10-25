import torch

from sheeprl.utils.utils import two_hot_encoder


def test_standard_case():
    tensor = torch.tensor(2.3)
    result = two_hot_encoder(tensor, 5)
    expected_result = torch.zeros(11)
    expected_result[5 + 2] = 0.7
    expected_result[5 + 3] = 0.3
    assert result.shape == torch.Size([11])
    assert torch.allclose(result, expected_result)


def test_standard_case_more_buckets():
    tensor = torch.tensor(2.3)
    result = two_hot_encoder(tensor, 5, 21)
    expected_result = torch.zeros(21)
    expected_result[10 + 4] = 0.4
    expected_result[10 + 5] = 0.6
    assert result.shape == torch.Size([21])
    assert torch.allclose(result, expected_result)


def test_batch_case():
    tensor = torch.tensor([[2.3], [3.4]])
    result = two_hot_encoder(tensor, 5)
    expected_result = torch.zeros(2, 11)
    expected_result[0, 5 + 2] = 0.7
    expected_result[0, 5 + 3] = 0.3
    expected_result[1, 5 + 3] = 0.6
    expected_result[1, 5 + 4] = 0.4
    assert result.shape == torch.Size([2, 11])
    assert torch.allclose(result, expected_result)


def test_support_size_1():
    tensor = torch.tensor(2.3)
    result = two_hot_encoder(tensor, 0)
    expected_result = torch.tensor([1.0])
    assert result.shape == torch.Size([1])
    assert torch.allclose(result, expected_result)


def test_overflow_support():
    tensor = torch.tensor(6.1)
    result = two_hot_encoder(tensor, 5)
    expected_result = torch.zeros(11)
    expected_result[10] = 1
    assert result.shape == torch.Size([11])
    assert torch.allclose(result, expected_result)


def test_underflow_support():
    tensor = torch.tensor(-6.1)
    result = two_hot_encoder(tensor, 5)
    expected_result = torch.zeros(11)
    expected_result[0] = 1
    assert result.shape == torch.Size([11])
    assert torch.allclose(result, expected_result)


def test_integer_value():
    tensor = torch.tensor(2)
    result = two_hot_encoder(tensor, 5)
    expected_result = torch.zeros(11)
    expected_result[5 + 2] = 1
    assert result.shape == torch.Size([11])
    assert torch.allclose(result, expected_result)


def test_positive_corner_case():
    tensor = torch.tensor(5)
    result = two_hot_encoder(tensor, 5)
    expected_result = torch.zeros(11)
    expected_result[5 + 5] = 1
    assert result.shape == torch.Size([11])
    assert torch.allclose(result, expected_result)


def test_negative_corner_case():
    tensor = torch.tensor(-5)
    result = two_hot_encoder(tensor, 5)
    expected_result = torch.zeros(11)
    expected_result[0] = 1
    assert result.shape == torch.Size([11])
    assert torch.allclose(result, expected_result)
