import torch

from sheeprl.utils.utils import two_hot_decoder


def test_standard_case():
    tensor = torch.zeros(11)
    tensor[5 + 2] = 0.7
    tensor[5 + 3] = 0.3
    result = two_hot_decoder(tensor, 5)
    expected_result = torch.tensor([2.3])
    assert result.shape == torch.Size([1])
    assert torch.allclose(result, expected_result)


def test_batch_case():
    tensor = torch.zeros(2, 11)
    tensor[0, 5 + 2] = 0.7
    tensor[0, 5 + 3] = 0.3
    tensor[1, 5 + 3] = 0.6
    tensor[1, 5 + 4] = 0.4
    result = two_hot_decoder(tensor, 5)
    expected_result = torch.tensor([[2.3], [3.4]])
    assert result.shape == torch.Size([2, 1])
    assert torch.allclose(result, expected_result)


def test_support_size_1():
    tensor = torch.tensor([1.0])
    result = two_hot_decoder(tensor, 0)
    expected_result = torch.tensor([0.0])
    assert result.shape == torch.Size([1])
    assert torch.allclose(result, expected_result)


def test_integer_value():
    tensor = torch.zeros(11)
    tensor[5 + 2] = 1
    result = two_hot_decoder(tensor, 5)
    expected_result = torch.tensor([2.0])
    assert result.shape == torch.Size([1])
    assert torch.allclose(result, expected_result)


def test_positive_corner_case():
    tensor = torch.zeros(11)
    tensor[5 + 5] = 1
    result = two_hot_decoder(tensor, 5)
    expected_result = torch.tensor([5.0])
    assert result.shape == torch.Size([1])
    assert torch.allclose(result, expected_result)


def test_negative_corner_case():
    tensor = torch.zeros(11)
    tensor[5 - 5] = 1
    result = two_hot_decoder(tensor, 5)
    expected_result = torch.tensor([-5.0])
    assert result.shape == torch.Size([1])
    assert torch.allclose(result, expected_result)
