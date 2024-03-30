from typing import Tuple

import pytest
import torch
from torch import nn

from sheeprl.models.models import MLP


@pytest.fixture()
def batch_size() -> int:
    return 32


@pytest.fixture()
def input_dims() -> int:
    return 10


@pytest.fixture()
def multidimensional_input_dims() -> Tuple[int, int, int]:
    return (5, 4, 6)


@pytest.fixture()
def hidden_sizes() -> Tuple[int, int]:
    return (64, 256)


@pytest.fixture()
def output_dim() -> int:
    return 4


def test_mlp_raises_value_error_if_no_layer():
    with pytest.raises(ValueError):
        MLP(input_dims=10, output_dim=None, hidden_sizes=tuple())


def test_mlp_with_zero_output_dim_(input_dims):
    mlp = MLP(input_dims=input_dims, output_dim=0, hidden_sizes=tuple())
    input_tensor = torch.rand(input_dims)
    assert mlp(input_tensor).shape == torch.Size([0])


def test_mlp_raises_value_error_if_wrong_args_type():
    with pytest.raises(ValueError):
        MLP(input_dims=10, output_dim=None, hidden_sizes=(10,), layer_args=10)


def test_mlp_only_output_dim(batch_size, input_dims, output_dim):
    mlp = MLP(input_dims=input_dims, output_dim=output_dim, hidden_sizes=tuple())
    input_tensor = torch.rand(batch_size, input_dims)
    assert mlp(input_tensor).shape == (batch_size, output_dim)


def test_mlp_multidimensional_input_dims(batch_size, multidimensional_input_dims, output_dim):
    mlp = MLP(input_dims=multidimensional_input_dims, output_dim=output_dim, hidden_sizes=tuple(), flatten_dim=1)
    input_tensor = torch.rand(batch_size, *multidimensional_input_dims)
    assert mlp(input_tensor).shape == (batch_size, output_dim)


def test_mlp_flatten_dims(batch_size, multidimensional_input_dims, output_dim):
    flatten_mlp = MLP(input_dims=multidimensional_input_dims, output_dim=output_dim, flatten_dim=1)
    mlp = MLP(input_dims=multidimensional_input_dims, output_dim=output_dim)
    mlp.load_state_dict(flatten_mlp.state_dict())
    input_tensor = torch.rand(batch_size, *multidimensional_input_dims)
    assert torch.allclose(mlp(input_tensor.flatten(1)), flatten_mlp(input_tensor))


def test_mlp_only_hidden_sizes(batch_size, input_dims, hidden_sizes):
    mlp = MLP(input_dims=input_dims, hidden_sizes=hidden_sizes)
    input_tensor = torch.rand(batch_size, input_dims)
    assert mlp(input_tensor).shape == (batch_size, hidden_sizes[-1])


def test_mlp_multidimensional_batch_size(input_dims, output_dim):
    batch_size = (16, 16)
    mlp = MLP(input_dims=input_dims, output_dim=output_dim, hidden_sizes=tuple())
    input_tensor = torch.rand(*batch_size, input_dims)
    assert mlp(input_tensor).shape == (*batch_size, output_dim)


def test_mlp_with_dropout_args_as_dict(batch_size, input_dims, hidden_sizes):
    mlp = MLP(input_dims=input_dims, hidden_sizes=hidden_sizes, dropout_layer=nn.Dropout, dropout_args={"p": 0.6})
    assert any([isinstance(x, nn.Dropout) for x in mlp.model])


def test_mlp_with_dropout_args_as_tuple(batch_size, input_dims, hidden_sizes):
    mlp = MLP(input_dims=input_dims, hidden_sizes=hidden_sizes, dropout_layer=nn.Dropout, dropout_args=(0.6,))
    assert any([isinstance(x, nn.Dropout) for x in mlp.model])


def test_mlp_with_wrong_dropout_args_type(batch_size, input_dims, hidden_sizes):
    with pytest.raises(ValueError):
        MLP(input_dims=input_dims, hidden_sizes=hidden_sizes, dropout_layer=nn.Dropout, dropout_args=[0.6])


def test_mlp_cast_dropout_args(batch_size, input_dims, hidden_sizes):
    mlp = MLP(
        input_dims=input_dims, hidden_sizes=hidden_sizes, dropout_layer=[nn.Dropout, nn.Dropout], dropout_args=(0.6,)
    )
    assert all([x.p == 0.6 for x in mlp.model if isinstance(x, nn.Dropout)])


def test_mlp_cast_multiple_dropout_args(batch_size, input_dims, hidden_sizes):
    mlp = MLP(
        input_dims=input_dims,
        hidden_sizes=hidden_sizes,
        dropout_layer=[nn.Dropout, nn.Dropout],
        dropout_args=[(0.6,), (0.5,)],
    )
    assert [x.p for x in mlp.model if isinstance(x, nn.Dropout)] == [0.6, 0.5]


def test_mlp_with_one_none_dropout_args(batch_size, input_dims, hidden_sizes):
    mlp = MLP(
        input_dims=input_dims,
        hidden_sizes=hidden_sizes,
        dropout_layer=[nn.Dropout, nn.Dropout],
        dropout_args=[(0.6,), None],
    )
    assert [x.p for x in mlp.model if isinstance(x, nn.Dropout)] == [0.6, 0.5]  # default value for p is 0.5


def test_mlp_with_one_none_dropout_layer_gives_only_one_dropout_layer(batch_size, input_dims, hidden_sizes):
    mlp = MLP(
        input_dims=input_dims, hidden_sizes=hidden_sizes, dropout_layer=[nn.Dropout, None], dropout_args=[(0.6,), 0.7]
    )
    assert [x.p for x in mlp.model if isinstance(x, nn.Dropout)] == [0.6]
