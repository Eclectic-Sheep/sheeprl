from typing import Tuple

import pytest
import torch
from torch import nn

from sheeprl.models.models import CNN


@pytest.fixture()
def batch_size() -> int:
    return 32


@pytest.fixture()
def input_channels() -> int:
    return 10


@pytest.fixture()
def hidden_channels() -> Tuple[int, int]:
    return (64, 256)


@pytest.fixture()
def layer_args() -> Tuple[int, int]:
    return [{"kernel_size": 3}, {"kernel_size": 5}]


@pytest.fixture()
def image_shape() -> Tuple[int, int]:
    return (28, 28)


def test_cnn_raises_value_error_if_no_layer(layer_args):
    with pytest.raises(ValueError):
        CNN(input_channels=3, hidden_channels=tuple(), layer_args=layer_args)


def test_cnn_raises_value_error_if_wrong_args_type():
    with pytest.raises(ValueError):
        CNN(input_channels=10, hidden_channels=(10,), layer_args=layer_args)


def test_cnn_shape(batch_size, input_channels, hidden_channels, layer_args, image_shape):
    cnn = CNN(input_channels=input_channels, hidden_channels=hidden_channels, layer_args=layer_args)
    input_tensor = torch.rand(batch_size, input_channels, *image_shape)
    assert cnn(input_tensor).shape[:2] == torch.Size([batch_size, hidden_channels[-1]])


def test_cnn_with_dropout_args_as_dict(batch_size, input_channels, hidden_channels, layer_args):
    cnn = CNN(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        layer_args=layer_args,
        dropout_layer=nn.Dropout,
        dropout_args={"p": 0.6},
    )
    assert any([isinstance(x, nn.Dropout) for x in cnn.model])


def test_cnn_with_dropout_args_as_tuple(batch_size, input_channels, hidden_channels, layer_args):
    cnn = CNN(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        layer_args=layer_args,
        dropout_layer=nn.Dropout,
        dropout_args=(0.6,),
    )
    assert any([isinstance(x, nn.Dropout) for x in cnn.model])


def test_cnn_with_wrong_dropout_args_type(batch_size, input_channels, hidden_channels, layer_args):
    with pytest.raises(ValueError):
        CNN(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            layer_args=layer_args,
            dropout_layer=nn.Dropout,
            dropout_args=[0.6],
        )


def test_cnn_cast_dropout_args(batch_size, input_channels, hidden_channels, layer_args):
    cnn = CNN(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        layer_args=layer_args,
        dropout_layer=[nn.Dropout, nn.Dropout],
        dropout_args=(0.6,),
    )
    assert all([x.p == 0.6 for x in cnn.model if isinstance(x, nn.Dropout)])


def test_cnn_cast_multiple_dropout_args(batch_size, input_channels, hidden_channels, layer_args):
    cnn = CNN(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        layer_args=layer_args,
        dropout_layer=[nn.Dropout, nn.Dropout],
        dropout_args=[(0.6,), (0.5,)],
    )
    assert [x.p for x in cnn.model if isinstance(x, nn.Dropout)] == [0.6, 0.5]


def test_cnn_with_one_none_dropout_args(batch_size, input_channels, hidden_channels, layer_args):
    cnn = CNN(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        layer_args=layer_args,
        dropout_layer=[nn.Dropout, nn.Dropout],
        dropout_args=[(0.6,), None],
    )
    assert [x.p for x in cnn.model if isinstance(x, nn.Dropout)] == [0.6, 0.5]  # default value for p is 0.5


def test_cnn_with_one_none_dropout_layer_gives_only_one_dropout_layer(
    batch_size, input_channels, hidden_channels, layer_args
):
    cnn = CNN(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        layer_args=layer_args,
        dropout_layer=[nn.Dropout, None],
        dropout_args=[(0.6,), 0.7],
    )
    assert [x.p for x in cnn.model if isinstance(x, nn.Dropout)] == [0.6]
