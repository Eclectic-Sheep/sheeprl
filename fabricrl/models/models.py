from functools import partial
from typing import Any, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from fabricrl.utils.model import create_mlp, per_layer_ortho_init_weights


class BaseModule(nn.Module):
    """Base feature extractor. Every child class must set the `self.mlp_model` attribute.

    Args:
        input_shape (int): observations dimension.
    """

    def __init__(self, input_shape: Sequence[int]):
        super().__init__()
        self._input_shape = input_shape
        self._output_dim: Union[int, Sequence[int]]
        self.mlp_model: torch.nn.Module
        self._num_recurrent_states: int = 0

    @property
    def input_shape(self) -> Sequence[int]:
        """Get the input dimension.
        Returns:
            the input dimension.
        """
        return self._input_shape

    @property
    def output_dim(self) -> int:
        """Get the output dimension.
        Returns:
            the output dimension."""
        return self._output_dim

    @output_dim.setter
    def output_dim(self, value) -> None:
        """Set the output dimension."""
        self._output_dim = value

    @property
    def num_recurrent_states(self) -> int:
        """Get the number of recurrent states.
        Returns:
            int: the number of recurrent states.
        """
        return self._num_recurrent_states

    @num_recurrent_states.setter
    def num_recurrent_states(self, value: int) -> None:
        """Set the number of recurrent states."""
        self._num_recurrent_states = value

    def forward(self, observations: torch.Tensor, **kwargs) -> Any:
        """Forward pass of the network. It must be implemented by the child class."""
        raise NotImplementedError()


class Identity(BaseModule):
    """Identity feature extractor simply returns what it has received as input. It sets `self.mlp_model` as a
    `torch.nn.Identiy`.

    Args:
        input_shape (int): the input dimension.
    """

    def __init__(self, input_shape: Sequence[int]):
        super().__init__(input_shape)
        self.output_dim = np.prod(input_shape)
        self.mlp_model = nn.Identity()

    def forward(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            observations (torch.Tensor): the observations.

        Returns:
            torch.Tensor: the features.
        """
        return self.mlp_model(observations)


class MLP(BaseModule):
    """Extract features with an MLP. The MLP is created using `twirl.utility.model.create_mlp` and
    cuold be empty.

    Args:
        input_shape (int): the observations dimension.
        hidden_sizes (Sequence[int]): a sequence of integers (possibly empty), which specifies the hidden dimensions
            of the MLP. If an empty sequence is passed, then `self.mlp_model` will simply be a `torch.nn.Identity`.
            Defaults to tuple().
        ortho_init (bool, optional): whether to apply the orthogonal initialization.
            Defaults to False.
        activation_fn (Optional[Type[nn.Module]], optional): the activation function between linear layers.
            Defaults to nn.ReLU.

    Examples:
        >>> MLP((8, ), [64, 32], (nn.ReLU(), nn.ReLU()))
        MLPExtractor(
            (mlp_model): Sequential(
                (0): Linear(in_features=8, out_features=64, bias=True)
                (1): ReLU()
                (2): Linear(in_features=64, out_features=32, bias=True)
                (3): ReLU()
            )
        )
        >>> MLP((8, ), tuple())
        MLPExtractor(
            (mlp_model): Identity()
        )
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        hidden_sizes: Sequence[int] = tuple(),
        activation_fn: Union[nn.Module, Sequence[nn.Module]] = nn.Identity(),
        ortho_init: bool = False,
    ):
        super().__init__(input_shape)
        self._hidden_sizes = hidden_sizes
        self._ortho_init = ortho_init
        self.output_dim = hidden_sizes[-1] if len(hidden_sizes) > 0 else np.prod(input_shape)
        activation_fn = activation_fn if isinstance(activation_fn, Sequence) else (activation_fn,) * len(hidden_sizes)
        self.mlp_model = create_mlp(
            input_dim=np.prod(input_shape), hidden_sizes=hidden_sizes, activation_fn=activation_fn
        )
        if self._ortho_init:
            self.mlp_model.apply(partial(per_layer_ortho_init_weights, gain=np.sqrt(2.0)))

    @property
    def hidden_sizes(self) -> Sequence[int]:
        """Get the hidden sizes.
        Returns:
            Sequence[int]: the hidden sizes.
        """
        return self._hidden_sizes

    @property
    def ortho_init(self) -> bool:
        """Get the orthogonal initialization flag.
        Returns:
            bool: the orthogonal initialization flag.
        """
        return self._ortho_init

    def forward(
        self, observations: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """Extracts features with an MLP.

        Args:
            observations (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: the extracted features.
        """
        return self.mlp_model(observations)


class Recurrent(MLP):
    """Extract features first with an MLP, the applies a recurrent (RNN/GRU) module on the extracted features.
    The MLP is created using `twirl.utility.model.create_mlp` and could be empty.

    Args:
        input_shape (int): the observations dimension.
        rnn_hidden_dim (Optional[int], optional): the RNN hidden dimension.
            If it is None, then it will be set as the last hidden dimension of the MLP; if there is no MLP,
            then it will be set as the input dimension.
            Defaults to None.
        rnn_num_layers (int, optional): the number of layers in the RNN. Defaults to 1.
        rnn_batch_first (bool, optional): whether to use BxTxD shape or TxBxD, where B is the batch size,
            T is the sequence length and D is feature dimension.
            Defaults to False.
        hidden_sizes (Sequence[int]): a sequence of integers (possibly empty), which specifies the hidden dimensions
            of the MLP. If an empty sequence is passed, then `self.mlp_model` will simply be a `torch.nn.Identity`.
            Defaults to tuple().
        ortho_init (bool, optional): whether to apply the orthogonal initialization.
            Defaults to False.
        activation_fn (Optional[Type[nn.Module]], optional): the activation function between linear layers.
            Defaults to nn.ReLU.

    Examples:
        >>> Recurrent((8, ), rnn_hidden_dim=32, rnn_num_layers=1, hidden_sizes=[32], activation_fn=(nn.ReLU(),))
        RecurrentExtractor(
            (mlp_model): Sequential(
                (0): Linear(in_features=8, out_features=32, bias=True)
                (1): ReLU()
            )
            (rnn): RNN(32, 32)
        )
        >>> Recurrent((8, ), rnn_hidden_dim=32, rnn_num_layers=1, hidden_sizes=tuple())
        RecurrentExtractor(
            (mlp_model): Identity()
            (rnn): RNN(8, 32)
        )
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        rnn_net: nn.Module = nn.RNN,
        rnn_hidden_dim: Optional[int] = None,
        rnn_num_layers: int = 1,
        rnn_batch_first: bool = False,
        hidden_sizes: Sequence[int] = tuple(),
        activation_fn: Union[nn.Module, Sequence[nn.Module]] = nn.Identity(),
        ortho_init: bool = False,
    ):
        super().__init__(
            input_shape,
            hidden_sizes=hidden_sizes,
            ortho_init=ortho_init,
            activation_fn=activation_fn,
        )
        self._rnn_input_dim = hidden_sizes[-1] if len(hidden_sizes) > 0 else np.prod(input_shape)
        self._rnn_hidden_dim = rnn_hidden_dim if rnn_hidden_dim is not None else self._rnn_input_dim
        self._rnn_num_layers = rnn_num_layers
        self._rnn_batch_first = rnn_batch_first
        self.rnn = rnn_net(
            input_size=self._rnn_input_dim,
            hidden_size=self._rnn_hidden_dim,
            num_layers=self._rnn_num_layers,
            batch_first=self._rnn_batch_first,
        )
        if self._ortho_init:
            self.rnn.apply(partial(per_layer_ortho_init_weights, gain=np.sqrt(2.0)))
        self._num_recurrent_states = 1
        self.output_dim = self._rnn_hidden_dim

    @property
    def rnn_input_dim(self) -> int:
        """Get the RNN input dimension.

        Returns:
            the RNN input dimension.
        """
        return self._rnn_input_dim

    @property
    def rnn_hidden_dim(self) -> int:
        """Get the RNN hidden dimension.
        Returns:
            the RNN hidden dimension.
        """
        return self._rnn_hidden_dim

    @property
    def rnn_num_layers(self) -> int:
        """Get the RNN number of layers.
        Returns:
            the RNN number of layers.
        """
        return self._rnn_num_layers

    @property
    def rnn_batch_first(self) -> int:
        """Get the RNN batch first flag.
        Returns:
            the RNN batch first flag.
        """
        return self._rnn_batch_first

    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Apply an RNN model.

        Args:
            observations (torch.Tensor): the input tensor.
            state (Tuple[torch.Tensor]): the input state for the RNN.
            lengths (List[int], optional): lengths of the sequences to be fed to the RNN. If not None,
                then a [torch.nn.utils.rnn.PackedSequence][] will be fed to the RNN.
                Defaults to None.

        Returns:
            the extracted output and the next rnn state.
        """
        feat = self.mlp_model(observations)
        state = state[0] if state is not None else state
        self.rnn.flatten_parameters()
        if lengths is not None:
            out, hx = self.rnn(
                pack_padded_sequence(feat, lengths, batch_first=False, enforce_sorted=False),
                state,
            )
            out, _ = pad_packed_sequence(out, batch_first=False)
        else:
            out, hx = self.rnn(feat, state)
            if out.shape[1] == 1:
                out = out.view(out.shape[0], -1)
        return out, hx


class LSTM(Recurrent):
    """Extract features first with an MLP, the applies an LSTM module on the extracted features.
    The MLP is created using `twirl.utility.model.create_mlp` and could be empty.

    Args:
        input_shape (int): the observations dimension.
        rnn_hidden_dim (Optional[int], optional): the LSTM hidden dimension.
            If it is None, then it will be set as the last hidden dimension of the MLP; if there is no MLP,
            then it will be set as the input dimension.
            Defaults to None.
        rnn_num_layers (int, optional): the number of layers in the LSTM. Defaults to 1.
        rnn_batch_first (bool, optional): whether to use BxTxD shape or TxBxD,
            where B is the batch size, T is the sequence length and D is feature dimension.
            Defaults to False.
        hidden_sizes (Sequence[int]): a sequence of integers (possibly empty), which specifies the hidden dimensions
            of the MLP. If an empty sequence is passed, then `self.mlp_model` will simply be a `torch.nn.Identity`.
            Defaults to tuple().
        ortho_init (bool, optional): whether to apply the orthogonal initialization.
            Defaults to False.
        activation_fn (Optional[Type[nn.Module]], optional): the activation function between linear layers.
            Defaults to nn.ReLU.

    Examples:
        >>> LSTM((8, ), rnn_hidden_dim=32, rnn_num_layers=1, hidden_sizes=(32,))
        LSTMExtractor(
            (features): Sequential(
                (0): Linear(in_features=8, out_features=32, bias=True)
                (1): ReLU()
            )
            (lstm): LSTM(32, 32)
        )
        >>> LSTM((8, ), rnn_hidden_dim=32, rnn_num_layers=1, hidden_sizes=tuple())
        LSTMExtractor(
            (features): Identity()
            (lstm): LSTM(8, 32)
        )
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        rnn_net: nn.Module = nn.LSTM,
        rnn_hidden_dim: Optional[int] = None,
        rnn_num_layers: int = 1,
        rnn_batch_first: bool = False,
        hidden_sizes: Sequence[int] = tuple(),
        ortho_init: bool = False,
        activation_fn: Union[nn.Module, Sequence[nn.Module]] = nn.Identity(),
    ):
        if rnn_net is not nn.LSTM:
            raise ValueError("rnn_net must be an instance of torch.nn.LSTM with LSTM")
        super().__init__(
            input_shape,
            rnn_net=rnn_net,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_batch_first=rnn_batch_first,
            hidden_sizes=hidden_sizes,
            ortho_init=ortho_init,
            activation_fn=activation_fn,
        )
        self._num_recurrent_states = 2

    def forward(
        self,
        observations: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Extract features with an LSTM model.

        Args:
            observations (torch.Tensor): the input tensor.
            state (Tuple[torch.Tensor, torch.Tensor]): the input state for the LSTM.
            lengths (List[int], optional): lengths of the sequences to be fed to the LSTM. If not None,
                then a [torch.nn.utils.rnn.PackedSequence][] will be fed to the LSTM.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor]]: the extracted features and the next lstm state.
        """
        feat = self.mlp_model(observations)
        self.rnn.flatten_parameters()
        if lengths is not None:
            out, (hx, cx) = self.rnn(
                pack_padded_sequence(feat, lengths, batch_first=False, enforce_sorted=False),
                state[:2] if state is not None else state,
            )
            out, _ = pad_packed_sequence(out, batch_first=False)
        else:
            out, (hx, cx) = self.rnn(feat, state[:2] if state is not None else state)
            if out.shape[1] == 1:
                out = out.view(out.shape[0], -1)
        return out, (hx, cx)


class ConvModule(BaseModule):
    def __init__(
        self,
        input_shape: Sequence[int],
        channel_outs: Sequence[int] = tuple(),
        cnn_activation_fn: Union[nn.Module, Sequence[nn.Module]] = nn.Identity,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int]] = 0,
        **kwargs
    ):
        super().__init__(input_shape=input_shape)
        self.layer_type: Type[nn.Module]
        self._channel_outs = channel_outs

        self.cnn_activation_fn = (
            cnn_activation_fn if isinstance(cnn_activation_fn, Sequence) else (cnn_activation_fn,) * len(channel_outs)
        )
        self.kernel_size = kernel_size if isinstance(kernel_size, Sequence) else (kernel_size,) * len(channel_outs)
        self.stride = stride if isinstance(stride, Sequence) else (stride,) * len(channel_outs)
        self.padding = padding if isinstance(padding, Sequence) else (padding,) * len(channel_outs)

    def build_cnn(self, channel_in, **kwargs) -> torch.nn.Module:
        """Create an CNN backbone consisting of nn.Conv2d followed by an activation function."""
        if len(self._channel_outs) > 0:
            layers = [
                self.layer_type(
                    channel_in,
                    self._channel_outs[0],
                    kernel_size=self.kernel_size[0],
                    stride=self.stride[0],
                    padding=self.padding[0],
                    **kwargs
                ),
                self.self.cnn_activation_fn[0],
            ]
            for dim_i in range(1, len(self._channel_outs)):
                layers.append(
                    self.layer_type(
                        self._channel_outs[dim_i - 1],
                        self._channel_outs[dim_i],
                        kernel_size=self.kernel_size[dim_i],
                        stride=self.stride[dim_i],
                        padding=self.padding[dim_i],
                        **kwargs
                    )
                )
                layers.append(self.cnn_activation_fn[dim_i])
            cnn = nn.Sequential(*layers)
        else:
            cnn = nn.Identity()
        return cnn

    def forward(self, observations: torch.Tensor, **kwargs) -> Any:
        raise NotImplementedError


class CNN(ConvModule):
    """Extract features first with a CNN, then applies an MLP on the extracted features."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        channel_outs: Sequence[int] = tuple(),
        hidden_sizes: Sequence[int] = tuple(),
        mlp_activation_fn: Union[nn.Module, Sequence[nn.Module]] = nn.Identity,
        cnn_activation_fn: Union[nn.Module, Sequence[nn.Module]] = nn.Identity,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int]] = 0,
        ortho_init: bool = False,
        **cnn_kwargs
    ):
        super().__init__(
            input_shape=input_shape,
            channel_outs=channel_outs,
            cnn_activation_fn=cnn_activation_fn,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **cnn_kwargs
        )

        self.cnn = self.build_cnn(channel_in=input_shape[0], **cnn_kwargs)
        self.embedding_shape = self.cnn(torch.zeros(*input_shape)).shape

        mlp_input_dim = self.cnn(torch.zeros(*input_shape)).numel()
        mlp_activation_fn = (
            mlp_activation_fn if isinstance(mlp_activation_fn, Sequence) else (mlp_activation_fn,) * len(hidden_sizes)
        )
        self._hidden_sizes = hidden_sizes
        self._ortho_init = ortho_init
        self.output_dim = hidden_sizes[-1] if len(hidden_sizes) > 0 else mlp_input_dim
        self.mlp_model = create_mlp(
            input_dim=mlp_input_dim,
            hidden_sizes=hidden_sizes,
            activation_fn=mlp_activation_fn,
        )

        if self._ortho_init:
            self.mlp_model.apply(partial(per_layer_ortho_init_weights, gain=np.sqrt(2.0)))

    def forward(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Extract features from input.

        Args:
            observations (torch.Tensor): observations to extract features from.

        Returns:
            torch.Tensor: extracted features.
        """
        feat = self.cnn(observations)
        feat = feat.flatten(start_dim=1)
        return self.mlp_model(feat)


class DeCNN(ConvModule):
    """Extract features first with a CNN, then applies an MLP on the extracted features."""

    def __init__(
        self,
        input_shape: Sequence[int],
        channel_outs: Sequence[int] = tuple(),
        hidden_sizes: Sequence[int] = tuple(),
        ortho_init: bool = False,
        mlp_activation_fn: Union[nn.Module, Sequence[nn.Module]] = nn.Identity(),
        cnn_activation_fn: Union[nn.Module, Sequence[nn.Module]] = nn.Identity(),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int]] = 0,
        **cnn_kwargs
    ):
        super().__init__(
            input_shape=input_shape,
            channel_outs=channel_outs,
            cnn_activation_fn=cnn_activation_fn,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **cnn_kwargs
        )

        self._hidden_sizes = hidden_sizes
        self._ortho_init = ortho_init
        mlp_activation_fn = (
            mlp_activation_fn if isinstance(mlp_activation_fn, Sequence) else (mlp_activation_fn,) * len(hidden_sizes)
        )
        self.mlp_model = create_mlp(
            input_dim=np.prod(self.input_shape),
            hidden_sizes=hidden_sizes,
            activation_fn=mlp_activation_fn,
        )
        mlp_output_dim = self.mlp_model(torch.zeros(1, np.prod(self.input_shape))).shape[1]

        self.layer_type: type[nn.Module] = nn.ConvTranspose2d
        self._channel_outs = channel_outs
        self.activation_fn = (
            cnn_activation_fn if isinstance(cnn_activation_fn, Sequence) else (cnn_activation_fn,) * len(channel_outs)
        )
        self.kernel_size = kernel_size if isinstance(kernel_size, Sequence) else (kernel_size,) * len(channel_outs)
        self.stride = stride if isinstance(stride, Sequence) else (stride,) * len(channel_outs)
        self.padding = padding if isinstance(padding, Sequence) else (padding,) * len(channel_outs)
        self.de_cnn = self.build_cnn(channel_in=mlp_output_dim, **cnn_kwargs)

        self.output_dim = self.de_cnn(torch.zeros(1, mlp_output_dim, 1, 1)).shape

        if self._ortho_init:
            self.mlp_model.apply(partial(per_layer_ortho_init_weights, gain=np.sqrt(2.0)))

    def forward(self, embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            embedding (torch.Tensor): Embedding to be decoded.

        Returns:
            torch.Tensor: Decoded embedding.
        """
        feat = self.mlp_model(embedding)
        feat = feat.unsqueeze(-1).unsqueeze(-1)
        return self.de_cnn(feat)
