"""The inputs and outputs of the agents: what they read of the observations and what their decoders reconstruct.

An input (`Input`) is a group of observation keys encoded together: the `batch` stage of the transforms of every key,
one concatenation, one network. The inputs of an algorithm (`algo.inputs`) make its encoder (`MultiEncoder`), whose
features are the ones of its inputs concatenated in the order of the configuration. An output (`Output`) is a group of
encoded keys reconstructed together from the latent state by one network, split per key; the outputs of an algorithm
(`algo.outputs`) make its decoder (`MultiDecoder`).

The networks follow a contract: an encoder network takes `input_shape` (the shape of one sample of the concatenated
keys) and exposes `output_dim`; a decoder network takes `latent_size` and `output_shape`. Both accept any leading
dimensions.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import gymnasium as gym
import hydra
import numpy as np
import torch
from torch import Tensor, nn

from sheeprl.transforms.base import Pipeline


def to_device(obs: Dict[str, np.ndarray], device: str | torch.device) -> Dict[str, Tensor]:
    """The observations `obs` as tensors on `device`, with their dtype: the inputs cast them after their transforms."""
    return {k: torch.from_numpy(np.array(v, copy=True)).to(device) for k, v in obs.items()}


def _concat_dim(shapes: Sequence[Tuple[int, ...]], keys: Sequence[str], cat_dim: Optional[int]) -> int:
    if cat_dim is not None:
        return cat_dim
    ndims = {len(shape) for shape in shapes}
    if ndims == {3}:
        return -3
    if ndims == {1}:
        return -1
    raise ValueError(
        f"The keys {list(keys)} have the shapes {[tuple(s) for s in shapes]} after their transforms: an input "
        "concatenates images (3 dimensions, along the channels) or vectors (1 dimension). Transform them to one kind "
        "(e.g. `sheeprl.transforms.Flatten`) or set the `cat_dim` of the input"
    )


def _shape(space: gym.Space, key: str) -> Tuple[int, ...]:
    """The shape of the samples of `space`, the space of `key`."""
    if space.shape is None:
        raise TypeError(f"The samples of the key `{key}` have no fixed shape after its transforms: {space}")
    return tuple(space.shape)


def _concat_shape(shapes: Sequence[Tuple[int, ...]], cat_dim: int, what: str) -> Tuple[int, ...]:
    """The shape of the concatenation of samples of `shapes` along `cat_dim`."""
    first = list(shapes[0])
    axis = cat_dim % len(first)
    for shape in shapes[1:]:
        if len(shape) != len(first) or any(a != b for i, (a, b) in enumerate(zip(shape, first)) if i != axis):
            raise ValueError(f"{what}: the shapes {[tuple(s) for s in shapes]} cannot be concatenated along {cat_dim}")
    first[axis] = sum(shape[axis] for shape in shapes)
    return tuple(first)


class Input(nn.Module):
    """A group of observation keys encoded together: the `batch` stage of the transforms of every key, their
    concatenation along `cat_dim`, one network. After their transforms, the tensors of the keys are cast to the default
    dtype of torch, the one of the inputs of the precision of Fabric inside its modules (e.g. `bf16-true`)."""

    def __init__(self, keys: Sequence[str], pipelines: Sequence[Pipeline], net: nn.Module, cat_dim: int) -> None:
        super().__init__()
        self.keys = tuple(keys)
        # A list: the observation keys are any strings, not all valid names of modules
        self.pipelines = nn.ModuleList(pipelines)
        self.net = net
        self.cat_dim = cat_dim
        # The networks of the contract expose it (checked by `from_config`)
        self.output_dim = int(getattr(net, "output_dim"))

    @classmethod
    def from_config(cls, cfg: Any, store_space: gym.spaces.Dict) -> Input:
        """The input of `cfg` (`keys`, `transforms`, `net`, optionally `cat_dim`), from the spaces of the store."""
        keys = list(cfg["keys"])
        missing = [k for k in keys if k not in store_space.spaces]
        if missing:
            raise ValueError(f"The keys {missing} are not in the store, whose keys are {list(store_space.spaces)}")
        pipelines = [Pipeline.from_config(cfg.get("transforms") or [], store_space[k], "batch", "batch") for k in keys]
        shapes = [_shape(p.batch_space, k) for k, p in zip(keys, pipelines)]
        cat_dim = _concat_dim(shapes, keys, cfg.get("cat_dim"))
        input_shape = _concat_shape(shapes, cat_dim, f"The input of the keys {keys}")
        net = hydra.utils.instantiate(cfg["net"], input_shape=list(input_shape), _convert_="all")
        if not isinstance(getattr(net, "output_dim", None), int):
            raise TypeError(f"The network of an input must expose `output_dim: int`, {type(net).__name__} doesn't")
        return cls(keys, pipelines, net, cat_dim)

    def transformed(self, obs: Dict[str, Tensor], phase: str = "train") -> List[Tensor]:
        """The tensors of the keys through their `batch` transforms of `phase`, in the default dtype."""
        dtype = torch.get_default_dtype()
        return [p.batch(obs[k], phase).to(dtype) for k, p in zip(self.keys, self._pipelines())]

    def forward(self, obs: Dict[str, Tensor], phase: str = "train", **kwargs: Any) -> Tensor:
        """The features of the keys of `obs`; the keyword arguments go to the network (e.g. the
        `detach_encoder_features` of the encoders of SAC-AE)."""
        return self.net(torch.cat(self.transformed(obs, phase), dim=self.cat_dim), **kwargs)

    def _pipelines(self) -> Sequence[Pipeline]:
        # `nn.ModuleList` types its items as `nn.Module`: they are the pipelines given to `__init__`
        return cast("Sequence[Pipeline]", self.pipelines)

    def pipeline(self, key: str) -> Pipeline:
        """The pipeline of the `batch` transforms of `key`."""
        return self._pipelines()[self.keys.index(key)]

    def batch_space(self, key: str) -> gym.Space:
        """The space of `key` after its `batch` transforms."""
        return self.pipeline(key).batch_space


class MultiEncoder(nn.Module):
    """The inputs of an algorithm (`algo.inputs`), their features concatenated in the order of the configuration.
    `keys` are the keys of all of them, each once: what the policies read and the writers store."""

    def __init__(self, inputs: Dict[str, Input]) -> None:
        super().__init__()
        if not inputs:
            raise ValueError("An encoder needs at least one input with keys (`algo.inputs.<name>.keys`)")
        self.inputs = nn.ModuleDict(inputs)
        keys: List[str] = []
        for inp in inputs.values():
            keys += [k for k in inp.keys if k not in keys]
        self.keys = tuple(keys)
        self.output_dim = sum(inp.output_dim for inp in inputs.values())

    @classmethod
    def from_config(cls, cfg_inputs: Any, store_space: gym.spaces.Dict) -> MultiEncoder:
        """The inputs configured in `cfg_inputs` (`algo.inputs`), skipping the ones without keys."""
        return cls(
            {
                name: Input.from_config(spec, store_space)
                for name, spec in (cfg_inputs or {}).items()
                if spec and spec.get("keys")
            }
        )

    def _inputs(self) -> Mapping[str, Input]:
        # `nn.ModuleDict` types its items as `nn.Module`: they are the inputs given to `__init__`
        return cast("Mapping[str, Input]", self.inputs)

    def forward(self, obs: Dict[str, Tensor], phase: str = "train", **kwargs: Any) -> Tensor:
        features = [inp(obs, phase, **kwargs) for inp in self._inputs().values()]
        return features[0] if len(features) == 1 else torch.cat(features, dim=-1)

    def store_space(self, key: str) -> gym.Space:
        """The space of `key` in the store (before the `batch` transforms)."""
        for inp in self._inputs().values():
            if key in inp.keys:
                return inp.pipeline(key).envs_space
        raise KeyError(f"No input reads the key `{key}`: the keys of the inputs are {list(self.keys)}")

    def batch_space(self, key: str) -> gym.Space:
        """The space of `key` after the `batch` transforms of the first input that reads it."""
        for inp in self._inputs().values():
            if key in inp.keys:
                return inp.batch_space(key)
        raise KeyError(f"No input reads the key `{key}`: the keys of the inputs are {list(self.keys)}")

    def targets(self, obs: Dict[str, Tensor], keys: Sequence[str]) -> Dict[str, Tensor]:
        """The tensors of `keys` through the `batch` transforms of the first input that reads them, in the `train`
        phase: the targets of the losses of the decoders."""
        targets = {}
        for k in keys:
            for inp in self._inputs().values():
                if k in inp.keys:
                    targets[k] = inp.pipeline(k).batch(obs[k], "train").to(torch.get_default_dtype())
                    break
            else:
                raise KeyError(f"No input reads the key `{k}`: a decoded key must be encoded")
        return targets


def build_encoder(cfg: Any, store_space: gym.spaces.Dict) -> MultiEncoder:
    """The encoder of the inputs of `cfg` (`algo.inputs`), of the class `algo.encoder._target_` (`MultiEncoder` by
    default, for a fusion of the inputs other than the concatenation of their features)."""
    target = ((cfg.algo.get("encoder") or {}).get("_target_")) or "sheeprl.models.io.MultiEncoder"
    return hydra.utils.get_class(target).from_config(cfg.algo.inputs, store_space)


class Output(nn.Module):
    """A group of encoded keys reconstructed together: one network from the latent state to their concatenation, split
    per key. The targets of the reconstructions are the keys in the domain of the `batch` transforms of the encoder, or,
    with `target_pipelines`, of the output's own transforms (e.g. images scaled to [0, 1] while the encoder reads them
    in [-0.5, 0.5]). `likelihood`, if any, makes the distribution of the reconstruction of a key, which scores its
    target (`log_prob`)."""

    def __init__(
        self,
        keys: Sequence[str],
        shapes: Sequence[Tuple[int, ...]],
        net: nn.Module,
        cat_dim: int,
        likelihood: Optional[Callable[..., Any]] = None,
        target_pipelines: Optional[Sequence[Pipeline]] = None,
    ) -> None:
        super().__init__()
        self.keys = tuple(keys)
        self.shapes = [tuple(s) for s in shapes]
        self.net = net
        self.cat_dim = cat_dim
        self.likelihood = likelihood
        self.target_pipelines = nn.ModuleList(target_pipelines) if target_pipelines is not None else None
        axis = cat_dim % len(self.shapes[0])
        self.split_sizes = [s[axis] for s in self.shapes]

    @classmethod
    def from_config(cls, cfg: Any, encoder: MultiEncoder, latent_size: int) -> Output:
        """The output of `cfg` (`keys`, `net`, optionally `likelihood`, `transforms` of its targets and `cat_dim`): the
        shape of every key is the one the encoder gives it after its `batch` transforms."""
        keys = list(cfg["keys"])
        shapes = [_shape(encoder.batch_space(k), k) for k in keys]
        target_pipelines = None
        if cfg.get("transforms") is not None:
            target_pipelines = [
                Pipeline.from_config(cfg["transforms"], encoder.store_space(k), "batch", "batch") for k in keys
            ]
        cat_dim = _concat_dim(shapes, keys, cfg.get("cat_dim"))
        output_shape = _concat_shape(shapes, cat_dim, f"The output of the keys {keys}")
        net = hydra.utils.instantiate(
            cfg["net"], latent_size=int(latent_size), output_shape=list(output_shape), _convert_="all"
        )
        likelihood = hydra.utils.instantiate(cfg["likelihood"]) if cfg.get("likelihood") else None
        return cls(keys, shapes, net, cat_dim, likelihood, target_pipelines)

    def forward(self, latent: Tensor) -> Dict[str, Tensor]:
        out = self.net(latent)
        return dict(zip(self.keys, torch.split(out, self.split_sizes, dim=self.cat_dim)))

    def targets(self, obs: Dict[str, Tensor], encoder: MultiEncoder) -> Dict[str, Tensor]:
        """The targets of the reconstructions of the keys of `obs` (as stored)."""
        if self.target_pipelines is None:
            return encoder.targets(obs, self.keys)
        dtype = torch.get_default_dtype()
        # `nn.ModuleList` types its items as `nn.Module`: they are the pipelines given to `__init__`
        target_pipelines = cast("Sequence[Pipeline]", self.target_pipelines)
        return {k: p.batch(obs[k], "train").to(dtype) for k, p in zip(self.keys, target_pipelines)}

    def log_probs(self, reconstructions: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """The log-likelihood of the target of every key under the distribution of its reconstruction."""
        if self.likelihood is None:
            raise RuntimeError(f"The output of the keys {list(self.keys)} has no likelihood")
        return {
            k: self.likelihood(reconstructions[k], dims=len(shape)).log_prob(targets[k])
            for k, shape in zip(self.keys, self.shapes)
        }

    def log_prob(self, reconstructions: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Tensor:
        """The sum over the keys of the log-likelihood of their targets under the distribution of their
        reconstruction."""
        # A tensor: an output has at least one key
        return cast(Tensor, sum(self.log_probs(reconstructions, targets).values()))


class MultiDecoder(nn.Module):
    """The outputs of an algorithm (`algo.outputs`): the reconstructions of all their keys."""

    def __init__(self, outputs: Dict[str, Output]) -> None:
        super().__init__()
        self.outputs = nn.ModuleDict(outputs)
        self.keys = tuple(k for out in outputs.values() for k in out.keys)

    @classmethod
    def from_config(cls, cfg_outputs: Any, encoder: MultiEncoder, latent_size: int) -> MultiDecoder:
        """The outputs configured in `cfg_outputs` (`algo.outputs`), skipping the ones without keys."""
        return cls(
            {
                name: Output.from_config(spec, encoder, latent_size)
                for name, spec in (cfg_outputs or {}).items()
                if spec and spec.get("keys")
            }
        )

    def _outputs(self) -> Mapping[str, Output]:
        # `nn.ModuleDict` types its items as `nn.Module`: they are the outputs given to `__init__`
        return cast("Mapping[str, Output]", self.outputs)

    def forward(self, latent: Tensor) -> Dict[str, Tensor]:
        reconstructions: Dict[str, Tensor] = {}
        for out in self._outputs().values():
            reconstructions.update(out(latent))
        return reconstructions

    def targets(self, obs: Dict[str, Tensor], encoder: MultiEncoder) -> Dict[str, Tensor]:
        """The targets of the reconstructions of all the keys of `obs` (as stored)."""
        targets: Dict[str, Tensor] = {}
        for out in self._outputs().values():
            targets.update(out.targets(obs, encoder))
        return targets

    def log_probs(self, reconstructions: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """The log-likelihood of every key, in the order of the outputs."""
        log_probs: Dict[str, Tensor] = {}
        for out in self._outputs().values():
            log_probs.update(out.log_probs(reconstructions, targets))
        return log_probs

    def log_prob(self, reconstructions: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Tensor:
        # A tensor: the decoders reconstruct at least one key (`check_reconstructs`)
        return cast(Tensor, sum(out.log_prob(reconstructions, targets) for out in self._outputs().values()))


def log_inputs_and_outputs(
    fabric: Any, cfg: Any, encoder: MultiEncoder, decoder: Optional[MultiDecoder] = None
) -> None:
    """Print the keys of the inputs and of the outputs, with `metric.log_level`."""
    if cfg.metric.log_level > 0:
        for name, inp in encoder._inputs().items():
            fabric.print(f"Input `{name}`: {list(inp.keys)}")
        for name, out in decoder._outputs().items() if decoder is not None else ():
            fabric.print(f"Output `{name}`: {list(out.keys)}")


def check_reconstructs(decoder: MultiDecoder, algorithm: str) -> None:
    """An algorithm with a decoder reconstructs at least one key."""
    if len(decoder.keys) == 0:
        raise RuntimeError(f"{algorithm} reconstructs at least one key: set the keys of an output (`algo.outputs`)")
