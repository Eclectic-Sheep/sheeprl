"""From: https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py"""

import math
from numbers import Number
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Bernoulli, Categorical, Distribution, constraints
from torch.distributions.kl import _kl_categorical_categorical, register_kl
from torch.distributions.utils import broadcast_all

from sheeprl.utils.utils import symexp, symlog

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

__all__ = ["OneHotCategoricalValidateArgs", "OneHotCategoricalStraightThroughValidateArgs"]


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any((self.a >= self.b).view(-1).tolist()):
            raise ValueError("Incorrect truncation range")
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a
        ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)

    def entropy(self):
        return self._entropy


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale**2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale


# TODO: let distributions be complaint to torch.distributions.Distribution API
# From https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/jaxutils.py
class SymlogDistribution:
    def __init__(
        self,
        mode: Tensor,
        dims: int,
        dist: str = "mse",
        agg: str = "sum",
        tol: float = 1e-8,
    ):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._dist = dist
        self._agg = agg
        self._tol = tol
        self._batch_shape = mode.shape[: len(mode.shape) - dims]
        self._event_shape = mode.shape[len(mode.shape) - dims :]

    @property
    def mode(self) -> Tensor:
        return symexp(self._mode)

    @property
    def mean(self) -> Tensor:
        return symexp(self._mode)

    def log_prob(self, value: Tensor) -> Tensor:
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class MSEDistribution:
    def __init__(self, mode: Tensor, dims: int, agg: str = "sum"):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self._batch_shape = mode.shape[: len(mode.shape) - dims]
        self._event_shape = mode.shape[len(mode.shape) - dims :]

    @property
    def mode(self) -> Tensor:
        return self._mode

    @property
    def mean(self) -> Tensor:
        return self._mode

    def log_prob(self, value: Tensor) -> Tensor:
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class TwoHotEncodingDistribution:
    def __init__(
        self,
        logits: Tensor,
        dims: int = 0,
        low: int = -20,
        high: int = 20,
        transfwd: Callable[[Tensor], Tensor] = symlog,
        transbwd: Callable[[Tensor], Tensor] = symexp,
    ):
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
        self.dims = tuple([-x for x in range(1, dims + 1)])
        self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)
        self.low = low
        self.high = high
        self.transfwd = transfwd
        self.transbwd = transbwd
        self._batch_shape = logits.shape[: len(logits.shape) - dims]
        self._event_shape = logits.shape[len(logits.shape) - dims : -1] + (1,)

    @property
    def mean(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

    @property
    def mode(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

    def log_prob(self, x: Tensor) -> Tensor:
        x = self.transfwd(x)
        # below in [-1, len(self.bins) - 1]
        below = (self.bins <= x).type(torch.int32).sum(dim=-1, keepdim=True) - 1
        # above in [0, len(self.bins)]
        above = below + 1

        # above in [0, len(self.bins) - 1]
        above = torch.minimum(above, torch.full_like(above, len(self.bins) - 1))
        # below in [0, len(self.bins) - 1]
        below = torch.maximum(below, torch.zeros_like(below))

        equal = below == above
        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, len(self.bins)) * weight_below[..., None]
            + F.one_hot(above, len(self.bins)) * weight_above[..., None]
        ).squeeze(-2)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdims=True)
        return (target * log_pred).sum(dim=self.dims)


# Taken from https://github.com/pytorch/pytorch/blob/main/torch/distributions/one_hot_categorical.py
# to add validate args to the categorical distribution
class OneHotCategoricalValidateArgs(Distribution):
    r"""
    Creates a one-hot categorical distribution parameterized by :attr:`probs` or
    :attr:`logits`.

    Samples are one-hot coded vectors of size ``probs.size(-1)``.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    See also: :func:`torch.distributions.Categorical` for specifications of
    :attr:`probs` and :attr:`logits`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = OneHotCategoricalValidateArgs(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor([ 0.,  0.,  0.,  1.])

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    """

    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    support = constraints.one_hot
    has_enumerate_support = True

    def __init__(self, probs=None, logits=None, validate_args=None):
        self._categorical = Categorical(probs, logits, validate_args=validate_args)
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(OneHotCategoricalValidateArgs, _instance)
        batch_shape = torch.Size(batch_shape)
        new._categorical = self._categorical.expand(batch_shape)
        super(OneHotCategoricalValidateArgs, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    @property
    def _param(self):
        return self._categorical._param

    @property
    def probs(self):
        return self._categorical.probs

    @property
    def logits(self):
        return self._categorical.logits

    @property
    def mean(self):
        return self._categorical.probs

    @property
    def mode(self):
        probs = self._categorical.probs
        mode = probs.argmax(axis=-1)
        return torch.nn.functional.one_hot(mode, num_classes=probs.shape[-1]).to(probs)

    @property
    def variance(self):
        return self._categorical.probs * (1 - self._categorical.probs)

    @property
    def param_shape(self):
        return self._categorical.param_shape

    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        probs = self._categorical.probs
        num_events = self._categorical._num_events
        indices = self._categorical.sample(sample_shape)
        return torch.nn.functional.one_hot(indices, num_events).to(probs)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        indices = value.max(-1)[1]
        return self._categorical.log_prob(indices)

    def entropy(self):
        return self._categorical.entropy()

    def enumerate_support(self, expand=True):
        n = self.event_shape[0]
        values = torch.eye(n, dtype=self._param.dtype, device=self._param.device)
        values = values.view((n,) + (1,) * len(self.batch_shape) + (n,))
        if expand:
            values = values.expand((n,) + self.batch_shape + (n,))
        return values


class OneHotCategoricalStraightThroughValidateArgs(OneHotCategoricalValidateArgs):
    r"""
    Creates a reparameterizable :class:`OneHotCategoricalValidateArgs` distribution based on the straight-
    through gradient estimator from [1].

    [1] Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation
    (Bengio et al, 2013)
    """

    has_rsample = True

    def rsample(self, sample_shape=torch.Size()):
        samples = self.sample(sample_shape)
        probs = self._categorical.probs  # cached via @lazy_property
        return samples + (probs - probs.detach())


@register_kl(OneHotCategoricalValidateArgs, OneHotCategoricalValidateArgs)
def _kl_onehotcategoricalvalidateargs_onehotcategoricalvalidateargs(p, q):
    return _kl_categorical_categorical(p._categorical, q._categorical)


class BernoulliSafeMode(Bernoulli):
    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs, logits, validate_args)

    @property
    def mode(self):
        mode = (self.probs > 0.5).to(self.probs)
        return mode
