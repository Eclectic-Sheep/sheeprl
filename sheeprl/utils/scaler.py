from __future__ import annotations

import torch
from torch import Tensor, nn


# Taken from: https://github.com/tianheyu927/mopo/blob/e5efe3eaa548850af109c920bfc2c1ec96bf2285/mopo/models/utils.py#L45
class TensorStandardScaler(nn.Module):
    """Helper class for automatically normalizing inputs into the network."""

    def __init__(self, input_dims: int, device: torch.device | str = "cpu"):
        """Initializes a scaler.

        Arguments:
        input_dims: The dimensionality of the inputs into the scaler.
        device: The device used for training. Default to "cpu".

        Returns: None.
        """
        super().__init__()
        self.device = device
        if isinstance(device, str):
            self.device = torch.device(device)
        self.fitted = False
        self.register_buffer("mu", torch.zeros(1, input_dims, device=self.device))
        self.register_buffer("sigma", torch.ones(1, input_dims, device=self.device))

        self.cached_mu, self.cached_sigma = (
            torch.zeros(0, input_dims, device=self.device),
            torch.ones(1, input_dims, device=self.device),
        )

    def fit(self, data: Tensor):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data: A tensor containing the input.

        Returns: None.
        """
        mu: Tensor = torch.mean(data, dim=0, keepdim=True)
        sigma: Tensor = torch.std(data, dim=0, keepdim=True)
        sigma[sigma < 1e-12] = 1.0

        self.mu: Tensor = mu.type_as(self.mu)
        self.sigma: Tensor = sigma.type_as(self.sigma)
        self.fitted = True
        self.cache()

    def transform(self, data: Tensor) -> Tensor:
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data: A tensor array containing the points to be transformed.

        Returns: The transformed dataset.
        """
        return (data - self.mu) / self.sigma

    def inverse_transform(self, data: Tensor) -> Tensor:
        """Undoes the transformation performed by this scaler.

        Arguments:
        data: A tensor array containing the points to be transformed.

        Returns: The transformed dataset.
        """
        return self.sigma * data + self.mu

    def cache(self):
        """Caches current values of this scaler."""
        self.cached_mu = self.mu
        self.cached_sigma = self.sigma

    def load_cache(self):
        """Loads values from the cache."""
        self.mu = self.cached_mu.type_as(self.mu)
        self.sigma = self.cached_sigma.type_as(self.sigma)
