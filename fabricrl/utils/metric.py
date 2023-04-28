from collections import deque
from typing import Any, Dict, Optional, Union

import torch
from torchmetrics import Metric


class MetricAggregatorException(Exception):
    """A custom exception used to report errors in use of timer class"""


class MetricAggregator:
    """A metric aggregator class to aggregate metrics to be tracked.
    Args:
        metrics (Optional[Dict[str, Metric]]): Dict of metrics to aggregate.
    """

    def __init__(self, metrics: Optional[Dict[str, Metric]] = None):
        self.metrics: Dict[str, Metric] = {}
        if metrics is not None:
            self.metrics = metrics

    def add(self, name: str, metric: Metric):
        """Add a metric to the aggregator

        Args:
            name (str): Name of the metric
            metric (Metric): Metric to add.

        Raises:
            MetricAggregatorException: If the metric already exists.
        """
        if name not in self.metrics:
            self.metrics.setdefault(name, metric)
        else:
            raise MetricAggregatorException(f"Metric {name} already exists")

    @torch.no_grad()
    def update(self, name: str, value: Any) -> None:
        """Update the metric with the value

        Args:
            name (str): Name of the metric
            value (Any): Value to update the metric with.

        Raises:
            MetricAggregatorException: If the metric does not exist.
        """
        if name not in self.metrics:
            raise MetricAggregatorException(f"Metric {name} does not exist")
        self.metrics[name].update(value)

    def pop(self, name: str) -> None:
        """Remove a metric from the aggregator with the given name
        Args:
            name (str): Name of the metric
        """
        if name not in self.metrics:
            raise MetricAggregatorException(f"Metric {name} does not exist")
        self.metrics.pop(name)

    def reset(self):
        """Reset all metrics to their initial state"""
        for metric in self.metrics.values():
            metric.reset()

    def to(self, device: Union[str, torch.device] = "cpu") -> None:
        """Move all metrics to the given device
        Args:
            device (Union[str, torch.device], optional): Device to move the metrics to. Defaults to "cpu".
        """
        if self.metrics:
            for k, v in self.metrics.items():
                self.metrics[k] = v.to(device)

    @torch.no_grad()
    def compute(self) -> Dict[str, torch.Tensor]:
        """Reduce the metrics to a single value
        Returns:
            Reduced metrics
        """
        reduced_metrics = {}
        if self.metrics:
            for k, v in self.metrics.items():
                if v._update_called:
                    reduced_metrics[k] = v.compute().tolist()
        return reduced_metrics


class MovingAverageMetric(Metric):
    """Metric for tracking moving average of a value.

    Args:
        name (str): Name of the metric
        window_size (int): Window size for computing moving average
        device (str): Device to store the metric
    """

    def __init__(self, name: str, window_size: int = 100, device: str = "cpu") -> None:
        super().__init__(sync_on_compute=False)
        self.window_size = window_size
        self._values = deque(maxlen=window_size)
        self._sum = torch.tensor(0.0, device=self._device)

    def update(self, value: Union[torch.Tensor, float]) -> None:
        """Update the moving average with a new value.

        Args:
            value (Union[torch.Tensor, float]): New value to update the moving average
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        if len(self._values) == self.window_size:
            self._sum -= self._values.popleft()
        self._sum += value
        self._values.append(value)

    def compute(self) -> Dict:
        """Computes the moving average.

        Returns:
            Dict: Dictionary with the moving average
        """
        if len(self._values) == 0:
            return None
        average = self._sum / len(self._values)
        std = torch.std(torch.tensor(self._values, device=self._device))
        torch.max(torch.tensor(self._values, device=self._device))
        torch.min(torch.tensor(self._values, device=self._device))
        return average, std.item()

    def reset(self) -> None:
        """Resets the moving average."""
        self._values.clear()
        self._sum = torch.tensor(0.0, device=self._device)
