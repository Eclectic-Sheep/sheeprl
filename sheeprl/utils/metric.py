from __future__ import annotations

import warnings
from math import isnan
from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import Fabric
from torch import Tensor
from torchmetrics import Metric


class MetricAggregatorException(Exception):
    """A custom exception used to report errors in use of timer class"""


class MetricAggregator:
    """A metric aggregator class to aggregate metrics to be tracked.
    Args:
        metrics (Optional[Dict[str, Metric]]): Dict of metrics to aggregate.
    """

    disabled: bool = False

    def __init__(self, metrics: Optional[Dict[str, Metric]] = None, raise_on_missing: bool = False):
        self.metrics: Dict[str, Metric] = {}
        if metrics is not None:
            self.metrics = metrics
        self._raise_on_missing = raise_on_missing

    def __iter__(self):
        return iter(self.metrics.keys())

    def add(self, name: str, metric: Metric):
        """Add a metric to the aggregator

        Args:
            name (str): Name of the metric
            metric (Metric): Metric to add.

        Raises:
            MetricAggregatorException: If the metric already exists.
        """
        if not self.disabled:
            if name not in self.metrics:
                self.metrics.setdefault(name, metric)
            else:
                if self._raise_on_missing:
                    raise MetricAggregatorException(f"Metric {name} already exists")
                else:
                    warnings.warn(
                        f"The key '{name}' is already in the metric aggregator. Nothing will be added.", UserWarning
                    )

    @torch.no_grad()
    def update(self, name: str, value: Any) -> None:
        """Update the metric with the value

        Args:
            name (str): Name of the metric
            value (Any): Value to update the metric with.

        Raises:
            MetricAggregatorException: If the metric does not exist.
        """
        if not self.disabled:
            if name not in self.metrics:
                if self._raise_on_missing:
                    raise MetricAggregatorException(f"Metric {name} does not exist")
                else:
                    warnings.warn(
                        f"The key '{name}' is missing from the metric aggregator. Nothing will be added.", UserWarning
                    )
            else:
                self.metrics[name].update(value)

    def pop(self, name: str) -> None:
        """Remove a metric from the aggregator with the given name
        Args:
            name (str): Name of the metric
        """
        if not self.disabled:
            if name not in self.metrics:
                if self._raise_on_missing:
                    raise MetricAggregatorException(f"Metric {name} does not exist")
                else:
                    warnings.warn(
                        f"The key '{name}' is missing from the metric aggregator. Nothing will be popped.", UserWarning
                    )
            self.metrics.pop(name, None)

    def reset(self):
        """Reset all metrics to their initial state"""
        if not self.disabled:
            for metric in self.metrics.values():
                metric.reset()

    def to(self, device: str | torch.device = "cpu") -> "MetricAggregator":
        """Move all metrics to the given device
        Args:
            device (str |torch.device, optional): Device to move the metrics to. Defaults to "cpu".
        """
        if not self.disabled:
            if self.metrics:
                for k, v in self.metrics.items():
                    self.metrics[k] = v.to(device)
        return self

    @torch.no_grad()
    def compute(self) -> Dict[str, Any]:
        """Reduce the metrics to a single value
        Returns:
            Reduced metrics
        """
        reduced_metrics = {}
        if not self.disabled:
            if self.metrics:
                for k, v in self.metrics.items():
                    reduced = v.compute()
                    is_tensor = torch.is_tensor(reduced)
                    if is_tensor and reduced.numel() == 1:
                        reduced_metrics[k] = reduced.item()
                    else:
                        if not is_tensor:
                            warnings.warn(
                                f"The reduced metric {k} is not a scalar tensor: type={type(reduced)}. "
                                "This may create problems during the logging phase.",
                                category=RuntimeWarning,
                            )
                        else:
                            warnings.warn(
                                f"The reduced metric {k} is not a scalar: size={v.size()}. "
                                "This may create problems during the logging phase.",
                                category=RuntimeWarning,
                            )
                        reduced_metrics[k] = reduced

                    is_tensor = torch.is_tensor(reduced_metrics[k])
                    if (is_tensor and torch.isnan(reduced_metrics[k]).any()) or (
                        not is_tensor and isnan(reduced_metrics[k])
                    ):
                        reduced_metrics.pop(k, None)
        return reduced_metrics


class RankIndependentMetricAggregator:
    def __init__(
        self,
        fabric: Fabric,
        metrics: Dict[str, Metric] | MetricAggregator,
    ) -> None:
        """This metric is useful when one wants to maintain per-rank-independent metrics of some quantities,
        while still being able to broadcast them to all the processes in a `torch.distributed` group. Internally,
        this metric uses a `MetricAggregator` to keep track of the metrics, and then broadcasts the metrics
        to all the processes thanks to Fabric.

        Args:
            fabric (Fabric): the fabric object.
            metrics (Sequence[str]): the metrics.
        """
        super().__init__()
        self._fabric = fabric
        self._aggregator = metrics
        if isinstance(metrics, dict):
            self._aggregator = MetricAggregator(metrics)
        for m in self._aggregator.metrics.values():
            m._to_sync = False
            m.sync_on_compute = False

    def update(self, name: str, value: float | Tensor) -> None:
        self._aggregator.update(name, value)

    @torch.no_grad()
    def compute(self) -> Tensor | Dict | List | Tuple:
        """Compute the means, one for every metric. The metrics are first broadcasted

        Returns:
            the computed metrics, broadcasted from and to every processes.
        """
        computed_metrics = self._aggregator.compute()
        gathered_data = self._fabric.all_gather(computed_metrics)
        return gathered_data

    def to(self, device: str | torch.device = "cpu") -> "RankIndependentMetricAggregator":
        """Move all metrics to the given device

        Args:
            device (str |torch.device, optional): Device to move the metrics to. Defaults to "cpu".
        """
        self._aggregator.to(device)
        return self

    def reset(self) -> None:
        """Reset the internal state of the metrics"""
        self._aggregator.reset()
