import warnings
from math import isnan
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torchmetrics import Metric


def _distributed_available() -> bool:
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


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

    def to(self, device: Union[str, torch.device] = "cpu") -> "MetricAggregator":
        """Move all metrics to the given device
        Args:
            device (Union[str, torch.device], optional): Device to move the metrics to. Defaults to "cpu".
        """
        if not self.disabled:
            if self.metrics:
                for k, v in self.metrics.items():
                    self.metrics[k] = v.to(device)
        return self

    @torch.no_grad()
    def compute(self) -> Dict[str, List]:
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
        metrics: Union[Dict[str, Metric], MetricAggregator],
        process_group: Optional[ProcessGroup] = None,
    ) -> None:
        """Rank-independent MetricAggregator.
        This metric is useful when one wants to maintain per-rank-independent metrics of some quantities,
        while still being able to broadcast them to all the processes in a `torch.distributed` group.

        Args:
            metrics (Sequence[str]): the metrics.
            process_group (Optional[ProcessGroup], optional): the distributed process group.
                Defaults to None.
        """
        super().__init__()
        self._aggregator = metrics
        if isinstance(metrics, dict):
            self._aggregator = MetricAggregator(metrics)
        for m in self._aggregator.metrics.values():
            m._to_sync = False
            m.sync_on_compute = False
        self._process_group = process_group if process_group is not None else torch.distributed.group.WORLD
        self._distributed_available = _distributed_available()
        self._world_size = dist.get_world_size(self._process_group) if self._distributed_available else 1

    def update(self, name: str, value: Union[float, Tensor]) -> None:
        self._aggregator.update(name, value)

    @torch.no_grad()
    def compute(self) -> List[Dict[str, Tensor]]:
        """Compute the means, one for every metric. The metrics are first broadcasted

        Returns:
            List[Dict[str, List]]: the computed metrics, broadcasted from and to every processes.
            The list of the data returned is equal to the number of processes in the process group.
        """
        computed_metrics = self._aggregator.compute()
        if not self._distributed_available:
            return [computed_metrics]
        gathered_data = [None for _ in range(self._world_size)]
        dist.all_gather_object(gathered_data, computed_metrics, group=self._process_group)
        return gathered_data

    def to(self, device: Union[str, torch.device] = "cpu") -> "RankIndependentMetricAggregator":
        """Move all metrics to the given device

        Args:
            device (Union[str, torch.device], optional): Device to move the metrics to. Defaults to "cpu".
        """
        self._aggregator.to(device)
        return self

    def reset(self) -> None:
        """Reset the internal state of the metrics"""
        self._aggregator.reset()
