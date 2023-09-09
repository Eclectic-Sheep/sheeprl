from collections import deque
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from lightning.fabric.utilities.distributed import _distributed_available
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torchmetrics import Metric
from torchmetrics.wrappers.running import Running


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
    def compute(self) -> Dict[str, List]:
        """Reduce the metrics to a single value
        Returns:
            Reduced metrics
        """
        reduced_metrics = {}
        if self.metrics:
            for k, v in self.metrics.items():
                reduced: Tensor = v.compute()
                if v.update_called or isinstance(v, Running):
                    reduced_metrics[k] = reduced.tolist()
        return reduced_metrics


class RankIndependentMetricAggregator:
    def __init__(
        self,
        metrics: Union[MetricAggregator, Dict[str, Metric]],
        process_group: Optional[ProcessGroup] = None,
    ) -> None:
        """Rank-independent MetricAggregator.
        This metric is useful when one wants to maintain per-rank-independent metrics of some quantities,
        while still being able to broadcast them to all the processes in a `torch.distributed` group.

        Args:
            metrics (Union[MetricAggregator, Dict[str, Metric]]): the metrics to be aggregated.
                If a dictionary of metrics is passed, then the aggregator is constructed from it.
            process_group (Optional[ProcessGroup], optional): the distributed process group.
                Defaults to None.
        """
        super().__init__()
        if isinstance(metrics, dict):
            aggregator = MetricAggregator(metrics)
        self._aggregator: MetricAggregator = aggregator
        for m in aggregator.metrics.values():
            m.sync_on_compute = False
        self._process_group = process_group if process_group is not None else torch.distributed.group.WORLD

    def update(self, key: str, value: Union[float, Tensor]) -> None:
        """Update the metric specified by `name` with `value`

        Args:
            key (str): the name of the metric to be updated.
            value (Union[float, Tensor]): value to update the metric with.
        """
        self._aggregator.update(key, value)

    @torch.no_grad()
    def compute(self) -> List[Dict[str, List]]:
        """Compute the metric independently for every rank and broadcast the result to all
        the processes in the process group.

        Returns:
            List[Dict[str, List]]: the computed metrics, broadcasted from and to every processes.
            The list of the data returned is equal to the number of processes in the process group.
        """
        computed_metrics = self._aggregator.compute()
        if not _distributed_available():
            return [computed_metrics]
        gathered_data = [None for _ in range(dist.get_world_size(self._process_group))]
        dist.all_gather_object(gathered_data, computed_metrics, group=self._process_group)
        return gathered_data

    def to(self, device: Union[str, torch.device] = "cpu") -> None:
        """Move all metrics to the given device

        Args:
            device (Union[str, torch.device], optional): Device to move the metrics to. Defaults to "cpu".
        """
        self._aggregator.to(device)

    def reset(self) -> None:
        """Reset the internal state of the metrics"""
        self._aggregator.reset()


class MovingAverageMetric(Metric):
    """Metric for tracking moving average of a value.

    Args:
        name (str): Name of the metric
        window (int): Window size for computing moving average
        device (str): Device to store the metric
    """

    sum_value: Tensor

    def __init__(self, window: int = 100, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._window = window
        self._values = deque(maxlen=window)
        self.add_state("sum_value", torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=False))

    def update(self, value: Union[torch.Tensor, float]) -> None:
        """Update the moving average with a new value.

        Args:
            value (Union[torch.Tensor, float]): New value to update the moving average.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        if len(self._values) == self._window:
            self.sum_value -= self._values.popleft()
        self.sum_value += value
        self._values.append(value)

    def compute(self) -> Tensor:
        """Computes the moving average.

        Returns:
            Tensor: the moving average
        """
        if len(self._values) == 0:
            return torch.nan
        average = self.sum_value / len(self._values)
        return average

    def reset(self) -> None:
        """Resets the moving average."""
        super().reset()
        self._values.clear()
