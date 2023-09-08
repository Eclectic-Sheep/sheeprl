from collections import deque
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.distributed as dist
from lightning.fabric.utilities.distributed import _distributed_available
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torchmetrics import Metric
from torchmetrics.aggregation import CatMetric


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
                reduced = v.compute()
                if v._update_called:
                    reduced_metrics[k] = reduced.tolist()
        return reduced_metrics


class IndependentMeanMetric:
    def __init__(
        self,
        names: Sequence[str],
        device: Union[str, torch.device] = "cpu",
        process_group: Optional[ProcessGroup] = None,
    ) -> None:
        """Collection of N independent mean metrics, where `N` is given by the
        length of the `names` parameter. This metric is useful when one wants to
        maintain averages of some quantities, while still being able to broadcast them
        to all the processes in a `torch.distributed` group.

        Args:
            names (Sequence[str]): the names of the metrics.
            device (Union[str, torch.device], optional): the device where the metrics reside.
                Defaults to "cpu".
            process_group (Optional[ProcessGroup], optional): the distributed process group.
                Defaults to None.
        """
        super().__init__()
        if len(names) <= 0:
            raise ValueError(f"`names` length must be greater than 0: got {len(names)}")
        self._names = names
        self._device = device
        self._metrics: dict[str, Metric] = {}
        for n in names:
            m = CatMetric(sync_on_compute=False)
            self._metrics[n] = m.to(self._device)
        self._process_group = process_group if process_group is not None else torch.distributed.group.WORLD

    def update(self, value: float, name: str) -> None:
        self._metrics[name].update(value)

    @torch.no_grad()
    def compute(self) -> Dict[str, Tensor]:
        """Compute the means, one for every metric. The metrics are first broadcasted

        Returns:
            Dict[str, Tensor]: _description_
        """
        computed_metrics = {}
        for k, v in self._metrics.items():
            computed_v = v.compute()
            if not isinstance(computed_v, Tensor):
                computed_metrics[k] = torch.tensor(computed_v, device=self._device)
            else:
                computed_metrics[k] = computed_v
        if not _distributed_available():
            return computed_metrics
        gathered_data = [None for _ in range(dist.get_world_size(self._process_group))]
        dist.all_gather_object(gathered_data, computed_metrics, group=self._process_group)
        return_data = gathered_data[0]
        for rank in range(1, len(gathered_data)):
            for k, rank_v in gathered_data[rank].items():
                if isinstance(rank_v, Tensor):
                    rank_v = torch.flatten(rank_v)
                    return_data[k] = torch.cat((return_data[k], rank_v))
        return {k: torch.mean(v) for k, v in return_data.items() if len(v)}

    def to(self, device: Union[str, torch.device] = "cpu") -> None:
        """Move all metrics to the given device

        Args:
            device (Union[str, torch.device], optional): Device to move the metrics to. Defaults to "cpu".
        """
        for k, v in self._metrics.items():
            self._metrics[k] = v.to(device)

    def reset(self) -> None:
        """Reset the internal state of the metrics"""
        for v in self._metrics.values():
            v.reset()


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
