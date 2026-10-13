"""The aggregation of the logged metrics."""

import math

import pytest
import torch
from torchmetrics import MeanMetric, SumMetric

from sheeprl.utils.metric import MetricAggregator


@pytest.fixture(autouse=True)
def enabled_aggregator(monkeypatch):
    monkeypatch.setattr(MetricAggregator, "disabled", False)


@pytest.mark.parametrize("add", [False, True])
def test_a_nan_value_is_logged_as_nan(add):
    # The mean metrics dropped the NaN values (with a warning) and the aggregator dropped the NaN results: a diverging
    # loss vanished from the logs
    metrics = {"Loss/a": MeanMetric(), "Loss/b": SumMetric()}
    aggregator = MetricAggregator() if add else MetricAggregator(metrics)
    if add:
        for name, metric in metrics.items():
            aggregator.add(name, metric)
    for name in metrics:
        aggregator.update(name, torch.tensor(1.0))
        aggregator.update(name, torch.tensor(float("nan")))
    computed = aggregator.compute()
    assert math.isnan(computed["Loss/a"]) and math.isnan(computed["Loss/b"])


def test_a_metric_without_values_is_not_logged():
    aggregator = MetricAggregator({"Rewards/rew_avg": MeanMetric(), "Loss/a": MeanMetric()})
    aggregator.update("Loss/a", torch.tensor(2.0))
    assert aggregator.compute() == {"Loss/a": 2.0}
